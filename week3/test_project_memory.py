"""Offline memory lifecycle and exact hosted SDK HTTP contract tests."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from dataclasses import replace
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import httpx

from week3.mem0_hosted import live_backend, sdk_backend
from week3.memory_client import human_decision, main
from week3.memory_ledger import FACTS, Ledger, MemorySafetyError, Scope
from week3.mock_mem0 import Mem0Fixture, fixture_backend
from week3.project_memory import ProjectMemory, error_diagnostic


ADD = "/v3/memories/add/"


class ProjectMemoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.scope = Scope()
        self.ledger = Ledger(self.path / "operations.sqlite3", self.scope)
        self.fixture = Mem0Fixture(self.path / "fixture-remote.sqlite3")
        self.scenario = "success"
        self.service = ProjectMemory(self.ledger, lambda: fixture_backend(self.scope, self.path, self.scenario))

    def propose(self, key="java_comparisons", approve=None):
        row = self.ledger.propose(key)
        if approve is not None:
            row = self.ledger.decide(row["operation_id"], approve=approve)
        return row

    def test_exact_human_preview_and_approval(self):
        row = self.propose()
        with patch("sys.stdin.isatty", return_value=True), patch("builtins.input", return_value="APPROVE"), patch("sys.stdout", new_callable=io.StringIO) as output:
            approved = human_decision(self.ledger, row)
        self.assertIn(FACTS["java_comparisons"], output.getvalue())
        self.assertIn(self.scope.project_id, output.getvalue())
        self.assertEqual(approved["state"], "approved")
        self.assertEqual(self.fixture.calls(ADD), 0)

    def test_rejection_and_unapproved_calls_do_not_initialize_sdk(self):
        for approve in (None, False):
            row = self.propose("redis" if approve is False else "java_comparisons", approve)
            with patch("week3.mem0_hosted.sdk_backend", side_effect=AssertionError("Must not initialize")):
                result = self.service.remember(row["operation_id"])
                replay = self.service.remember(row["operation_id"])
            self.assertEqual(result, replay)
            self.assertEqual(result["status"], "rejected" if approve is False else "proposed")
        self.assertEqual(self.fixture.calls(ADD), 0)

    def test_noninteractive_input_cannot_approve(self):
        row = self.propose()
        with patch("sys.stdin.isatty", return_value=False), patch("sys.stdout", new_callable=io.StringIO):
            with self.assertRaises(MemorySafetyError):
                human_decision(self.ledger, row)
        self.assertEqual(self.ledger.get(row["operation_id"])["state"], "proposed")

    def test_completed_save_replay_and_duplicate_proposal(self):
        row = self.propose(approve=True)
        result = self.service.remember(row["operation_id"])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(self.propose()["operation_id"], row["operation_id"])
        self.assertEqual(self.service.remember(row["operation_id"]), result)
        self.assertEqual(self.fixture.calls(ADD), 1)
        self.assertEqual(self.service.recall()["facts"][0]["fact"], row["fact"])

    def test_scope_and_destination_bound_to_approval(self):
        row = self.propose(approve=True)
        for scope in (replace(self.scope, project_id="another-synthetic-project"),
                      replace(self.scope, user_id="another-synthetic-user"),
                      replace(self.scope, mode="hosted_mem0", cloud_org="o", cloud_project="p")):
            other = Ledger(self.path / "operations.sqlite3", scope)
            with self.assertRaises(MemorySafetyError):
                other.get(row["operation_id"])
            self.assertNotEqual(other.propose("java_comparisons")["operation_id"], row["operation_id"])

    def test_wrong_project_remote_results_are_not_returned(self):
        row = self.propose(approve=True)
        self.service.remember(row["operation_id"])
        other = replace(self.scope, project_id="another-synthetic-project")
        service = ProjectMemory(Ledger(self.path / "operations.sqlite3", other),
                                lambda: fixture_backend(other, self.path, "wrong_scope"))
        self.assertEqual(service.recall()["facts"], [])

    def test_timeout_after_commit_reconciles_without_second_add(self):
        row = self.propose(approve=True)
        self.scenario = "timeout_after"
        self.assertEqual(self.service.remember(row["operation_id"])["status"], "uncertain")
        # A fresh service/ledger simulates a restart; only scoped reads are allowed.
        fresh = ProjectMemory(Ledger(self.path / "operations.sqlite3", self.scope),
                              lambda: fixture_backend(self.scope, self.path))
        self.assertEqual(fresh.remember(row["operation_id"])["status"], "completed")
        self.assertEqual(self.fixture.calls(ADD), 1)

    def test_provider_added_metadata_allows_read_only_reconciliation_and_recall(self):
        row = self.propose(approve=True)
        self.scenario = "timeout_after"
        self.assertEqual(self.service.remember(row["operation_id"])["status"], "uncertain")
        item = self.fixture.items()[0]
        item["metadata"]["provider_annotation"] = {"value": "synthetic-extra-must-not-be-returned"}
        with closing(sqlite3.connect(self.fixture.path)) as db, db:
            db.execute("UPDATE facts SET body=?", (json.dumps(item),))
        self.scenario = "success"
        self.assertEqual(self.service.remember(row["operation_id"])["status"], "completed")
        result = self.service.recall()
        self.assertEqual(result["facts"][0]["fact"], row["fact"])
        self.assertNotIn("synthetic-extra", json.dumps(result))
        self.assertEqual(self.fixture.calls(ADD), 1)

    def test_missing_or_changed_required_metadata_blocks_completion_and_recall(self):
        row = self.propose(approve=True)
        self.scenario = "timeout_after"
        self.service.remember(row["operation_id"])
        self.scenario = "success"
        item = self.fixture.items()[0]
        original = self.service.metadata(row)
        for field in original:
            for alteration in ("missing", "changed"):
                with self.subTest(field=field, alteration=alteration):
                    metadata = {**original, "provider_annotation": "synthetic-extra"}
                    if alteration == "missing":
                        del metadata[field]
                    else:
                        metadata[field] = "synthetic-wrong-value"
                    candidate = {**item, "metadata": metadata}
                    self.assertFalse(self.service.matches(candidate, row))
                    with closing(sqlite3.connect(self.fixture.path)) as db, db:
                        db.execute("UPDATE facts SET body=?", (json.dumps(candidate),))
                    result = self.service.remember(row["operation_id"])
                    self.assertEqual(result["status"], "uncertain")
                    if field != "operation_id":
                        self.assertEqual(result["diagnostic"]["metadata_field_match_counts"][field], 0)
                    self.assertEqual(self.service.recall()["facts"], [])
        # Do not use Python's broad True == 1 equality as a marker check.
        self.assertFalse(self.service.matches({**item, "metadata": {**original, "synthetic": 1}}, row))
        self.assertEqual(self.fixture.calls(ADD), 1)

    def test_literal_true_text_marker_reconciles_without_another_save(self):
        row = self.propose(approve=True)
        self.scenario = "timeout_after"
        self.service.remember(row["operation_id"])
        item = self.fixture.items()[0]
        item["metadata"]["synthetic"] = "true"
        with closing(sqlite3.connect(self.fixture.path)) as db, db:
            db.execute("UPDATE facts SET body=?", (json.dumps(item),))
        self.scenario = "success"
        result = self.service.remember(row["operation_id"])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(self.service.recall()["facts"][0]["fact"], row["fact"])
        for value in (True, "true", "True", "TRUE"):
            self.assertTrue(self.service.matches({**item, "metadata": {**item["metadata"], "synthetic": value}}, row))
        self.assertEqual(self.fixture.calls(ADD), 1)

    def test_false_missing_and_unknown_markers_stay_uncertain_without_leaking_values(self):
        row = self.propose(approve=True)
        self.scenario = "timeout_after"
        self.service.remember(row["operation_id"])
        self.scenario = "success"
        item = self.fixture.items()[0]
        for value, category in ((False, "boolean_false"), ("false", "text_false"),
                                (None, "null"), (0, "integer_zero"), (1, "integer_one"),
                                ("synthetic-secret-must-not-leak", "other")):
            with self.subTest(category=category):
                candidate = {**item, "metadata": {**item["metadata"], "synthetic": value}}
                with closing(sqlite3.connect(self.fixture.path)) as db, db:
                    db.execute("UPDATE facts SET body=?", (json.dumps(candidate),))
                result = self.service.remember(row["operation_id"])
                self.assertEqual(result["status"], "uncertain")
                self.assertEqual(result["diagnostic"]["synthetic_marker_forms"], {category: 1})
                self.assertNotIn("synthetic-secret", json.dumps(result))
                self.assertEqual(self.service.recall()["facts"], [])
        del item["metadata"]["synthetic"]
        with closing(sqlite3.connect(self.fixture.path)) as db, db:
            db.execute("UPDATE facts SET body=?", (json.dumps(item),))
        result = self.service.remember(row["operation_id"])
        self.assertEqual(result["status"], "uncertain")
        self.assertEqual(result["diagnostic"]["synthetic_marker_forms"], {"missing": 1})
        self.assertEqual(self.fixture.calls(ADD), 1)

    def test_timeout_without_visible_result_is_not_retried(self):
        row = self.propose(approve=True)
        self.scenario = "timeout_before"
        for _ in range(3):
            self.assertEqual(self.service.remember(row["operation_id"])["status"], "uncertain")
        self.assertEqual(self.fixture.calls(ADD), 1)
        self.assertEqual(self.service.recall()["facts"], [])

    def test_pending_event_blocks_recall_until_confirmed(self):
        row = self.propose(approve=True)
        self.scenario = "pending"
        self.assertEqual(self.service.remember(row["operation_id"])["status"], "pending_remote")
        self.assertEqual(self.service.recall()["facts"], [])
        self.scenario = "completed_event"
        self.assertEqual(self.service.remember(row["operation_id"])["status"], "completed")
        self.assertEqual(self.fixture.calls(ADD), 1)
        self.assertEqual(len(self.service.recall()["facts"]), 1)

    def test_failed_event_never_retries_add(self):
        row = self.propose(approve=True)
        self.scenario = "failed_event"
        for _ in range(2):
            self.assertEqual(self.service.remember(row["operation_id"])["status"], "failed")
        self.assertEqual(self.fixture.calls(ADD), 1)

    def test_crash_after_claim_only_reconciles(self):
        row = self.propose(approve=True)
        self.assertTrue(self.ledger.claim(row["operation_id"]))
        self.assertEqual(self.service.remember(row["operation_id"])["status"], "uncertain")
        self.assertEqual(self.fixture.calls(ADD), 0)

    def test_atomic_claim_across_concurrent_connections(self):
        row = self.propose(approve=True)
        ledgers = [Ledger(self.path / "operations.sqlite3", self.scope) for _ in range(6)]
        with ThreadPoolExecutor(max_workers=6) as pool:
            claims = list(pool.map(lambda ledger: ledger.claim(row["operation_id"]), ledgers))
        self.assertEqual(sum(claims), 1)

    def test_catalog_rejects_private_text_and_recall_free_text(self):
        with self.assertRaises(MemorySafetyError):
            self.ledger.propose("arbitrary config or credential")
        with self.assertRaises(MemorySafetyError):
            self.service.recall("arbitrary config or credential")

    def test_sdk_http_contract_and_scoping(self):
        requests = []
        fixture = self.fixture
        def handle(request):
            requests.append((request.url.path, json.loads(request.content) if request.content else {}))
            return fixture(request)
        @contextmanager
        def factory():
            with sdk_backend(self.scope, self.path, api_key="synthetic-not-a-real-key",
                             transport=httpx.MockTransport(handle)) as backend:
                yield backend
        service = ProjectMemory(self.ledger, factory)
        row = self.propose(approve=True)
        self.assertEqual(service.remember(row["operation_id"])["status"], "completed")
        service.recall()
        add = next(body for path, body in requests if path == ADD)
        self.assertIs(add["infer"], False)
        self.assertEqual(add["messages"], [{"role": "user", "content": row["fact"]}])
        self.assertEqual((add["user_id"], add["app_id"]), (self.scope.user_id, self.scope.project_id))
        for path, body in requests:
            if path in {"/v3/memories/", "/v3/memories/search/"}:
                self.assertEqual(body["filters"], self.scope.filters)
                self.assertNotIn("user_id", body)
        from mem0.memory.telemetry import client_telemetry
        self.assertIsNone(client_telemetry.posthog)

    def test_live_gate_before_sdk_initialization(self):
        with patch.dict("os.environ", {}, clear=True), patch("week3.mem0_hosted.sdk_backend") as sdk:
            with self.assertRaises(MemorySafetyError):
                with live_backend(self.scope, self.path):
                    pass
            sdk.assert_not_called()

    def test_wrong_cloud_project_stops_add(self):
        scope = replace(self.scope, mode="hosted_mem0", cloud_org="expected-org", cloud_project="expected-project")
        ledger = Ledger(self.path / "operations.sqlite3", scope)
        row = ledger.propose("java_comparisons")
        ledger.decide(row["operation_id"], approve=True)
        service = ProjectMemory(ledger, lambda: fixture_backend(scope, self.path))
        with self.assertRaises(MemorySafetyError):
            service.remember(row["operation_id"])
        self.assertEqual(ledger.get(row["operation_id"])["state"], "approved")
        self.assertEqual(self.fixture.calls(ADD), 0)

    def test_expected_cloud_destination_can_save_through_mocked_hosted_sdk(self):
        scope = replace(self.scope, mode="hosted_mem0", cloud_org="fixture-org", cloud_project="fixture-project")
        ledger = Ledger(self.path / "operations.sqlite3", scope)
        row = ledger.propose("java_comparisons")
        ledger.decide(row["operation_id"], approve=True)
        service = ProjectMemory(ledger, lambda: fixture_backend(scope, self.path))
        self.assertEqual(service.remember(row["operation_id"])["status"], "completed")
        self.assertEqual(self.fixture.calls(ADD), 1)

    def test_altered_remote_fact_does_not_establish_completion(self):
        row = self.propose(approve=True)
        self.scenario = "timeout_after"
        self.service.remember(row["operation_id"])
        item = self.fixture.items()[0]
        item["memory"] = "Different unapproved content"
        import sqlite3
        with closing(sqlite3.connect(self.fixture.path)) as db, db:
            db.execute("UPDATE facts SET body=?", (json.dumps(item),))
        self.scenario = "success"
        self.assertEqual(self.service.remember(row["operation_id"])["status"], "uncertain")
        self.assertEqual(self.service.recall()["facts"], [])
        self.assertEqual(self.fixture.calls(ADD), 1)

    def test_client_rejection_never_starts_mcp(self):
        with patch("sys.stdin.isatty", return_value=True), patch("builtins.input", return_value="REJECT"), patch("week3.memory_client.call_memory") as call, patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(main(["--mock", "--state-dir", str(self.path), "propose", "java_comparisons"]), 0)
        call.assert_not_called()
        self.assertEqual(self.propose()["state"], "rejected")

    def test_client_recall_waits_for_completed_save(self):
        row = self.propose(approve=True)
        with patch("week3.memory_client.call_memory") as call, patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(main(["--mock", "--state-dir", str(self.path), "recall", "--after", row["operation_id"]]), 1)
        call.assert_not_called()

    def test_live_cost_rejection_never_starts_mcp_or_reads_key(self):
        with patch("sys.stdin.isatty", return_value=True), patch("builtins.input", side_effect=["APPROVE", "CANCEL"]), patch("week3.memory_client.getpass") as key, patch("week3.memory_client.call_memory") as call, patch("sys.stdout", new_callable=io.StringIO):
            result = main(["--live", "--state-dir", str(self.path), "--cloud-org", "synthetic-org",
                           "--cloud-project", "synthetic-project", "--limits-checked",
                           "propose", "java_comparisons"])
        self.assertEqual(result, 0)
        call.assert_not_called()
        key.assert_not_called()

    def test_unchecked_account_limits_block_live_requests(self):
        with patch("week3.memory_client.call_memory") as call, patch("week3.memory_client.getpass") as key, patch("sys.stdout", new_callable=io.StringIO):
            result = main(["--live", "--state-dir", str(self.path), "--cloud-org", "synthetic-org",
                           "--cloud-project", "synthetic-project", "recall"])
        self.assertEqual(result, 1)
        call.assert_not_called()
        key.assert_not_called()

    def test_diagnostic_explains_empty_readback_without_retrying_write(self):
        row = self.propose(approve=True)
        self.ledger.claim(row["operation_id"])
        result = self.service.remember(row["operation_id"])
        self.assertEqual(result["diagnostic"], {
            "stage": "read_back", "code": "NO_EXACT_MATCH", "returned_count": 0,
            "same_operation_count": 0, "exact_text_count": 0, "exact_metadata_count": 0,
        })
        self.assertEqual(self.fixture.calls(ADD), 0)

    def test_event_http_failure_is_sanitized_and_does_not_retry_write(self):
        row = self.propose(approve=True)
        self.scenario = "event_error"
        for _ in range(2):
            result = self.service.remember(row["operation_id"])
            self.assertEqual(result["status"], "uncertain")
            self.assertEqual(result["diagnostic"], {"stage": "event_status", "code": "HTTP_ERROR", "http_status": 503})
            self.assertNotIn("synthetic-secret", json.dumps(result))
        self.assertEqual(self.fixture.calls(ADD), 1)

    def test_sdk_wrapped_http_error_exposes_only_safe_category(self):
        row = self.propose(approve=True)
        self.ledger.claim(row["operation_id"])
        self.scenario = "read_error"
        result = self.service.remember(row["operation_id"])
        self.assertEqual(result["diagnostic"], {"stage": "read_back", "code": "HTTP_ERROR", "http_status": 503})
        self.assertNotIn("synthetic-secret", json.dumps(result))
        self.assertEqual(self.fixture.calls(ADD), 0)

    def test_unknown_event_status_does_not_echo_remote_text(self):
        row = self.propose(approve=True)
        self.scenario = "unknown_event"
        result = self.service.remember(row["operation_id"])
        self.assertEqual(result["diagnostic"], {"stage": "event_status", "code": "EVENT_STATUS_UNRECOGNIZED"})
        self.assertNotIn("synthetic-secret", json.dumps(result))

    def test_exception_diagnostic_drops_arbitrary_debug_info(self):
        error = RuntimeError("synthetic-secret-must-not-leak")
        error.debug_info = {"status_code": 429, "url": "synthetic-secret-must-not-leak", "response": "private config"}
        self.assertEqual(error_diagnostic(error, "event_status"),
                         {"stage": "event_status", "code": "HTTP_ERROR", "http_status": 429})


if __name__ == "__main__":
    unittest.main()
