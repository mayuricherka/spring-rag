"""Streamlit AppTest with real MCP transport and synthetic service fixtures only."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from streamlit.testing.v1 import AppTest

from week3.memory_ledger import FACTS, Scope
from week3.mock_mem0 import Mem0Fixture
from week3.ui_backend import UIBackend, ui_scope


APP = Path(__file__).resolve().parent / "streamlit_app.py"
ADD = "/v3/memories/add/"


class StreamlitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        env = patch.dict(os.environ, {"SPRING_RAG_UI_STATE_DIR": str(self.path),
                                     "SPRING_RAG_UI_MOCK_FAILURE": "",
                                     "SPRING_RAG_UI_MOCK_MEMORY_SCENARIO": ""})
        env.start()
        self.addCleanup(env.stop)
        # Inject synthetic services in tests only. The real UI always asks for live scope.
        self.synthetic_services = True
        def test_scope(mode, secrets_getter):
            self.assertEqual(mode, "live")
            if self.synthetic_services:
                return Scope()
            return ui_scope(mode, secrets_getter)
        scope_patch = patch("week3.ui_backend.ui_scope", side_effect=test_scope)
        scope_patch.start()
        self.addCleanup(scope_patch.stop)
        # AppTest supplies a nonempty in-memory secrets fixture. These guards
        # also prevent any accidental real credential-file access during tests.
        guard = patch("streamlit.runtime.secrets.Secrets._parse_file_path",
                      side_effect=AssertionError("Credential-file parsing is forbidden during UI tests"))
        self.secret_files = guard.start()
        self.addCleanup(guard.stop)

    def tearDown(self):
        self.secret_files.assert_not_called()

    def app(self, query=None):
        app = AppTest.from_file(APP, default_timeout=30)
        app.secrets = {"synthetic_fixture_only": "no credential file"}
        if query:
            app.query_params.update(query)
        app.run()
        self.assertEqual(len(app.exception), 0)
        return app

    def button(self, app, label):
        return next(button for button in app.button if button.label == label)

    def click(self, app, label):
        self.button(app, label).click().run()
        self.assertEqual(len(app.exception), 0)
        return app

    def text(self, app):
        return "\n".join(str(item.value) for name in ("markdown", "caption", "code", "info", "warning", "error", "success")
                         for item in getattr(app, name))

    def add_count(self):
        path = self.path / "fixture-remote.sqlite3"
        return Mem0Fixture(path).calls(ADD) if path.exists() else 0

    def calls(self):
        path = self.path / "fixture-workflow-calls.jsonl"
        return path.read_text() if path.exists() else ""

    def test_question_answer_sources_and_completed_result_survive_rerun_and_browser_refresh(self):
        app = self.app()
        app.text_area[0].input("How do I enable caching?")
        self.click(app, "Ask")
        self.assertIn("This synthetic passage exercises the MCP workflow [1].", self.text(app))
        self.assertIn("Synthetic passage for an MCP transport test", self.text(app))
        self.assertTrue(any(item.value == "Source passages" for item in app.subheader))
        before = self.calls()
        app.run()
        self.click(app, "Ask")  # Duplicate submission must use the completed run.
        refreshed = self.app(dict(app.query_params))  # New Streamlit session state, same URL.
        self.assertIn("This synthetic passage exercises", self.text(refreshed))
        self.assertEqual(refreshed.text_area[0].value, "How do I enable caching?")
        self.assertEqual(self.calls(), before)

    def test_pending_preview_survives_refresh_rejection_sends_no_service_request(self):
        app = self.app()
        self.click(app, "Preview memory")
        self.assertIn(FACTS["java_comparisons"], self.text(app))
        refreshed = self.app(dict(app.query_params))
        self.assertIn("Status: proposed", self.text(refreshed))
        self.click(refreshed, "Reject")
        self.assertIn("Proposal rejected", self.text(refreshed))
        self.assertFalse((self.path / "fixture-remote.sqlite3").exists())
        another = self.app(dict(refreshed.query_params))
        self.assertIn("Status: rejected", self.text(another))
        self.assertTrue(self.button(another, "Approve").disabled)

    def test_approve_double_click_and_refresh_make_one_add_then_new_session_recalls(self):
        app = self.app()
        app.text_area[0].input("How do I enable caching?")
        self.click(app, "Ask")
        self.click(app, "Preview memory")
        self.click(app, "Approve")
        self.assertIn("Save completed", self.text(app))
        self.assertEqual(self.add_count(), 1)
        self.assertTrue(self.button(app, "Approve").disabled)
        # A queued event must also be harmless at the backend boundary.
        backend = UIBackend(Scope(), self.path)
        view_id = app.query_params["view"]
        operation_id = backend.proposal(view_id)["operation_id"]
        backend.approve(view_id, operation_id)
        app.run()
        refreshed = self.app(dict(app.query_params))
        self.assertIn("Save completed", self.text(refreshed))
        self.assertEqual(self.add_count(), 1)
        old_view = dict(refreshed.query_params)
        self.click(refreshed, "New session and recall")
        self.assertNotEqual(dict(refreshed.query_params), old_view)
        self.assertIn(FACTS["java_comparisons"], self.text(refreshed))
        self.assertEqual(refreshed.text_area[0].value, "")
        self.assertFalse(any(item.value == "Answer" for item in refreshed.subheader))
        self.assertTrue(any(item.value == "Recalled project memory" for item in refreshed.subheader))
        count = Mem0Fixture(self.path / "fixture-remote.sqlite3").calls("/v3/memories/search/")
        refreshed.run()
        again = self.app(dict(refreshed.query_params))
        self.assertIn(FACTS["java_comparisons"], self.text(again))
        self.assertEqual(Mem0Fixture(self.path / "fixture-remote.sqlite3").calls("/v3/memories/search/"), count)
        self.assertEqual(self.add_count(), 1)

    def test_uncertain_save_blocks_recall_then_read_only_check_confirms_without_second_add(self):
        os.environ["SPRING_RAG_UI_MOCK_MEMORY_SCENARIO"] = "timeout_after"
        app = self.app()
        self.click(app, "Preview memory")
        self.click(app, "Approve")
        self.assertIn("Status: uncertain", self.text(app))
        self.assertNotIn("synthetic-secret", self.text(app))
        before = dict(app.query_params)
        self.click(app, "New session and recall")
        self.assertEqual(dict(app.query_params), before)
        self.assertIn("Complete or reject", self.text(app))
        refreshed = self.app(before)
        self.assertIn("Status: uncertain", self.text(refreshed))
        self.assertEqual(self.add_count(), 1)
        os.environ["SPRING_RAG_UI_MOCK_MEMORY_SCENARIO"] = "success"
        self.click(refreshed, "Check save status")
        self.assertIn("Save completed", self.text(refreshed))
        self.assertEqual(self.add_count(), 1)

    def test_tool_failure_resume_and_rerun_do_not_repeat_completed_recall(self):
        os.environ["SPRING_RAG_UI_MOCK_FAILURE"] = "docs"
        app = self.app()
        app.text_area[0].input("How do I enable caching?")
        self.click(app, "Ask")
        self.assertIn("unfinished steps", self.text(app))
        before = self.calls()
        app.run()
        self.assertEqual(self.calls(), before)
        os.environ["SPRING_RAG_UI_MOCK_FAILURE"] = ""
        self.click(app, "Resume answer")
        self.assertIn("This synthetic passage exercises", self.text(app))
        operations = [json.loads(line)["operation"] for line in self.calls().splitlines()]
        self.assertEqual(operations.count("recall_project_facts"), 1)
        self.assertEqual(operations.count("search_caching_docs"), 2)

    def test_fixture_backed_ui_has_no_key_fields_and_never_loads_credentials(self):
        with patch("week3.ui_backend.load_credentials", side_effect=AssertionError("Must not load keys in mock mode")):
            app = self.app()
            app.text_area[0].input("How do I enable caching?")
            self.click(app, "Ask")
            self.click(app, "Preview memory")
            self.click(app, "Reject")
        self.assertEqual(len(app.text_input), 0)
        self.assertEqual([item.label for item in app.text_area], ["Question"])

    def test_live_is_the_only_ui_mode_even_with_an_old_mock_url(self):
        self.synthetic_services = False
        os.environ.update({"MEM0_EXPECTED_ORG": "org_Synthetic", "MEM0_EXPECTED_PROJECT": "proj_Synthetic"})
        with patch("week3.ui_backend.load_credentials", side_effect=AssertionError("No action approved")) as keys:
            app = self.app({"mode": "mock"})
            app.run()
            keys.assert_not_called()
        self.assertEqual(len(app.radio), 0)
        self.assertEqual(app.query_params["mode"], "live")
        self.assertIn("Live services: Nebius, Pinecone and hosted Mem0", self.text(app))
        self.assertNotIn("Synthetic demo", self.text(app))
        scope = Scope(mode="hosted_mem0", cloud_org="org_Synthetic", cloud_project="proj_Synthetic")
        view = UIBackend(scope, self.path).store.get(app.query_params["view"])
        self.assertIsNone(view["thread_id"])
        self.assertFalse((self.path / "fixture-remote.sqlite3").exists())

    def test_live_setup_names_missing_or_invalid_settings_without_exposing_values(self):
        self.synthetic_services = False
        sentinel = "synthetic-private-setting-must-not-appear"
        cases = (
            ("", "", "Missing server setting: MEM0_EXPECTED_ORG"),
            ("org_Synthetic", "", "Missing server setting: MEM0_EXPECTED_PROJECT"),
            (sentinel, "proj_Synthetic", "Invalid server setting: MEM0_EXPECTED_ORG"),
            ("org_Synthetic", sentinel, "Invalid server setting: MEM0_EXPECTED_PROJECT"),
        )
        with patch("week3.ui_backend.load_credentials", side_effect=AssertionError("No action approved")) as keys:
            for org, project, message in cases:
                with self.subTest(message=message), patch.dict(os.environ, {
                    "MEM0_EXPECTED_ORG": org, "MEM0_EXPECTED_PROJECT": project, "MEM0_API_KEY": sentinel,
                }):
                    app = self.app()
                    self.assertIn(message, self.text(app))
                    self.assertIn(".streamlit/secrets.toml", self.text(app))
                    self.assertNotIn(sentinel, self.text(app))
                    self.assertEqual(len(app.text_area), 0)
                    self.assertNotIn("view", app.query_params)
            keys.assert_not_called()
        self.assertFalse((self.path / "ui-sessions.sqlite3").exists())

    def test_unexpected_setup_failure_keeps_raw_errors_hidden(self):
        sentinel = "synthetic-private-error-must-not-appear"
        with patch("week3.ui_backend.UIBackend", side_effect=RuntimeError(sentinel)):
            app = self.app()
        self.assertIn("access to the local state directory", self.text(app))
        self.assertNotIn(sentinel, self.text(app))

    def test_live_keys_load_only_on_approved_action_and_never_appear_in_ui_state_logs_or_storage(self):
        self.synthetic_services = False
        sentinel = "synthetic-ui-credential-must-not-appear"
        os.environ.update({"MEM0_EXPECTED_ORG": "org_Synthetic", "MEM0_EXPECTED_PROJECT": "proj_Synthetic",
                           "MEM0_API_KEY": sentinel, "NEBIUS_API_KEY": sentinel, "PINECONE_API_KEY": sentinel})
        output = io.StringIO()
        with patch("week3.ui_backend.call_memory", new_callable=AsyncMock,
                   side_effect=RuntimeError(sentinel)) as remote, redirect_stdout(output), redirect_stderr(output):
            app = self.app({"mode": "live"})
            app.run()
            remote.assert_not_called()
            self.click(app, "New session and recall")  # Missing human service approval.
            remote.assert_not_called()
            app.checkbox[0].check().run()
            self.click(app, "New session and recall")
            self.assertEqual(remote.await_count, 1)
            self.assertEqual(remote.call_args.kwargs["credentials"], {"MEM0_API_KEY": sentinel})
            self.assertNotIn(sentinel, self.text(app))
            self.assertNotIn(sentinel, json.dumps(app.session_state.to_dict()))
            self.assertEqual(len(app.text_input), 0)
        self.assertNotIn(sentinel, output.getvalue())
        for path in self.path.glob("*.sqlite3*"):
            self.assertNotIn(sentinel.encode(), path.read_bytes())


if __name__ == "__main__":
    unittest.main()
