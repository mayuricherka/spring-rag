"""Memory calls across real stdio MCP connections, synthetic HTTP services only."""

from dataclasses import replace
from contextlib import closing
import json
import sqlite3
import sys
import tempfile
from pathlib import Path
import unittest

from fastmcp import Client
from fastmcp.client.transports import StdioTransport

from week3.mcp_client import ROOT
from week3.memory_client import call_memory, memory_transport
from week3.memory_ledger import Ledger, Scope
from week3.mock_mem0 import Mem0Fixture


class MCPMemoryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.scope = Scope()
        self.ledger = Ledger(self.path / "operations.sqlite3", self.scope)

    def fixture_transport(self, scenario, log=None):
        return StdioTransport(
            command=sys.executable, args=["-m", "week3.mock_mcp_server"], cwd=str(ROOT),
            env={"SPRING_RAG_MEMORY_STATE": str(self.path), "MCP_MEMORY_SCENARIO": scenario},
            log_file=log,
        )

    async def test_approval_save_then_new_server_session_recall_and_duplicate(self):
        row = self.ledger.propose("java_comparisons")
        self.ledger.decide(row["operation_id"], approve=True)
        arguments = {"operation_id": row["operation_id"]}
        saved = await call_memory(self.scope, self.path, "remember_project_fact", arguments)
        self.assertEqual(saved["result"]["status"], "completed")
        self.assertEqual(self.ledger.get(row["operation_id"])["state"], "completed")
        # call_memory closes each server before starting the next one.
        recalled = await call_memory(self.scope, self.path, "recall_project_facts", {})
        self.assertEqual(recalled["result"]["facts"][0]["fact"], row["fact"])
        self.assertEqual(recalled["result"]["service_mode"], "mock_services")
        replay = await call_memory(self.scope, self.path, "remember_project_fact", arguments)
        self.assertEqual(replay["result"], saved["result"])
        fixture = Mem0Fixture(self.path / "fixture-remote.sqlite3")
        self.assertEqual(fixture.calls("/v3/memories/add/"), 1)

    async def test_rejection_persists_across_new_server(self):
        row = self.ledger.propose("redis")
        self.ledger.decide(row["operation_id"], approve=False)
        for _ in range(2):
            result = await call_memory(self.scope, self.path, "remember_project_fact", {"operation_id": row["operation_id"]})
            self.assertEqual(result["result"]["status"], "rejected")
        self.assertFalse((self.path / "fixture-remote.sqlite3").exists())

    async def test_model_boolean_does_not_authorize_write_and_schema_has_no_approval(self):
        row = self.ledger.propose("java_comparisons")
        async with Client(memory_transport(self.scope, self.path), timeout=30) as client:
            tools = {tool.name: tool for tool in await client.list_tools()}
            self.assertEqual(set(tools), {"search_caching_docs", "recall_project_facts", "remember_project_fact"})
            schema = tools["remember_project_fact"].input_schema
            self.assertEqual(set(schema["properties"]), {"operation_id"})
            await client.call_tool("remember_project_fact", {"operation_id": row["operation_id"], "approved": True}, raise_on_error=False)
        self.assertEqual(self.ledger.get(row["operation_id"])["state"], "proposed")
        self.assertFalse((self.path / "fixture-remote.sqlite3").exists())

    async def test_project_isolation_for_writes_and_reads_over_mcp(self):
        row = self.ledger.propose("spring_boot_4")
        self.ledger.decide(row["operation_id"], approve=True)
        await call_memory(self.scope, self.path, "remember_project_fact", {"operation_id": row["operation_id"]})
        other = replace(self.scope, project_id="another-synthetic-project")
        async with Client(memory_transport(other, self.path), timeout=30) as client:
            response = await client.call_tool("remember_project_fact", {"operation_id": row["operation_id"]}, raise_on_error=False)
            self.assertTrue(response.is_error)
            recall = await client.call_tool("recall_project_facts", {"topic": "project"})
            self.assertEqual(recall.data["facts"], [])

    async def test_timed_out_write_reconciles_in_new_mcp_server_without_duplicate(self):
        row = self.ledger.propose("java_comparisons")
        self.ledger.decide(row["operation_id"], approve=True)
        arguments = {"operation_id": row["operation_id"]}
        with tempfile.TemporaryFile(mode="w+") as log:
            async with Client(self.fixture_transport("timeout_after", log), timeout=30) as client:
                result = await client.call_tool("remember_project_fact", arguments)
                self.assertEqual(result.data["status"], "uncertain")
                self.assertEqual(result.data["diagnostic"], {"stage": "add", "code": "TIMEOUT"})
                self.assertNotIn("synthetic-secret", str(result.content))
            log.seek(0)
            self.assertNotIn("synthetic-secret", log.read())
        confirmed = await call_memory(self.scope, self.path, "remember_project_fact", arguments)
        self.assertEqual(confirmed["result"]["status"], "completed")
        fixture = Mem0Fixture(self.path / "fixture-remote.sqlite3")
        self.assertEqual(fixture.calls("/v3/memories/add/"), 1)
        self.assertNotIn(b"synthetic-secret", (self.path / "operations.sqlite3").read_bytes())

    async def test_recall_service_failure_is_sanitized_over_mcp_and_stderr(self):
        with tempfile.TemporaryFile(mode="w+") as log:
            async with Client(self.fixture_transport("read_error", log), timeout=30) as client:
                result = await client.call_tool("recall_project_facts", {}, raise_on_error=False)
                self.assertTrue(result.is_error)
                self.assertIn("Memory recall failed", str(result.content))
                self.assertNotIn("synthetic-secret", str(result.content))
            log.seek(0)
            self.assertNotIn("synthetic-secret", log.read())

    async def test_event_diagnostic_travels_over_mcp_without_raw_error(self):
        row = self.ledger.propose("java_comparisons")
        self.ledger.decide(row["operation_id"], approve=True)
        with tempfile.TemporaryFile(mode="w+") as log:
            async with Client(self.fixture_transport("event_error", log), timeout=30) as client:
                result = await client.call_tool("remember_project_fact", {"operation_id": row["operation_id"]})
                self.assertEqual(result.data["diagnostic"], {"stage": "event_status", "code": "HTTP_ERROR", "http_status": 503})
                self.assertNotIn("synthetic-secret", str(result.content))
            log.seek(0)
            self.assertNotIn("synthetic-secret", log.read())

    async def test_provider_metadata_extensions_and_true_text_reconcile_across_sessions(self):
        row = self.ledger.propose("java_comparisons")
        self.ledger.decide(row["operation_id"], approve=True)
        arguments = {"operation_id": row["operation_id"]}
        async with Client(self.fixture_transport("timeout_after"), timeout=30) as client:
            response = await client.call_tool("remember_project_fact", arguments)
            self.assertEqual(response.data["status"], "uncertain")
        fixture = Mem0Fixture(self.path / "fixture-remote.sqlite3")
        item = fixture.items()[0]
        item["metadata"]["provider_annotation"] = "synthetic-extra-must-not-be-returned"
        item["metadata"]["synthetic"] = "true"
        with closing(sqlite3.connect(fixture.path)) as db, db:
            db.execute("UPDATE facts SET body=?", (json.dumps(item),))
        result = await call_memory(self.scope, self.path, "remember_project_fact", arguments)
        self.assertEqual(result["result"]["status"], "completed")
        recalled = await call_memory(self.scope, self.path, "recall_project_facts", {})
        self.assertEqual(recalled["result"]["facts"][0]["fact"], row["fact"])
        self.assertNotIn("synthetic-extra", json.dumps(recalled))
        self.assertEqual(fixture.calls("/v3/memories/add/"), 1)


if __name__ == "__main__":
    unittest.main()
