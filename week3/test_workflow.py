"""Fixed workflow with real LangGraph, SQLite, stdio MCP and synthetic services."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from week3.workflow import checkpointed_workflow, execute_run, initial_state, prepare_run
from week3.memory_ledger import FACTS, Ledger, Scope
from week3.mock_mem0 import fixture_backend
from week3.project_memory import ProjectMemory
from week3.workflow_services import MCPReads, WorkflowError, fixture_answer, make_messages, workflow_transport


QUESTION = "How do I enable caching?"


class WorkflowTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Each test gets temporary state, separate from the existing demo ledger.
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.scope = Scope()

    async def run_graph(self, *, question=QUESTION, thread="synthetic-thread", failure=None,
                        pause=False, scope=None):
        scope = scope or self.scope
        # MCP and LangGraph are real; external services and answers are synthetic.
        tools = MCPReads(lambda: workflow_transport(scope, self.path, {}, mock_failure=failure))
        async def answer(state):
            return await fixture_answer(state, self.path, fail=failure == "model")
        async with checkpointed_workflow(self.path, tools, answer, scope, pause_after_docs=pause) as graph:
            plan = await prepare_run(graph, scope, thread, question=question)
            return await execute_run(graph, plan)

    def calls(self):
        path = self.path / "fixture-workflow-calls.jsonl"
        return [json.loads(line)["operation"] for line in path.read_text().splitlines()] if path.exists() else []

    def seed_fact(self, name):
        # Simulate trusted approval and completion in the fixture, not hosted Mem0.
        ledger = Ledger(self.path / "operations.sqlite3", self.scope)
        row = ledger.propose(name)
        ledger.decide(row["operation_id"], approve=True)
        service = ProjectMemory(ledger, lambda: fixture_backend(self.scope, self.path))
        self.assertEqual(service.remember(row["operation_id"])["status"], "completed")

    async def test_empty_memory_still_answers_with_documentation_citations(self):
        result = await self.run_graph()
        self.assertEqual(result["status"], "completed")
        state = result["state"]
        self.assertEqual(state["memory_status"], "empty")
        self.assertEqual(state["preferences"], [])
        self.assertIn("[1]", state["answer"])
        self.assertEqual(state["citations"][0]["id"], "synthetic-caching-001")
        self.assertEqual(state["citations"][0]["documentation_version"], "4.0")
        self.assertEqual(self.calls(), ["recall_project_facts", "search_caching_docs", "nemotron_fixture"])

    async def test_approved_style_is_separate_from_evidence_and_project_facts_are_excluded(self):
        self.seed_fact("java_comparisons")
        self.seed_fact("redis")
        result = await self.run_graph()
        state = result["state"]
        self.assertEqual(state["preferences"], [FACTS["java_comparisons"]])
        messages = make_messages(state)
        preferences = json.loads(messages[1]["content"])["PRESENTATION_PREFERENCES"]
        evidence = json.loads(messages[2]["content"])["DOCUMENTATION_EVIDENCE"]
        self.assertEqual(preferences, [FACTS["java_comparisons"]])
        self.assertNotIn(FACTS["java_comparisons"], json.dumps(evidence))
        self.assertNotIn(FACTS["redis"], json.dumps(messages))
        self.assertEqual(state["documentation_version"], "4.0")
        self.assertNotIn("remember_project_fact", self.calls())

    async def test_memory_tool_failure_continues_without_inventing_preferences(self):
        result = await self.run_graph(failure="memory")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["state"]["memory_status"], "unavailable")
        self.assertEqual(result["state"]["preferences"], [])
        self.assertEqual(result["state"]["warnings"], ["MEMORY_UNAVAILABLE_CONTINUING_WITH_DOCUMENTATION"])
        self.assertIn("[1]", result["state"]["answer"])

    async def test_no_documentation_never_answers_from_memory(self):
        self.seed_fact("java_comparisons")
        result = await self.run_graph(failure="no_docs")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["state"]["phase"], "not_found")
        self.assertEqual(result["state"]["citations"], [])
        self.assertNotIn("nemotron_fixture", self.calls())

    async def test_document_tool_failure_stops_generation_and_resume_retries_only_failed_step(self):
        failed = await self.run_graph(failure="docs")
        self.assertEqual(failed["status"], "blocked")
        self.assertEqual(failed["error_code"], "DOCUMENTATION_TOOL_FAILED")
        self.assertEqual(failed["next_nodes"], ["retrieve_documentation"])
        self.assertEqual(self.calls(), ["recall_project_facts", "search_caching_docs"])
        resumed = await self.run_graph(question=None)
        self.assertEqual(resumed["status"], "completed")
        self.assertEqual(self.calls(), ["recall_project_facts", "search_caching_docs",
                                       "search_caching_docs", "nemotron_fixture"])
        for path in self.path.glob("workflow-checkpoints.sqlite3*"):
            self.assertNotIn(b"synthetic-secret-must-not-leak", path.read_bytes())

    async def test_generation_failure_and_resume_do_not_repeat_mcp_reads(self):
        failed = await self.run_graph(failure="model")
        self.assertEqual(failed["status"], "blocked")
        self.assertEqual(failed["next_nodes"], ["generate_answer"])
        result = await self.run_graph(question=None)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(self.calls(), ["recall_project_facts", "search_caching_docs",
                                       "nemotron_fixture", "nemotron_fixture"])

    async def test_checkpoint_resume_skips_completed_calls_and_completed_replay_is_read_only(self):
        # Reopen the paused graph and verify completed work is not requested again.
        paused = await self.run_graph(pause=True)
        self.assertEqual(paused["status"], "paused")
        self.assertEqual(paused["next_nodes"], ["generate_answer"])
        self.assertEqual(paused["state"]["answer"], "")
        self.assertEqual(self.calls(), ["recall_project_facts", "search_caching_docs"])
        completed = await self.run_graph(question=None)
        again = await self.run_graph(question=None)
        self.assertEqual(again, completed)
        self.assertEqual(self.calls(), ["recall_project_facts", "search_caching_docs", "nemotron_fixture"])
        self.assertTrue((self.path / "operations.sqlite3").exists())
        self.assertTrue((self.path / "workflow-checkpoints.sqlite3").exists())

    async def test_resume_rejects_another_project_or_service_mode_before_tool_calls(self):
        await self.run_graph(pause=True)
        before = self.calls()
        for scope in (replace(self.scope, project_id="another-synthetic-project"),
                      replace(self.scope, mode="hosted_mem0", cloud_org="synthetic-org", cloud_project="synthetic-project")):
            with self.assertRaisesRegex(WorkflowError, "SCOPE_OR_SERVICE_MODE_MISMATCH"):
                await self.run_graph(question=None, scope=scope)
        self.assertEqual(self.calls(), before)

    async def test_start_cannot_overwrite_an_existing_thread(self):
        await self.run_graph(pause=True)
        before = self.calls()
        with self.assertRaisesRegex(WorkflowError, "THREAD_ALREADY_EXISTS"):
            await self.run_graph(question="What does Cacheable do?")
        self.assertEqual(self.calls(), before)

    async def test_sensitive_or_configuration_input_is_rejected_before_checkpointing(self):
        question = "How does caching work? password=synthetic-secret-must-not-leak"
        with self.assertRaises(WorkflowError):
            await self.run_graph(question=question)
        self.assertEqual(self.calls(), [])
        for path in self.path.glob("workflow-checkpoints.sqlite3*"):
            self.assertNotIn(b"synthetic-secret-must-not-leak", path.read_bytes())
        with self.assertRaises(WorkflowError):
            initial_state("spring.cache.type=redis", self.scope)

    async def test_workflow_gateway_cannot_call_memory_write_tool(self):
        def forbidden():
            self.fail("A write request must not even open an MCP connection")
        with self.assertRaisesRegex(WorkflowError, "WORKFLOW_TOOL_NOT_ALLOWED"):
            await MCPReads(forbidden)("remember_project_fact", {"operation_id": "synthetic"})


if __name__ == "__main__":
    unittest.main()
