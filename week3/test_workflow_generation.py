"""Nemotron adapter contract and evidence-only citation validation; no model API."""

import json
import unittest
from unittest.mock import AsyncMock, Mock

from openai.types.chat import ChatCompletion

from week3.memory_ledger import FACTS, Scope
from week3.retrieval import SOURCE_URL
from week3.workflow import initial_state
from week3.workflow_services import (
    ANSWER_MODEL, NemotronAnswerer, WorkflowError, parse_answer,
    read_passages, read_preferences, required_credentials,
)


def completion(text, *, finish_reason="stop"):
    return ChatCompletion.model_validate({
        "id": "synthetic", "created": 0, "model": ANSWER_MODEL, "object": "chat.completion",
        "choices": [{"index": 0, "finish_reason": finish_reason,
                     "message": {"role": "assistant", "content": text}}], "usage": None,
    })


class GenerationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.state = initial_state("How do I enable caching?", Scope())
        self.state["preferences"] = [FACTS["java_comparisons"]]
        self.state["passages"] = [{"id": "synthetic-doc", "text": "Synthetic documentation evidence.",
                                   "source": SOURCE_URL, "documentation_version": "4.0"}]

    async def test_exact_model_no_advertised_tools_separate_preference_and_doc_blocks(self):
        client = Mock()
        client.chat.completions.create = AsyncMock(return_value=completion("ANSWER: Synthetic fact [1]."))
        result = await NemotronAnswerer(client)(self.state)
        request = client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["model"], ANSWER_MODEL)
        self.assertNotIn("tools", request)
        self.assertNotIn("tool_choice", request)
        self.assertEqual(request["max_tokens"], 600)
        preferences = json.loads(request["messages"][1]["content"])
        docs = json.loads(request["messages"][2]["content"])
        self.assertEqual(preferences["PRESENTATION_PREFERENCES"], self.state["preferences"])
        self.assertNotIn(FACTS["java_comparisons"], json.dumps(docs))
        self.assertEqual(result["citations"], [{"number": 1, "id": "synthetic-doc",
                                                "source": SOURCE_URL, "documentation_version": "4.0"}])
        self.assertIsNone(result["generation_usage"])

    async def test_missing_out_of_range_and_negative_citations_are_rejected(self):
        for text in ("ANSWER: Uncited claim.", "ANSWER: Claim [2].", "ANSWER: Claim [0].",
                     "ANSWER: Claim [1] and [-1].", "Unlabelled answer [1]."):
            with self.subTest(text=text), self.assertRaises(WorkflowError):
                parse_answer(completion(text), self.state["passages"])

    async def test_incomplete_answer_is_not_persisted_as_success(self):
        with self.assertRaises(WorkflowError):
            parse_answer(completion("ANSWER: Partial [1]", finish_reason="length"), self.state["passages"])

    async def test_not_found_is_explicit_and_has_no_sources(self):
        result = parse_answer(completion("NOT_FOUND: No evidence."), self.state["passages"])
        self.assertEqual(result["citations"], [])

    async def test_model_failure_is_sanitized_without_automatic_retry(self):
        client = Mock()
        client.chat.completions.create = AsyncMock(side_effect=RuntimeError("synthetic-secret-must-not-leak"))
        with self.assertRaisesRegex(WorkflowError, "^ANSWER_SERVICE_FAILED$"):
            await NemotronAnswerer(client)(self.state)
        self.assertEqual(client.chat.completions.create.await_count, 1)


class WorkflowServiceTests(unittest.TestCase):
    def test_preferences_remain_scoped_deduplicated_and_separate_from_project_facts(self):
        scope = Scope()
        response = {"status": "ok", "service_mode": scope.mode, "user_id": scope.user_id,
                    "project_id": scope.project_id, "facts": [
                        {"fact": FACTS["redis"]}, {"fact": FACTS["java_comparisons"]},
                        {"fact": FACTS["java_comparisons"]},
                    ]}
        self.assertEqual(read_preferences(response, scope), [FACTS["java_comparisons"]])
        self.assertEqual(read_preferences({**response, "facts": []}, scope), [])
        for field in ("service_mode", "user_id", "project_id"):
            with self.subTest(field=field), self.assertRaisesRegex(WorkflowError, "MEMORY_RESPONSE_SCOPE_INVALID"):
                read_preferences({**response, field: "another-synthetic-scope"}, scope)

    def test_document_evidence_keeps_source_version_and_mode_guards(self):
        state = initial_state("How do I enable caching?", Scope())
        passage = {"id": "synthetic-doc", "score": 0.8, "text": "Synthetic evidence.",
                   "source": SOURCE_URL, "title": "Synthetic", "documentation_version": "4.0",
                   "version_basis": "stored_source_url", "extra": "omit this field"}
        response = {"status": "ok", "service_mode": state["service_mode"],
                    "documentation_version": "4.0", "passages": [passage]}
        self.assertNotIn("extra", read_passages(response, state)[0])
        for field, value in (("source", "https://example.com/synthetic"),
                             ("documentation_version", "3.0"), ("text", "")):
            with self.subTest(field=field), self.assertRaisesRegex(WorkflowError, "DOCUMENTATION_EVIDENCE_INVALID"):
                read_passages({**response, "passages": [{**passage, field: value}]}, state)
        with self.assertRaisesRegex(WorkflowError, "DOCUMENTATION_RESPONSE_INVALID"):
            read_passages({**response, "service_mode": "live_services"}, state)

    def test_resume_requires_only_keys_for_unfinished_steps(self):
        cases = [
            (["recall_preferences"], ["MEM0_API_KEY", "PINECONE_API_KEY", "NEBIUS_API_KEY"]),
            (["retrieve_documentation"], ["PINECONE_API_KEY", "NEBIUS_API_KEY"]),
            (["generate_answer"], ["NEBIUS_API_KEY"]),
            (["not_found"], []),
            ([], []),
        ]
        for nodes, expected in cases:
            with self.subTest(nodes=nodes):
                self.assertEqual(required_credentials(nodes), expected)


if __name__ == "__main__":
    unittest.main()
