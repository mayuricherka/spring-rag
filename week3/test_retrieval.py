"""Offline tests for retrieval, with external service clients injected as mocks."""

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from week3.mcp_client import main
from week3.mcp_server import live_search
from week3.mock_mcp_server import make_mock_service
from week3.retrieval import (
    DIMENSIONS, DOCUMENTATION_VERSION, EMBEDDING_MODEL, INDEX_NAME, NAMESPACE,
    SOURCE_URL, RetrievalError,
)


class RetrievalTests(unittest.TestCase):
    def test_project_corpus_matches_configuration(self):
        rows = json.loads((Path(__file__).resolve().parent.parent / "chunks.json").read_text())
        self.assertEqual(len(rows), 14)
        self.assertEqual({row["embedding_model"] for row in rows}, {EMBEDDING_MODEL})
        self.assertEqual({row["source"] for row in rows}, {SOURCE_URL})

    def test_question_embedding_and_read_only_query_return_evidence(self):
        service = make_mock_service()
        result = service.search(" How do I enable caching? ")
        service.pinecone_client.describe_index.assert_called_once_with(INDEX_NAME)
        service.embedding_client.embeddings.create.assert_called_once_with(
            model=EMBEDDING_MODEL,
            input=["Instruct: Retrieve Spring documentation passages that answer the question.\n"
                   "Query: How do I enable caching?"], encoding_format="float",
        )
        index = service.pinecone_client.Index.return_value
        query = index.query.call_args.kwargs
        self.assertEqual(query["namespace"], NAMESPACE)
        self.assertEqual(query["top_k"], 5)
        self.assertFalse(query["include_values"])
        self.assertTrue(query["include_metadata"])
        self.assertEqual(query["filter"]["source"]["$eq"], SOURCE_URL)
        self.assertEqual(len(query["vector"]), DIMENSIONS)
        self.assertEqual(result["service_mode"], "mock_services")
        self.assertEqual(result["index_settings"]["namespace_vectors"], 14)
        self.assertEqual(result["embedding_usage"]["total_tokens"], 12)
        passage = result["passages"][0]
        self.assertEqual(passage["id"], "synthetic-caching-001")
        self.assertEqual(passage["score"], 0.85)
        self.assertEqual(passage["source"], SOURCE_URL)
        self.assertEqual(passage["documentation_version"], DOCUMENTATION_VERSION)
        self.assertEqual(passage["version_basis"], "stored_source_url")
        self.assertIn("Synthetic", passage["text"])
        service.pinecone_client.create_index.assert_not_called()
        index.upsert.assert_not_called()
        index.close.assert_called_once()

    def test_bad_questions_and_unsupported_versions_use_no_services(self):
        for question, version in [(" ", "4.0"), ("x" * 2001, "4.0"), ("How?", "3.5")]:
            with self.subTest(version=version, length=len(question)):
                service = make_mock_service()
                with self.assertRaises(RetrievalError):
                    service.search(question, version)
                service.embedding_client.embeddings.create.assert_not_called()
                service.pinecone_client.describe_index.assert_not_called()

    def test_wrong_index_settings_stop_before_embedding(self):
        for field, value in [("dimension", 1024), ("metric", "dotproduct"),
                             ("status", {"ready": False})]:
            service = make_mock_service()
            setattr(service.pinecone_client.describe_index.return_value, field, value)
            with self.assertRaises(RetrievalError):
                service.search("How do I enable caching?")
            service.embedding_client.embeddings.create.assert_not_called()
            service.pinecone_client.Index.assert_not_called()

    def test_missing_or_changed_namespace_stops_before_embedding(self):
        for namespaces in [{}, {NAMESPACE: SimpleNamespace(vector_count=13)}]:
            service = make_mock_service()
            index = service.pinecone_client.Index.return_value
            index.describe_index_stats.return_value.namespaces = namespaces
            with self.assertRaises(RetrievalError):
                service.search("How do I enable caching?")
            service.embedding_client.embeddings.create.assert_not_called()
            index.query.assert_not_called()
            index.close.assert_called_once()

    def test_invalid_embedding_is_not_sent_to_pinecone(self):
        for data in [[], [SimpleNamespace(index=0, embedding=[0.1])],
                     [SimpleNamespace(index=0, embedding=[float("nan")] * DIMENSIONS)]]:
            service = make_mock_service()
            service.embedding_client.embeddings.create.return_value.data = data
            with self.assertRaises(RetrievalError):
                service.search("How do I enable caching?")
            service.pinecone_client.Index.return_value.query.assert_not_called()
            self.assertEqual(service.embedding_client.embeddings.create.call_count, 1)

    def test_unverified_metadata_is_excluded_and_empty_results_are_explicit(self):
        for metadata in [None, {}, {"text": "unverified", "source": SOURCE_URL.replace("4.0", "3.5"),
                                   "embedding_model": EMBEDDING_MODEL}]:
            service = make_mock_service()
            index = service.pinecone_client.Index.return_value
            index.query.return_value.matches[0].metadata = metadata
            result = service.search("How do I enable caching?")
            self.assertEqual(result["passages"], [])
            self.assertEqual(result["status"], "no_matches")
        index.query.return_value.matches = []
        self.assertEqual(service.search("How do I enable caching?")["passages"], [])

    def test_provider_error_is_sanitized_without_embedding_retry(self):
        service = make_mock_service("provider_error")
        with self.assertRaises(RetrievalError) as caught:
            service.search("How do I enable caching?")
        self.assertNotIn("synthetic-secret", str(caught.exception))
        self.assertEqual(service.embedding_client.embeddings.create.call_count, 1)
        service.pinecone_client.Index.return_value.query.assert_not_called()

    def test_missing_usage_remains_unknown(self):
        service = make_mock_service()
        service.embedding_client.embeddings.create.return_value.usage = None
        self.assertIsNone(service.search("How do I enable caching?")["embedding_usage"])

    def test_server_live_guard_does_not_initialize_providers(self):
        with patch.dict(os.environ, {}, clear=True), patch("week3.mcp_server.OpenAI") as client:
            with self.assertRaises(RetrievalError):
                live_search("How do I enable caching?", "4.0")
            client.assert_not_called()

    def test_rejected_live_confirmation_never_launches_mcp_or_reads_keys(self):
        with patch("builtins.input", return_value="NO"), \
             patch("week3.mcp_client.getpass") as key_input, \
             patch("week3.mcp_client.search_over_mcp") as search, \
             redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--live", "How do I enable caching?"]), 0)
        key_input.assert_not_called()
        search.assert_not_called()


if __name__ == "__main__":
    unittest.main()
