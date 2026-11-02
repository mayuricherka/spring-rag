"""Test-only server: real MCP, retrieval and memory code; synthetic services."""

import os
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from week3.mcp_server import create_server
from week3.mem0_hosted import configured_memory
from week3.retrieval import DocumentationSearch, DIMENSIONS, EMBEDDING_MODEL, NAMESPACE, SOURCE_URL


def record_mock_call(name, path=None):
    """Synthetic fixture audit: operation names only, never inputs or keys."""
    path = path or os.environ.get("MCP_TEST_AUDIT_PATH")
    if path:
        with Path(path).open("a") as output:
            output.write(json.dumps({"operation": name}) + "\n")


def mock_memory_factory():
    memory = configured_memory(mock=True)
    def recall(topic):
        record_mock_call("recall_project_facts")
        return memory.recall(topic)
    return SimpleNamespace(recall=recall, remember=memory.remember)


def make_mock_service(scenario="success"):
    embeddings = Mock()
    embeddings.embeddings.create.return_value = SimpleNamespace(
        data=[SimpleNamespace(index=0, embedding=[0.01] * DIMENSIONS)],
        usage=SimpleNamespace(prompt_tokens=12, total_tokens=12),
    )
    pc, index = Mock(), Mock()
    pc.describe_index.return_value = SimpleNamespace(
        dimension=DIMENSIONS, metric="cosine", status={"ready": True}, host="synthetic.example",
    )
    pc.Index.return_value = index
    index.describe_index_stats.return_value = SimpleNamespace(
        namespaces={NAMESPACE: SimpleNamespace(vector_count=14)},
    )
    index.query.return_value = SimpleNamespace(matches=[SimpleNamespace(
        id="synthetic-caching-001", score=0.85,
        metadata={
            "text": "Synthetic passage for an MCP transport test, not live documentation evidence.",
            "source": SOURCE_URL, "title": "Synthetic fixture", "embedding_model": EMBEDDING_MODEL,
        },
    )])
    if scenario == "provider_error":
        embeddings.embeddings.create.side_effect = RuntimeError("synthetic-secret-must-not-leak")
    if scenario == "no_matches":
        index.query.return_value = SimpleNamespace(matches=[])
    return DocumentationSearch(embeddings, pc, service_mode="mock_services")


if __name__ == "__main__":
    service = make_mock_service(os.environ.get("MCP_TEST_SCENARIO", "success"))
    def search(question, documentation_version):
        record_mock_call("search_caching_docs")
        return service.search(question, documentation_version)
    create_server(search, memory_factory=mock_memory_factory).run(
        transport="stdio", show_banner=False, log_level="ERROR",
    )
