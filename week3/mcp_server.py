"""Local stdio MCP server for documentation and memory; live services are gated."""

import os

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from openai import OpenAI
from pinecone import Pinecone

from week3.retrieval import DocumentationSearch, NEBIUS_BASE_URL, RetrievalError, validate_request
from week3.mem0_hosted import configured_memory
from week3.memory_ledger import MemorySafetyError


def live_search(question, documentation_version):
    validate_request(question, documentation_version)
    if os.environ.get("SPRING_RAG_ALLOW_LIVE") != "1":
        raise RetrievalError("Live retrieval is disabled. Use the client --live option and approve the embedding request.")
    keys = {name: os.environ.get(name, "").strip() for name in ("NEBIUS_API_KEY", "PINECONE_API_KEY")}
    if not all(keys.values()):
        raise RetrievalError("Both service API keys are required. No credential files are loaded by this server.")
    try:
        with OpenAI(
            api_key=keys["NEBIUS_API_KEY"], base_url=NEBIUS_BASE_URL,
            max_retries=0, timeout=30.0,
        ) as embedding_client, Pinecone(api_key=keys["PINECONE_API_KEY"]) as pc:
            return DocumentationSearch(embedding_client, pc).search(question, documentation_version)
    except RetrievalError:
        raise
    except Exception:
        raise RetrievalError("Could not initialize the retrieval services. Check local credential setup.") from None


def create_server(search=live_search, memory_factory=configured_memory):
    server = FastMCP("Spring caching documentation", mask_error_details=True)

    @server.tool(annotations={"readOnlyHint": True, "destructiveHint": False})
    def search_caching_docs(question: str, documentation_version: str = "4.0") -> dict:
        """Search existing Spring Boot 4.0 passages; one live question embedding may be billed.

        documentation_version names the reference line, such as 4.0, not a patch release.
        """
        try:
            return search(question, documentation_version)
        except RetrievalError as error:
            raise ToolError(str(error)) from None

    @server.tool(annotations={"readOnlyHint": True, "destructiveHint": False})
    def recall_project_facts(topic: str = "preferences") -> dict:
        """Recall completed, approved synthetic facts for this server's fixed project.

        topic is preferences or project. Live reads require separate cost approval.
        """
        try:
            return memory_factory().recall(topic)
        except MemorySafetyError as error:
            raise ToolError(str(error)) from None
        except Exception:
            raise ToolError("Memory service unavailable. Raw errors are hidden.") from None

    @server.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
    def remember_project_fact(operation_id: str) -> dict:
        """Save ONLY an operation approved by a human in the trusted local client.

        An operation ID is not approval. No fact, scope or approved argument is
        accepted here. Repeated uncertain operations only reconcile with reads.
        """
        try:
            return memory_factory().remember(operation_id)
        except MemorySafetyError as error:
            raise ToolError(str(error)) from None
        except Exception:
            raise ToolError("Memory service unavailable. Raw errors are hidden.") from None

    return server


if __name__ == "__main__":
    create_server().run(transport="stdio", show_banner=False, log_level="ERROR")
