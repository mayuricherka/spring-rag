"""Real MCP stdio integration with mocked Nebius/Pinecone responses."""

import sys
import tempfile
import unittest

from fastmcp import Client
from fastmcp.client.transports import StdioTransport

from week3.mcp_client import ROOT, search_over_mcp, server_transport


class MCPRetrievalTests(unittest.IsolatedAsyncioTestCase):
    async def test_discover_and_search_through_separate_process(self):
        result = await search_over_mcp("How do I enable caching?", mock=True)
        self.assertEqual(set(result["discovered_tools"]), {
            "search_caching_docs", "recall_project_facts", "remember_project_fact",
        })
        self.assertEqual(result["result"]["service_mode"], "mock_services")
        passage = result["result"]["passages"][0]
        self.assertEqual(passage["documentation_version"], "4.0")
        self.assertEqual(passage["id"], "synthetic-caching-001")
        self.assertIn("Synthetic", passage["text"])

    async def test_discovery_schema_and_unsupported_version(self):
        async with Client(server_transport(mock=True), timeout=15) as client:
            tools = await client.list_tools()
            schema = next(tool for tool in tools if tool.name == "search_caching_docs").input_schema
            self.assertIn("question", schema["required"])
            self.assertIn("documentation_version", schema["properties"])
            result = await client.call_tool("search_caching_docs", {
                "question": "How do I enable caching?", "documentation_version": "3.5",
            }, raise_on_error=False)
            self.assertTrue(result.is_error)
            self.assertIn("Only Spring Boot 4.0", result.content[0].text)

    async def test_real_server_allows_discovery_but_blocks_unapproved_live_call(self):
        async with Client(server_transport(), timeout=15) as client:
            self.assertEqual({tool.name for tool in await client.list_tools()}, {
                "search_caching_docs", "recall_project_facts", "remember_project_fact",
            })
            result = await client.call_tool("search_caching_docs", {
                "question": "How do I enable caching?",
            }, raise_on_error=False)
            self.assertTrue(result.is_error)
            self.assertIn("Live retrieval is disabled", result.content[0].text)

    async def test_provider_failure_does_not_leak_raw_error_over_mcp_or_stderr(self):
        with tempfile.TemporaryFile(mode="w+") as log:
            transport = StdioTransport(
                command=sys.executable, args=["-m", "week3.mock_mcp_server"], cwd=str(ROOT),
                env={"MCP_TEST_SCENARIO": "provider_error"}, log_file=log,
            )
            async with Client(transport, timeout=15) as client:
                result = await client.call_tool("search_caching_docs", {
                    "question": "How do I enable caching?",
                }, raise_on_error=False)
                self.assertTrue(result.is_error)
                self.assertNotIn("synthetic-secret", str(result.content))
            log.seek(0)
            self.assertNotIn("synthetic-secret", log.read())


if __name__ == "__main__":
    unittest.main()
