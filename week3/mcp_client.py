"""Discover and invoke search_caching_docs over a separate-process MCP connection."""

import argparse
import asyncio
from getpass import getpass
import json
import os
from pathlib import Path
import sys

from fastmcp import Client
from fastmcp.client.transports import StdioTransport


ROOT = Path(__file__).resolve().parent.parent


def server_transport(*, mock=False, credentials=None):
    # Pass only the required credentials, never the complete parent environment.
    env = {}
    if credentials:
        env.update(credentials)
        env["SPRING_RAG_ALLOW_LIVE"] = "1"
    return StdioTransport(
        command=sys.executable,
        args=["-m", "week3.mock_mcp_server" if mock else "week3.mcp_server"],
        cwd=str(ROOT),
        env=env,
    )


async def search_over_mcp(question=None, documentation_version="4.0", *, mock=False, credentials=None):
    async with Client(server_transport(mock=mock, credentials=credentials), timeout=90) as client:
        tools = await client.list_tools()
        names = [tool.name for tool in tools]
        if "search_caching_docs" not in names:
            raise RuntimeError("Documentation search was not discovered over MCP.")
        result = {"discovered_tools": names}
        if question is not None:
            reply = await client.call_tool("search_caching_docs", {
                "question": question, "documentation_version": documentation_version,
            })
            result["result"] = reply.data
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?")
    parser.add_argument("--version", default="4.0", help="Spring Boot reference line; only 4.0 is indexed.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--mock", action="store_true", help="Use mock service responses over real MCP; no paid calls.")
    mode.add_argument("--live", action="store_true", help="Request confirmation for one live question embedding.")
    args = parser.parse_args(argv)
    credentials = None
    if args.live:
        if not args.question:
            parser.error("Supply a question for --live.")
        print("This may send ONE billable Nebius embedding request and read the existing Pinecone index.")
        print("Current pricing and credit balance have not been checked. No document upload or re-embedding occurs.")
        if input("Type RUN to approve this request, or anything else to cancel: ").strip() != "RUN":
            print("Cancelled. No requests sent.")
            return 0
        credentials = {name: os.environ.get(name) or getpass(f"{name} (hidden, not saved): ")
                       for name in ("PINECONE_API_KEY", "NEBIUS_API_KEY")}
    elif args.question and not args.mock:
        parser.error("Use --mock for offline service responses or --live to approve retrieval.")
    try:
        result = asyncio.run(search_over_mcp(
            args.question, args.version, mock=args.mock, credentials=credentials,
        ))
    except Exception:
        print("MCP search failed. Check the supported version, service setup, and index settings. Raw errors are hidden.")
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
