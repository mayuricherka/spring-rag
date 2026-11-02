"""Trusted terminal approval followed by real MCP invocation.

This client is the human boundary, like an approval screen in a Java application.
Only the MCP server calls the hosted memory service.
"""

import argparse
import asyncio
from dataclasses import asdict
from getpass import getpass
import json
import os
from pathlib import Path
import sys

from fastmcp import Client
from fastmcp.client.transports import StdioTransport

from week3.mcp_client import ROOT
from week3.memory_ledger import FACTS, Ledger, MemorySafetyError, Scope, STATE_DIR


def memory_transport(scope, state_dir, *, credentials=None, fixture_scenario=None):
    env = {"SPRING_RAG_MEMORY_STATE": str(Path(state_dir).resolve()),
           "SPRING_RAG_MEMORY_PROJECT": scope.project_id,
           "MEM0_EXPECTED_ORG": scope.cloud_org, "MEM0_EXPECTED_PROJECT": scope.cloud_project}
    if scope.mode == "mock_services" and fixture_scenario:
        env["MCP_MEMORY_SCENARIO"] = fixture_scenario
    if credentials and scope.mode == "hosted_mem0":
        env["MEM0_API_KEY"] = credentials["MEM0_API_KEY"]
        env["SPRING_RAG_ALLOW_MEM0"] = "1"
    return StdioTransport(
        command=sys.executable,
        args=["-m", "week3.mock_mcp_server" if scope.mode == "mock_services" else "week3.mcp_server"],
        cwd=str(ROOT), env=env,
    )


async def call_memory(scope, state_dir, tool, arguments, *, credentials=None, fixture_scenario=None):
    async with Client(memory_transport(scope, state_dir, credentials=credentials,
                                       fixture_scenario=fixture_scenario), timeout=100) as client:
        names = [entry.name for entry in await client.list_tools()]
        if tool not in names:
            raise MemorySafetyError("Memory tool was not discovered over MCP.")
        response = await client.call_tool(tool, arguments)
        return {"discovered_tools": names, "result": response.data}


def human_decision(ledger, row):
    print(json.dumps({"exact_fact": row["fact"], "operation_id": row["operation_id"],
                      "destination": asdict(ledger.scope)}, indent=2), flush=True)
    if not sys.stdin.isatty():
        raise MemorySafetyError("Approval requires an interactive human terminal; no approval was recorded.")
    approved = input("Type APPROVE to save this exact fact, or anything else to reject: ").strip() == "APPROVE"
    return ledger.decide(row["operation_id"], approve=approved)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mock", action="store_true", help="Synthetic HTTP responses; no cloud requests.")
    mode.add_argument("--live", action="store_true", help="Hosted Mem0; still requires interactive approval.")
    parser.add_argument("--state-dir", type=Path, default=STATE_DIR)
    parser.add_argument("--cloud-org", default="")
    parser.add_argument("--cloud-project", default="")
    parser.add_argument("--limits-checked", action="store_true", help="Human has checked account quota/rate limits and pricing.")
    sub = parser.add_subparsers(dest="command", required=True)
    propose = sub.add_parser("propose", help="Preview a catalog fact, then approve/reject locally.")
    propose.add_argument("fact_key", choices=FACTS)
    remember = sub.add_parser("remember", help="Save an already-approved operation, or reconcile it using reads.")
    remember.add_argument("operation_id")
    status = sub.add_parser("status", help="Local ledger only; never calls Mem0.")
    status.add_argument("operation_id")
    recall = sub.add_parser("recall", help="Recall only confirmed saved facts in a new MCP session.")
    recall.add_argument("--topic", choices=("preferences", "project"), default="preferences")
    recall.add_argument("--after", help="Require this save to be completed before opening the recall session.")
    args = parser.parse_args(argv)
    if args.live and (not args.cloud_org or not args.cloud_project):
        parser.error("Live mode requires the expected --cloud-org and --cloud-project from your dashboard.")
    scope = Scope(mode="mock_services" if args.mock else "hosted_mem0",
                  cloud_org=args.cloud_org if args.live else "",
                  cloud_project=args.cloud_project if args.live else "")
    ledger = Ledger(args.state_dir / "operations.sqlite3", scope)
    try:
        needs_remote = True
        if args.command == "status":
            row = ledger.get(args.operation_id)
            print(json.dumps({"operation_id": row["operation_id"], "status": row["state"], "fact": row["fact"]}, indent=2))
            return 0
        if args.command == "propose":
            row = ledger.propose(args.fact_key)
            if row["state"] == "proposed":
                row = human_decision(ledger, row)
            if row["state"] in {"rejected", "completed", "failed"}:
                print(json.dumps({"operation_id": row["operation_id"], "status": row["state"]}, indent=2))
                return 0
            tool, arguments = "remember_project_fact", {"operation_id": row["operation_id"]}
        elif args.command == "remember":
            row = ledger.get(args.operation_id)
            needs_remote = row["state"] in {"approved", "in_flight", "pending_remote", "uncertain"}
            tool, arguments = "remember_project_fact", {"operation_id": args.operation_id}
        else:
            if args.after and ledger.get(args.after)["state"] != "completed":
                raise MemorySafetyError("Save completion is not confirmed; no new recall session was started.")
            tool, arguments = "recall_project_facts", {"topic": args.topic}
        credentials = None
        if args.live and needs_remote:
            if not args.limits_checked:
                raise MemorySafetyError("Check account quota, rate limits and pricing first. No Mem0 request was sent.")
            if not sys.stdin.isatty():
                raise MemorySafetyError("Live requests require an interactive human terminal.")
            print("Hosted Mem0 requests may consume credits. A save uses key validation, one add, and confirmation reads.")
            print("Reconciliation uses reads only; recall uses key validation and search. Exact billing is account-dependent.")
            if input("Type RUN to approve these service requests, or anything else to cancel: ").strip() != "RUN":
                print("Cancelled. No Mem0 requests sent.")
                return 0
            credentials = {"MEM0_API_KEY": os.environ.get("MEM0_API_KEY") or getpass("MEM0_API_KEY (hidden, not saved): ")}
        print(json.dumps(asyncio.run(call_memory(scope, args.state_dir, tool, arguments, credentials=credentials)), indent=2))
        return 0
    except MemorySafetyError as error:
        print(str(error))
        return 1
    except Exception:
        print("Memory MCP request failed. Check the local ledger status before continuing; raw errors are hidden.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
