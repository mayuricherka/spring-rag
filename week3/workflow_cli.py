"""Run or resume a single synthetic/public-documentation question through LangGraph."""

import argparse
import asyncio
from getpass import getpass
import json
import os
from pathlib import Path
import sys

from week3.workflow import checkpointed_workflow, execute_run, prepare_run
from week3.memory_ledger import Scope, STATE_DIR
from week3.retrieval import DOCUMENTATION_VERSION, RetrievalError
from week3.workflow_services import (
    MCPReads, WorkflowError, fixture_answer, live_answer,
    required_credentials, workflow_transport,
)


async def run(args):
    # scope is the identity shared by the workflow, memory tools and checkpoints.
    scope = Scope(mode="mock_services" if args.mock else "hosted_mem0",
                  cloud_org=args.cloud_org if args.live else "",
                  cloud_project=args.cloud_project if args.live else "")
    credentials = {}  # Captured by runtime services, never added to state/config.
    def server_connection():
        return workflow_transport(scope, args.state_dir, credentials, mock_failure=args.mock_failure)
    tools = MCPReads(server_connection)
    # Supply the graph with either a prepared answer or the live model adapter.
    async def answer(state):
        if args.mock:
            return await fixture_answer(state, args.state_dir, fail=args.mock_failure == "model")
        return await live_answer(state, credentials)
    async with checkpointed_workflow(args.state_dir, tools, answer, scope,
                                     pause_after_docs=args.pause_after_docs) as graph:
        # A plan describes new input plus remaining nodes; it has not run them yet.
        plan = await prepare_run(graph, scope, args.thread,
                                 question=args.question if args.command == "start" else None,
                                 documentation_version=args.version)
        if args.command == "status":
            # Reading saved progress stops here, before credentials or paid requests.
            return {"status": "pending" if plan["next_nodes"] else "completed",
                    "next_nodes": plan["next_nodes"], "state": plan["state"]}
        required_keys = required_credentials(plan["next_nodes"])
        # Live reads/generation require human cost approval; this graph never saves memory.
        if args.live and required_keys:
            if not args.limits_checked:
                raise WorkflowError("CHECK_CURRENT_PROVIDER_CREDIT_LIMITS_AND_PRICING_FIRST")
            if not sys.stdin.isatty():
                raise WorkflowError("LIVE_REQUESTS_REQUIRE_AN_INTERACTIVE_HUMAN_TERMINAL")
            print("Pending nodes: " + ", ".join(plan["next_nodes"]))
            print("This run/resume may use hosted Mem0 reads, a question embedding, Pinecone reads, and one Nemotron answer.")
            print("Completed checkpointed nodes are skipped. No automatic service retries or memory saves occur.")
            print("Exact charges and current account balances are not verified by this program.")
            if input("Type RUN to approve the remaining service requests, or anything else to cancel: ").strip() != "RUN":
                return {"status": "cancelled", "service_requests_started": False}
            for name in required_keys:
                credentials[name] = os.environ.get(name) or getpass(f"{name} (hidden, not saved): ")
        return await execute_run(graph, plan)


def main(argv=None):
    # Like Java main(String[] args): parse terminal options and choose a command.
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mock", action="store_true", help="Synthetic external services over real MCP; no paid requests.")
    mode.add_argument("--live", action="store_true", help="Requires separate approval for remaining paid requests.")
    parser.add_argument("--state-dir", type=Path, default=STATE_DIR)
    parser.add_argument("--thread", required=True, help="Non-secret stable workflow ID; use the same ID for resume.")
    parser.add_argument("--version", default=DOCUMENTATION_VERSION)
    parser.add_argument("--cloud-org", default="")
    parser.add_argument("--cloud-project", default="")
    parser.add_argument("--limits-checked", action="store_true")
    parser.add_argument("--pause-after-docs", action="store_true", help="Checkpoint and stop before generation.")
    parser.add_argument("--mock-failure", choices=("memory", "docs", "model", "no_docs"), help="Synthetic failure/empty result for practice.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("start").add_argument("question")
    commands.add_parser("resume")
    commands.add_parser("status")
    args = parser.parse_args(argv)
    if args.live and (not args.cloud_org or not args.cloud_project):
        parser.error("Live mode requires expected --cloud-org and --cloud-project.")
    if args.live and args.mock_failure:
        parser.error("--mock-failure is available only with --mock.")
    try:
        # asyncio.run starts the event loop needed by async MCP/model calls.
        result = asyncio.run(run(args))
    except (WorkflowError, RetrievalError) as error:
        print(json.dumps({"status": "blocked", "error_code": str(error)}))
        return 1
    except Exception:
        print(json.dumps({"status": "blocked", "error_code": "WORKFLOW_INTERNAL_FAILURE_RAW_ERRORS_HIDDEN"}))
        return 1
    print(json.dumps(result, indent=2))
    return 1 if result["status"] == "blocked" else 0


if __name__ == "__main__":
    # Entry point when launched with: python -m week3.workflow_cli ...
    raise SystemExit(main())
