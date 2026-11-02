"""Hosted MemoryClient adapter. Never instantiates Mem0's local Memory class."""

from contextlib import contextmanager
import logging
import os
from pathlib import Path
from urllib.parse import quote

import httpx

from week3.memory_ledger import Ledger, MemorySafetyError, Scope, STATE_DIR
from week3.project_memory import ProjectMemory


class HostedBackend:
    def __init__(self, sdk, scope):
        self.sdk, self.scope = sdk, scope

    def add(self, fact, metadata):
        return self.sdk.add(
            messages=[{"role": "user", "content": fact}],
            user_id=self.scope.user_id, app_id=self.scope.project_id,
            metadata=metadata, infer=False,
        )

    def search(self, query):
        response = self.sdk.search(query, filters=self.scope.filters, top_k=5, rerank=False)
        return response["results"]

    def list_facts(self):
        facts = []
        for page in range(1, 4):
            response = self.sdk.get_all(filters=self.scope.filters, page=page, page_size=50)
            facts.extend(response["results"])
            if not response.get("next"):
                return facts
        # Never follow a provider-supplied URL, or assume a truncated list is complete.
        raise MemorySafetyError("Reconciliation read limit reached; save remains uncertain.")

    def event_status(self, event_id):
        # mem0ai 2.2.1 has no get_event method. Use its authenticated HTTP
        # client with the documented endpoint; never return the raw event payload.
        response = self.sdk.client.get(f"/v1/event/{quote(event_id, safe='')}/")
        response.raise_for_status()
        return response.json().get("status")


@contextmanager
def sdk_backend(scope, state_dir, *, api_key, transport=None):
    # The SDK initializes a config file at import time. Keep it isolated and
    # disable telemetry BEFORE import. No existing home config is read.
    os.environ["MEM0_TELEMETRY"] = "false"
    os.environ["MEM0_DIR"] = str(Path(state_dir).resolve() / "sdk")
    for name in ("mem0", "mem0.client.utils", "httpx", "httpcore"):
        logging.getLogger(name).disabled = True
        logging.getLogger(name).setLevel(logging.CRITICAL + 1)
    from mem0 import MemoryClient

    # No HTTP transport retries, redirects, proxy environment, or HTTP debug logs.
    transport = transport if transport is not None else httpx.HTTPTransport(retries=0)
    with httpx.Client(transport=transport, timeout=20, trust_env=False, follow_redirects=False) as http:
        sdk = MemoryClient(api_key=api_key, host="https://api.mem0.ai", client=http)
        if scope.mode == "hosted_mem0" and (
            sdk.org_id != scope.cloud_org or sdk.project_id != scope.cloud_project
        ):
            raise MemorySafetyError("Mem0 key destination differs from the approved cloud project.")
        yield HostedBackend(sdk, scope)


def scope_from_env(*, mock=False):
    return Scope(
        project_id=os.environ.get("SPRING_RAG_MEMORY_PROJECT", "spring-rag-week3-demo"),
        mode="mock_services" if mock else "hosted_mem0",
        cloud_org="" if mock else os.environ.get("MEM0_EXPECTED_ORG", ""),
        cloud_project="" if mock else os.environ.get("MEM0_EXPECTED_PROJECT", ""),
    )


@contextmanager
def live_backend(scope, state_dir):
    # A trusted client sets this only AFTER explicit request/cost approval.
    if os.environ.get("SPRING_RAG_ALLOW_MEM0") != "1":
        raise MemorySafetyError("Live Mem0 requests are disabled.")
    key = os.environ.get("MEM0_API_KEY", "").strip()
    if not key or not scope.cloud_org or not scope.cloud_project:
        raise MemorySafetyError("Mem0 key and expected cloud organization/project are required.")
    with sdk_backend(scope, state_dir, api_key=key) as backend:
        yield backend


def configured_memory(*, mock=False):
    scope = scope_from_env(mock=mock)
    state_dir = Path(os.environ.get("SPRING_RAG_MEMORY_STATE", STATE_DIR))
    ledger = Ledger(state_dir / "operations.sqlite3", scope)
    if mock:
        from week3.mock_mem0 import fixture_backend
        factory = lambda: fixture_backend(scope, state_dir, os.environ.get("MCP_MEMORY_SCENARIO", "success"))
    else:
        factory = lambda: live_backend(scope, state_dir)
    return ProjectMemory(ledger, factory)
