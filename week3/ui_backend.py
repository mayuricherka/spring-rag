"""Trusted local UI actions. Credentials stay in action-local variables.

The view record is like a Java UI model containing references to durable domain
records. Approval authority stays in Ledger, not a Streamlit widget value.
"""

import asyncio
from contextlib import closing, contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import sqlite3
from uuid import uuid4

from week3.workflow import checkpointed_workflow, execute_run, initial_state, prepare_run
from week3.workflow_services import (
    MCPReads, fixture_answer, live_answer, required_credentials, workflow_transport,
)
from week3.memory_client import call_memory
from week3.memory_ledger import FACTS, Ledger, MemorySafetyError, Scope, STATE_DIR


def runtime_value(name, secrets_getter=None):
    value = os.environ.get(name)
    if not value and secrets_getter is not None:
        try:
            value = secrets_getter(name)
        except Exception:
            value = None
    return value.strip() if isinstance(value, str) else ""


def load_credentials(names, secrets_getter=None):
    values = {name: runtime_value(name, secrets_getter) for name in names}
    if not all(values.values()):
        raise MemorySafetyError("Live credentials are not configured on the server.")
    return values


def ui_scope(mode, secrets_getter=None):
    if mode == "mock":
        return Scope()
    org = runtime_value("MEM0_EXPECTED_ORG", secrets_getter)
    project = runtime_value("MEM0_EXPECTED_PROJECT", secrets_getter)
    for name, value, pattern in (
        ("MEM0_EXPECTED_ORG", org, r"org_[A-Za-z0-9]+"),
        ("MEM0_EXPECTED_PROJECT", project, r"proj_[A-Za-z0-9]+"),
    ):
        if not value:
            raise MemorySafetyError(f"Missing server setting: {name}.")
        if not re.fullmatch(pattern, value):
            raise MemorySafetyError(f"Invalid server setting: {name}. Use the Mem0 cloud ID, not its display name.")
    return Scope(mode="hosted_mem0", cloud_org=org, cloud_project=project)


class ViewStore:
    def __init__(self, state_dir, scope):
        self.state_dir, self.scope = Path(state_dir), scope
        self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = self.state_dir / "ui-sessions.sqlite3"
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(fd)
        os.chmod(self.path, 0o600)
        with closing(self.connect()) as db, db:
            db.execute("""CREATE TABLE IF NOT EXISTS ui_sessions (
                id TEXT PRIMARY KEY, scope_key TEXT NOT NULL, thread_id TEXT,
                question TEXT, proposal_id TEXT, recall_json TEXT
            )""")

    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        return db

    def get(self, view_id):
        if not isinstance(view_id, str) or not re.fullmatch(r"[a-f0-9]{32}", view_id):
            raise MemorySafetyError("Session unavailable. Start a new session.")
        with closing(self.connect()) as db:
            row = db.execute("SELECT * FROM ui_sessions WHERE id=? AND scope_key=?",
                             (view_id, self.scope.key)).fetchone()
        if row is None:
            raise MemorySafetyError("Session unavailable in this project.")
        return dict(row)

    def resolve(self, view_id=None):
        if view_id:
            try:
                return self.get(view_id)
            except MemorySafetyError:
                pass
        view_id = uuid4().hex
        with closing(self.connect()) as db, db:
            db.execute("INSERT INTO ui_sessions(id,scope_key) VALUES (?,?)", (view_id, self.scope.key))
        return self.get(view_id)

    def set_run(self, view_id, question):
        thread_id = "ui-" + uuid4().hex
        with closing(self.connect()) as db, db:
            db.execute("UPDATE ui_sessions SET thread_id=?,question=? WHERE id=? AND scope_key=?",
                       (thread_id, question, view_id, self.scope.key))
        return self.get(view_id)

    def set_proposal(self, view_id, operation_id):
        with closing(self.connect()) as db, db:
            db.execute("UPDATE ui_sessions SET proposal_id=? WHERE id=? AND scope_key=?",
                       (operation_id, view_id, self.scope.key))

    def set_recall(self, view_id, result):
        with closing(self.connect()) as db, db:
            db.execute("UPDATE ui_sessions SET recall_json=? WHERE id=? AND scope_key=?",
                       (json.dumps(result), view_id, self.scope.key))

    @contextmanager
    def action(self, view_id):
        self.get(view_id)  # Validate scope and ID before constructing a path.
        folder = self.state_dir / "ui-locks"
        folder.mkdir(exist_ok=True, mode=0o700)
        fd = os.open(folder / (view_id + ".lock"), os.O_CREAT | os.O_RDWR, 0o600)
        with os.fdopen(fd, "a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise MemorySafetyError("An action is already running in this session. Wait, then refresh.") from None
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)


class UIBackend:
    def __init__(self, scope, state_dir=STATE_DIR, *, secrets_getter=None):
        self.scope, self.state_dir = scope, Path(state_dir)
        self.secrets_getter = secrets_getter
        self.store = ViewStore(self.state_dir, scope)

    def ledger(self):
        return Ledger(self.state_dir / "operations.sqlite3", self.scope)

    def credentials(self, names, permit_live):
        if self.scope.mode == "mock_services":
            return {}
        if permit_live is not True:
            raise MemorySafetyError("Approve live service requests using the checkbox before this action.")
        return load_credentials(names, self.secrets_getter)

    async def _workflow(self, view, *, inspect=False, permit_live=False):
        credentials = {}
        failure = os.environ.get("SPRING_RAG_UI_MOCK_FAILURE") if self.scope.mode == "mock_services" else None
        def server_connection():
            return workflow_transport(self.scope, self.state_dir, credentials, mock_failure=failure)
        tools = MCPReads(server_connection)
        async def answer(state):
            if self.scope.mode == "mock_services":
                return await fixture_answer(state, self.state_dir, fail=failure == "model")
            return await live_answer(state, credentials)
        async with checkpointed_workflow(self.state_dir, tools, answer, self.scope) as graph:
            config = {"configurable": {"thread_id": view["thread_id"]}}
            snapshot = await graph.aget_state(config)
            if snapshot.values and snapshot.values.get("scope_key") != self.scope.key:
                raise MemorySafetyError("The saved answer belongs to another project or service mode.")
            if inspect:
                return {"status": "pending" if not snapshot.values or snapshot.next else "completed",
                        "next_nodes": list(snapshot.next), "state": snapshot.values}
            plan = await prepare_run(graph, self.scope, view["thread_id"],
                                     question=None if snapshot.values else view["question"])
            names = required_credentials(plan["next_nodes"])
            if names:
                credentials.update(self.credentials(names, permit_live))
            return await execute_run(graph, plan)

    def result(self, view_id):
        view = self.store.get(view_id)
        return asyncio.run(self._workflow(view, inspect=True)) if view["thread_id"] else None

    def ask(self, view_id, question, *, permit_live=False):
        question = initial_state(question, self.scope)["question"]  # Validate before durable UI storage.
        with self.store.action(view_id):
            view = self.store.get(view_id)
            if view["thread_id"] and view["question"] == question:
                return self.result(view_id)  # Double-submit/rerun reuses the current run.
            if self.scope.mode != "mock_services" and permit_live is not True:
                raise MemorySafetyError("Approve live service requests using the checkbox before this action.")
            view = self.store.set_run(view_id, question)
            return asyncio.run(self._workflow(view, permit_live=permit_live))

    def resume(self, view_id, *, permit_live=False):
        with self.store.action(view_id):
            view = self.store.get(view_id)
            if not view["thread_id"]:
                raise MemorySafetyError("Ask a question first.")
            return asyncio.run(self._workflow(view, permit_live=permit_live))

    def proposal(self, view_id):
        operation_id = self.store.get(view_id)["proposal_id"]
        return self.ledger().get(operation_id) if operation_id else None

    def preview(self, view_id, fact_key):
        with self.store.action(view_id):
            current = self.proposal(view_id)
            if current and current["state"] in {"approved", "in_flight", "pending_remote", "uncertain"}:
                raise MemorySafetyError("Resolve the current save before proposing a different fact.")
            row = self.ledger().propose(fact_key)
            self.store.set_proposal(view_id, row["operation_id"])
            return row

    def _reviewed_row(self, view_id, operation_id):
        if self.store.get(view_id)["proposal_id"] != operation_id:
            raise MemorySafetyError("The proposal changed. Review the current preview first.")
        return self.ledger().get(operation_id)

    def reject(self, view_id, operation_id):
        with self.store.action(view_id):
            self._reviewed_row(view_id, operation_id)
            return self.ledger().decide(operation_id, approve=False)

    def approve(self, view_id, operation_id, *, permit_live=False):
        """Trusted button action, not an MCP argument or a model decision."""
        with self.store.action(view_id):
            row = self._reviewed_row(view_id, operation_id)
            if row["state"] in {"completed", "rejected", "failed"}:
                return row
            credentials = self.credentials(["MEM0_API_KEY"], permit_live)
            if row["state"] == "proposed":
                row = self.ledger().decide(operation_id, approve=True)
            scenario = os.environ.get("SPRING_RAG_UI_MOCK_MEMORY_SCENARIO") if self.scope.mode == "mock_services" else None
            asyncio.run(call_memory(self.scope, self.state_dir, "remember_project_fact",
                                    {"operation_id": operation_id}, credentials=credentials,
                                    fixture_scenario=scenario))
            return self.ledger().get(operation_id)

    def new_session(self, view_id, *, permit_live=False):
        with self.store.action(view_id):
            row = self.proposal(view_id)
            if row and row["state"] not in {"completed", "rejected", "failed"}:
                raise MemorySafetyError("Complete or reject the pending proposal before starting recall.")
            credentials = self.credentials(["MEM0_API_KEY"], permit_live)
            new_view = self.store.resolve()
            try:
                result = asyncio.run(call_memory(self.scope, self.state_dir, "recall_project_facts", {},
                                                 credentials=credentials))["result"]
                if (result.get("service_mode") != self.scope.mode or result.get("user_id") != self.scope.user_id
                        or result.get("project_id") != self.scope.project_id):
                    raise MemorySafetyError("Recall scope mismatch.")
                facts = []
                for item in result.get("facts", []):
                    try:
                        approved = self.ledger().get(item["operation_id"])
                    except (MemorySafetyError, KeyError, TypeError):
                        continue
                    if (approved["state"] == "completed" and item.get("id") == approved["memory_id"]
                            and item.get("fact") == approved["fact"] and approved["fact"] in FACTS.values()):
                        facts.append({"fact": approved["fact"], "id": approved["memory_id"]})
                recall = {"status": "ok" if facts else "no_facts", "facts": facts}
            except Exception:
                recall = {"status": "unavailable", "facts": []}
            self.store.set_recall(new_view["id"], recall)
            return self.store.get(new_view["id"])
