"""Synthetic hosted-service fixture, NOT a Mem0 local vector store.

SQLite here only simulates remote HTTP persistence between test server processes.
The production adapter still calls the real hosted SDK through MockTransport.
"""

from contextlib import closing, contextmanager
import json
from pathlib import Path
import sqlite3

import httpx

from week3.mem0_hosted import sdk_backend


class Mem0Fixture:
    def __init__(self, path, scenario="success"):
        self.path, self.scenario = Path(path), scenario
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS facts (id TEXT PRIMARY KEY, body TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS calls (path TEXT NOT NULL)")

    def calls(self, path):
        with closing(sqlite3.connect(self.path)) as db:
            return db.execute("SELECT count(*) FROM calls WHERE path=?", (path,)).fetchone()[0]

    def items(self):
        with closing(sqlite3.connect(self.path)) as db:
            return [json.loads(row[0]) for row in db.execute("SELECT body FROM facts")]

    def __call__(self, request):
        path = request.url.path
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("INSERT INTO calls VALUES (?)", (path,))
        if path == "/v1/ping/":
            return httpx.Response(200, json={"org_id": "fixture-org", "project_id": "fixture-project"})
        if path.startswith("/v1/event/"):
            if self.scenario == "event_error":
                return httpx.Response(503, json={"detail": "synthetic-secret-must-not-leak"})
            if self.scenario == "unknown_event":
                return httpx.Response(200, json={"status": "synthetic-secret-must-not-leak"})
            status = {"pending": "RUNNING", "failed_event": "FAILED"}.get(self.scenario, "SUCCEEDED")
            return httpx.Response(200, json={"status": status})
        body = json.loads(request.content)
        if path == "/v3/memories/add/":
            if self.scenario == "timeout_before":
                raise httpx.ReadTimeout("synthetic-secret-must-not-leak", request=request)
            operation_id = body["metadata"]["operation_id"]
            item = {"id": "memory-" + operation_id, "memory": body["messages"][0]["content"],
                    "user_id": body["user_id"], "app_id": body["app_id"],
                    "metadata": body["metadata"], "score": 0.93}
            with closing(sqlite3.connect(self.path)) as db, db:
                # Use INSERT, not replace: a duplicate add is visible as a test failure.
                db.execute("INSERT INTO facts VALUES (?, ?)", (item["id"], json.dumps(item)))
            if self.scenario == "timeout_after":
                raise httpx.ReadTimeout("synthetic-secret-must-not-leak", request=request)
            if self.scenario in {"pending", "completed_event", "failed_event", "event_error", "unknown_event"}:
                return httpx.Response(200, json={"event_id": "fixture-event", "status": "PENDING"})
            return httpx.Response(200, json={"message": "Synthetic save", "results": [item]})
        if path in {"/v3/memories/", "/v3/memories/search/"}:
            if self.scenario == "read_error":
                return httpx.Response(503, json={"detail": "synthetic-secret-must-not-leak"})
            filters = body["filters"]["AND"]
            user_id, app_id = filters[0]["user_id"], filters[1]["app_id"]
            results = [item for item in self.items()
                       if item["user_id"] == user_id and item["app_id"] == app_id]
            if self.scenario == "wrong_scope":
                results = self.items()  # Adversarial service ignores filters.
            return httpx.Response(200, json={"results": results, "count": len(results), "next": None})
        return httpx.Response(404, json={"detail": "Unrecognized synthetic route"})


@contextmanager
def fixture_backend(scope, state_dir, scenario="success"):
    fixture = Mem0Fixture(Path(state_dir) / "fixture-remote.sqlite3", scenario)
    with sdk_backend(scope, state_dir, api_key="synthetic-not-a-real-key",
                     transport=httpx.MockTransport(fixture)) as backend:
        yield backend
