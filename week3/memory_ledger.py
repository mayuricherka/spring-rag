"""Trusted local approval/operation log, separate from hosted long-term memory.

Like a Java repository with a conditional UPDATE: claiming an approved row is
atomic. A tool argument cannot create or approve a row.
"""

from contextlib import closing
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import sqlite3


STATE_DIR = Path(__file__).resolve().parent / ".state"
FACTS = {
    "java_comparisons": "Prefer explanations with Java comparisons.",
    "spring_boot_4": "The synthetic demo project uses Spring Boot 4.0.",
    "redis": "The synthetic demo project uses Redis for caching.",
}


class MemorySafetyError(ValueError):
    """A safe message that may be shown to the user; never a provider exception."""


@dataclass(frozen=True)
class Scope:
    user_id: str = "week3-synthetic-user"
    project_id: str = "spring-rag-week3-demo"
    mode: str = "mock_services"
    cloud_org: str = ""
    cloud_project: str = ""

    @property
    def key(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    @property
    def filters(self):
        # App ID is an application entity, distinct from the cloud project ID.
        return {"AND": [{"user_id": self.user_id}, {"app_id": self.project_id}]}


class Ledger:
    def __init__(self, path, scope):
        self.path, self.scope = Path(path), scope
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        # Create privately before sqlite opens it, rather than chmod after writing.
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(fd)
        os.chmod(self.path, 0o600)
        with closing(self.connect()) as db, db:
            db.execute("""CREATE TABLE IF NOT EXISTS operations (
                operation_id TEXT PRIMARY KEY, scope_key TEXT NOT NULL,
                fact_key TEXT NOT NULL, fact TEXT NOT NULL,
                state TEXT NOT NULL, event_id TEXT, memory_id TEXT,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )""")

    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        return db

    def propose(self, fact_key):
        if fact_key not in FACTS:
            raise MemorySafetyError("Choose a fact from the synthetic catalog; free-form content is disabled.")
        fact = FACTS[fact_key]
        operation_id = hashlib.sha256((self.scope.key + "\0" + fact).encode()).hexdigest()
        with closing(self.connect()) as db, db:
            db.execute("""INSERT OR IGNORE INTO operations
                (operation_id, scope_key, fact_key, fact, state) VALUES (?, ?, ?, ?, 'proposed')""",
                (operation_id, self.scope.key, fact_key, fact))
        return self.get(operation_id)

    def get(self, operation_id):
        if not isinstance(operation_id, str) or len(operation_id) != 64:
            raise MemorySafetyError("Unknown operation in this project scope.")
        with closing(self.connect()) as db:
            row = db.execute("SELECT * FROM operations WHERE operation_id=? AND scope_key=?",
                             (operation_id, self.scope.key)).fetchone()
        if row is None:
            raise MemorySafetyError("Unknown operation in this project scope.")
        return dict(row)

    def decide(self, operation_id, *, approve):
        """Trusted terminal/UI only. Deliberately not exposed as an MCP tool."""
        self.get(operation_id)
        with closing(self.connect()) as db, db:
            db.execute("""UPDATE operations SET state=?, updated_at=CURRENT_TIMESTAMP
                WHERE operation_id=? AND scope_key=? AND state='proposed'""",
                ("approved" if approve else "rejected", operation_id, self.scope.key))
        return self.get(operation_id)

    def claim(self, operation_id):
        with closing(self.connect()) as db, db:
            result = db.execute("""UPDATE operations SET state='in_flight', updated_at=CURRENT_TIMESTAMP
                WHERE operation_id=? AND scope_key=? AND state='approved'""",
                (operation_id, self.scope.key))
            return result.rowcount == 1

    def record(self, operation_id, state, *, event_id=None, memory_id=None):
        if state not in {"pending_remote", "completed", "uncertain", "failed"}:
            raise MemorySafetyError("Unsupported operation transition.")
        with closing(self.connect()) as db, db:
            # Terminal decisions cannot be undone by a concurrent reconciliation.
            db.execute("""UPDATE operations SET state=?, event_id=COALESCE(?, event_id),
                memory_id=COALESCE(?, memory_id), updated_at=CURRENT_TIMESTAMP
                WHERE operation_id=? AND scope_key=?
                AND state IN ('in_flight', 'pending_remote', 'uncertain')""",
                (state, event_id, memory_id, operation_id, self.scope.key))
        return self.get(operation_id)
