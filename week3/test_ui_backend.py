"""Durable UI actions: stale previews, concurrent events and cross-view dedupe."""

from concurrent.futures import ThreadPoolExecutor
import tempfile
from pathlib import Path
from threading import Event
import unittest
from unittest.mock import patch

from week3.memory_client import call_memory
from week3.memory_ledger import MemorySafetyError, Scope
from week3.mock_mem0 import Mem0Fixture
from week3.ui_backend import UIBackend


class UIBackendTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.backend = UIBackend(Scope(), self.path)
        self.view = self.backend.store.resolve()

    def test_stale_approve_cannot_approve_another_preview(self):
        old = self.backend.preview(self.view["id"], "java_comparisons")
        new = self.backend.preview(self.view["id"], "redis")
        with self.assertRaises(MemorySafetyError):
            self.backend.approve(self.view["id"], old["operation_id"])
        self.assertEqual(self.backend.ledger().get(new["operation_id"])["state"], "proposed")
        self.assertFalse((self.path / "fixture-remote.sqlite3").exists())

    def test_completed_save_deduplicates_across_different_ui_views(self):
        row = self.backend.preview(self.view["id"], "java_comparisons")
        self.backend.approve(self.view["id"], row["operation_id"])
        other = self.backend.store.resolve()
        replay = self.backend.preview(other["id"], "java_comparisons")
        self.assertEqual(replay["operation_id"], row["operation_id"])
        self.assertEqual(self.backend.approve(other["id"], row["operation_id"])["state"], "completed")
        self.assertEqual(Mem0Fixture(self.path / "fixture-remote.sqlite3").calls("/v3/memories/add/"), 1)

    def test_concurrent_duplicate_ui_actions_do_not_issue_another_save(self):
        row = self.backend.preview(self.view["id"], "java_comparisons")
        entered, release = Event(), Event()
        async def delayed(*args, **kwargs):
            entered.set()
            release.wait(timeout=10)
            return await call_memory(*args, **kwargs)
        with patch("week3.ui_backend.call_memory", side_effect=delayed), ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.backend.approve, self.view["id"], row["operation_id"])
            try:
                self.assertTrue(entered.wait(timeout=10))
                with self.assertRaisesRegex(MemorySafetyError, "already running"):
                    self.backend.approve(self.view["id"], row["operation_id"])
            finally:
                release.set()
            self.assertEqual(future.result(timeout=30)["state"], "completed")
        self.assertEqual(Mem0Fixture(self.path / "fixture-remote.sqlite3").calls("/v3/memories/add/"), 1)


if __name__ == "__main__":
    unittest.main()
