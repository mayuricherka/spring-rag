"""Process restart/resume and live-approval gates using synthetic inputs only."""

import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from week3.mcp_client import ROOT
from week3.workflow_cli import main


class WorkflowCLITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)

    def command(self, *arguments):
        result = subprocess.run([sys.executable, "-m", "week3.workflow_cli", "--mock",
                                 "--state-dir", str(self.path), "--thread", "restart-demo", *arguments],
                                cwd=ROOT, capture_output=True, text=True, timeout=45)
        self.assertNotIn("synthetic-secret", result.stdout + result.stderr)
        return result.returncode, json.loads(result.stdout)

    def test_process_exits_after_retrieval_then_new_process_resumes_without_repeating_reads(self):
        code, paused = self.command("--pause-after-docs", "start", "How do I enable caching?")
        self.assertEqual(code, 0)
        self.assertEqual(paused["status"], "paused")
        self.assertEqual(paused["next_nodes"], ["generate_answer"])
        code, resumed = self.command("resume")
        self.assertEqual(code, 0)
        self.assertEqual(resumed["status"], "completed")
        names = [json.loads(line)["operation"] for line in (self.path / "fixture-workflow-calls.jsonl").read_text().splitlines()]
        self.assertEqual(names, ["recall_project_facts", "search_caching_docs", "nemotron_fixture"])
        self.assertEqual(resumed["state"]["passages"], paused["state"]["passages"])
        before = (self.path / "fixture-workflow-calls.jsonl").read_text()
        self.command("resume")
        self.assertEqual((self.path / "fixture-workflow-calls.jsonl").read_text(), before)

    def test_live_confirmation_rejected_before_keys_or_service_calls(self):
        with patch("sys.stdin.isatty", return_value=True), patch("builtins.input", return_value="CANCEL"), patch("week3.workflow_cli.getpass") as key, patch("week3.workflow_cli.MCPReads") as tools, patch("week3.workflow_cli.live_answer") as model, patch("sys.stdout", new_callable=io.StringIO) as output:
            result = main(["--live", "--state-dir", str(self.path), "--thread", "synthetic-live-test",
                           "--cloud-org", "synthetic-org", "--cloud-project", "synthetic-project",
                           "--limits-checked", "start", "How do I enable caching?"])
        self.assertEqual(result, 0)
        self.assertIn("cancelled", output.getvalue())
        key.assert_not_called()
        tools.return_value.assert_not_called()
        model.assert_not_called()

    def test_resume_missing_thread_does_not_call_services(self):
        code, result = self.command("resume")
        self.assertEqual(code, 1)
        self.assertEqual(result["error_code"], "WORKFLOW_THREAD_NOT_FOUND")
        self.assertFalse((self.path / "fixture-workflow-calls.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
