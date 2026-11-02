"""Offline protocol tests. No Nebius, Pinecone, MCP, Mem0, or Langfuse requests."""

from contextlib import redirect_stdout
import io
import json
import unittest
from unittest.mock import Mock, patch

from openai.types.chat import ChatCompletion

from week3.probe_tool_calling import (
    MODEL, RECEIPT, SAMPLE, TOOL_NAME, ProbeFailure, main, run_probe,
)


def response(tool=True, arguments=None, name=TOOL_NAME, content=None, finish=None):
    message = {"role": "assistant", "content": content}
    if tool:
        message["tool_calls"] = [{
            "id": "synthetic-call-1", "type": "function", "function": {
                "name": name,
                "arguments": json.dumps(SAMPLE) if arguments is None else arguments,
            },
        }]
    return ChatCompletion.model_validate({
        "id": "synthetic-response", "object": "chat.completion", "created": 0,
        "model": MODEL,
        "choices": [{"index": 0, "message": message,
                     "finish_reason": finish or ("tool_calls" if tool else "stop")}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    })


class ProbeTests(unittest.TestCase):
    def test_valid_round_trip_and_tool_call_id(self):
        client = Mock()
        client.chat.completions.create.side_effect = [
            response(), response(tool=False, content=RECEIPT + " TTL is explicitly set."),
        ]
        report = run_probe(client)
        self.assertEqual(report["requests"], 2)
        self.assertFalse(report["mcp_tested"])
        self.assertEqual(report["usage_by_request"][0]["total_tokens"], 15)
        calls = client.chat.completions.create.call_args_list
        self.assertEqual(calls[0].kwargs["tool_choice"], "auto")
        self.assertEqual(calls[0].kwargs["model"], MODEL)
        self.assertEqual(len(calls[0].kwargs["messages"]), 2)
        self.assertEqual(calls[1].kwargs["messages"][-1]["tool_call_id"], "synthetic-call-1")

    def test_invalid_first_responses_stop_before_followup(self):
        invalid = [
            response(tool=False, content="Use Redis."),
            response(arguments="not-json"),
            response(name="save_report"),
            response(arguments=json.dumps({**SAMPLE, "unexpected": "value"})),
            response(finish="length"),
        ]
        for first in invalid:
            with self.subTest(first=first.choices[0].finish_reason):
                client = Mock()
                client.chat.completions.create.return_value = first
                with self.assertRaises(ProbeFailure):
                    run_probe(client)
                self.assertEqual(client.chat.completions.create.call_count, 1)

    def test_invalid_followups_fail_without_a_third_request(self):
        for final in [response(), response(tool=False, content="No receipt."),
                      response(tool=False, content=RECEIPT, finish="length")]:
            client = Mock()
            client.chat.completions.create.side_effect = [response(), final]
            with self.assertRaises(ProbeFailure):
                run_probe(client)
            self.assertEqual(client.chat.completions.create.call_count, 2)

    def test_default_is_offline(self):
        with patch("openai.OpenAI") as client, redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main([]), 0)
        client.assert_not_called()
        self.assertIn("live_model_support_unverified", output.getvalue())

    def test_rejected_live_confirmation_sends_nothing(self):
        with patch("builtins.input", return_value="NO"), \
             patch("openai.OpenAI") as client, \
             patch("week3.probe_tool_calling.getpass") as key_input, \
             redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--live"]), 0)
        client.assert_not_called()
        key_input.assert_not_called()


if __name__ == "__main__":
    unittest.main()
