"""Small protocol probe, not a diagnostic tool or an MCP implementation.

Default execution is offline. --live requires terminal confirmation before
at most two Nebius requests. All request content is a fixed synthetic example.
"""

import argparse
from getpass import getpass
import json


MODEL = "nvidia/Nemotron-3_5-Lightning"
BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
TOOL_NAME = "inspect_synthetic_cache_settings"
RECEIPT = "CACHE_DOCTOR_PROBE_OK"
SAMPLE = {"spring_boot_version": "4.0", "provider": "redis", "ttl": "10m"}


class ProbeFailure(ValueError):
    """A fixed failure code, with no raw API response or credentials."""


def build_request():
    """A Python dict is like a Java Map used as an HTTP request DTO."""
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": (
                "This is a synthetic tool-protocol test. Call the supplied tool "
                "exactly once to inspect the sample. Do not invent tool output. "
                "After receiving the tool result, return its receipt verbatim "
                "and summarize its observation in one sentence. Do not call another tool."
            )},
            {"role": "user", "content": (
                "Inspect this synthetic cache sample using the tool: "
                + json.dumps(SAMPLE)
            )},
        ],
        "tools": [{"type": "function", "function": {
            "name": TOOL_NAME,
            "description": "Inspect the supplied synthetic Spring cache settings.",
            "parameters": {
                "type": "object",
                "properties": {key: {"type": "string"} for key in SAMPLE},
                "required": list(SAMPLE),
                "additionalProperties": False,
            },
        }}],
        "tool_choice": "auto",
        "max_tokens": 256,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }


def token_usage(response):
    if response.usage is None:
        return None
    return {key: getattr(response.usage, key, None) for key in (
        "prompt_tokens", "completion_tokens", "total_tokens"
    )}


def run_probe(client):
    """Check one tool-call/result round trip; never execute a tool or save data."""
    request = build_request()
    first = client.chat.completions.create(**request)
    if len(first.choices) != 1:
        raise ProbeFailure("unexpected_choice_count")
    choice = first.choices[0]
    calls = choice.message.tool_calls or []
    if choice.finish_reason != "tool_calls" or len(calls) != 1:
        raise ProbeFailure("expected_one_structured_tool_call")
    call = calls[0]
    if call.type != "function" or not call.id or call.function.name != TOOL_NAME:
        raise ProbeFailure("unexpected_tool_identity")
    try:
        arguments = json.loads(call.function.arguments)
    except (TypeError, ValueError):
        raise ProbeFailure("invalid_tool_json") from None
    if arguments != SAMPLE:
        raise ProbeFailure("tool_arguments_differ_from_synthetic_sample")

    # A fixed fixture tests the wire protocol. No checker or MCP call occurs here.
    fixture = {
        "receipt": RECEIPT,
        "status": "synthetic_fixture",
        "observation": "TTL is explicitly set; suitability depends on application requirements.",
    }
    messages = request["messages"] + [
        {"role": "assistant", "content": choice.message.content,
         "tool_calls": [call.model_dump(exclude_none=True)]},
        {"role": "tool", "tool_call_id": call.id, "content": json.dumps(fixture)},
    ]
    second = client.chat.completions.create(**{**request, "messages": messages})
    if len(second.choices) != 1:
        raise ProbeFailure("unexpected_final_choice_count")
    final = second.choices[0]
    if final.finish_reason != "stop" or final.message.tool_calls:
        raise ProbeFailure("final_answer_not_complete")
    if RECEIPT not in (final.message.content or ""):
        raise ProbeFailure("tool_result_receipt_missing")
    return {
        "status": "passed_one_synthetic_round_trip",
        "requested_model": MODEL,
        "returned_models": [first.model, second.model],
        "requests": 2,
        "usage_by_request": [token_usage(first), token_usage(second)],
        "mcp_tested": False,
        "cost": "not_calculated; verify current provider pricing separately",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Opt in to a billable protocol test.")
    args = parser.parse_args(argv)
    if not args.live:
        print(json.dumps({
            "status": "offline_only; live_model_support_unverified",
            "model": MODEL,
            "request": build_request(),
            "maximum_live_requests_if_approved": 2,
        }, indent=2))
        return 0

    print("This sends up to TWO billable Nebius requests, with no automatic retries.")
    print("Current account pricing and credit balance have not been verified.")
    if input("Type RUN to approve this test, or anything else to cancel: ").strip() != "RUN":
        print("Cancelled. No requests sent.")
        return 0

    # Never read the project's secrets file or put a key in a command-line argument.
    from openai import OpenAI

    try:
        api_key = getpass("Nebius API key (hidden, not saved): ").strip()
        if not api_key:
            print("No API key entered. No requests sent.")
            return 1
        with OpenAI(base_url=BASE_URL, api_key=api_key, max_retries=0, timeout=30.0) as client:
            print(json.dumps(run_probe(client), indent=2))
    except ProbeFailure as error:
        print(json.dumps({"status": "failed", "reason": str(error)}))
        return 1
    except Exception as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__}))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
