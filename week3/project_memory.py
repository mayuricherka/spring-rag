"""Approval-gated memory service. Provider responses are untrusted data."""

import math
import re

import httpx

from week3.memory_ledger import FACTS, MemorySafetyError


SAFE_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")


def safe_id(value):
    return value if isinstance(value, str) and SAFE_ID.fullmatch(value) else None


def synthetic_marker_form(metadata):
    """Fixed categories only: unknown remote values must never be echoed."""
    if "synthetic" not in metadata:
        return "missing"
    value = metadata["synthetic"]
    if value is True:
        return "boolean_true"
    if value is False:
        return "boolean_false"
    if value is None:
        return "null"
    if type(value) is str and value.casefold() in {"true", "false"}:
        return "text_" + value.casefold()
    if type(value) is int and value in (0, 1):
        return "integer_one" if value == 1 else "integer_zero"
    return "other"


def receipt(row, scope):
    return {"operation_id": row["operation_id"], "fact": row["fact"],
            "status": row["state"], "service_mode": scope.mode,
            "user_id": scope.user_id, "project_id": scope.project_id,
            "memory_id": row["memory_id"]}


def error_diagnostic(error, stage):
    """Extract only fixed categories and numeric status; never exception text."""
    status = None
    current = error
    for _ in range(3):
        if current is None:
            break
        candidate = getattr(getattr(current, "response", None), "status_code", None)
        info = getattr(current, "debug_info", None)
        if candidate is None and isinstance(info, dict):
            candidate = info.get("status_code")
        if type(candidate) is int and 100 <= candidate <= 599:
            status = candidate
            break
        current = current.__cause__ or current.__context__
    if status is not None:
        code = "HTTP_ERROR"
    elif isinstance(error, httpx.TimeoutException) or getattr(error, "error_code", None) == "NET_TIMEOUT":
        code = "TIMEOUT"
    elif isinstance(error, httpx.RequestError) or type(error).__name__ == "NetworkError":
        code = "NETWORK_ERROR"
    elif isinstance(error, MemorySafetyError):
        code = "LOCAL_SAFETY_CHECK_FAILED"
    elif isinstance(error, (KeyError, TypeError, ValueError)):
        code = "RESPONSE_SHAPE_ERROR"
    else:
        code = "UNEXPECTED_ERROR"
    result = {"stage": stage, "code": code}
    if status is not None:
        result["http_status"] = status
    return result


def diagnosed_receipt(row, scope, diagnostic):
    return {**receipt(row, scope), "diagnostic": diagnostic}


class ProjectMemory:
    def __init__(self, ledger, backend_factory):
        self.ledger, self.scope = ledger, ledger.scope
        self.backend_factory = backend_factory

    def metadata(self, row):
        return {"operation_id": row["operation_id"], "scope_key": self.scope.key,
                "project_id": self.scope.project_id, "synthetic": True}

    def metadata_checks(self, actual, row):
        actual = actual if isinstance(actual, dict) else {}
        # Require our saved fields, including their types. Additional provider
        # fields do not change the approved fact or its scope.
        checks = {key: key in actual and type(actual[key]) is type(value) and actual[key] == value
                  for key, value in self.metadata(row).items()}
        # This describes synthetic demo data; it cannot authorize a save. Accept
        # only boolean true or literal true text, never truthiness or integer 1.
        checks["synthetic"] = synthetic_marker_form(actual) in {"boolean_true", "text_true"}
        return checks

    def matches(self, item, row):
        return (isinstance(item, dict) and safe_id(item.get("id")) is not None
                and item.get("memory") == row["fact"]
                and all(self.metadata_checks(item.get("metadata"), row).values())
                and item.get("user_id", self.scope.user_id) == self.scope.user_id
                and item.get("app_id", self.scope.project_id) == self.scope.project_id)

    def remember(self, operation_id):
        row = self.ledger.get(operation_id)
        if row["state"] in {"proposed", "rejected", "completed", "failed"}:
            return receipt(row, self.scope)  # No client initialization or remote call.
        stage = "initialize"
        try:
            with self.backend_factory() as backend:
                # The client may validate its key before this atomic claim. If
                # initialization fails, approval is retained and add was not sent.
                if self.ledger.claim(operation_id):
                    stage = "add"
                    try:
                        response = backend.add(row["fact"], self.metadata(row))
                        if not isinstance(response, dict):
                            raise ValueError("Unrecognized response")
                        event_id = safe_id(response.get("event_id"))
                        status = response.get("status")
                        state = "failed" if status == "FAILED" else "pending_remote"
                        self.ledger.record(operation_id, state, event_id=event_id)
                    except Exception as error:
                        # The request may have committed. Never issue add again.
                        self.ledger.record(operation_id, "uncertain")
                        return diagnosed_receipt(self.ledger.get(operation_id), self.scope,
                                                 error_diagnostic(error, stage))
                row = self.ledger.get(operation_id)
                if row["state"] in {"completed", "failed"}:
                    return receipt(row, self.scope)
                if row["event_id"]:
                    stage = "event_status"
                    status = backend.event_status(row["event_id"])
                    if status in {"PENDING", "RUNNING"}:
                        return receipt(self.ledger.record(operation_id, "pending_remote"), self.scope)
                    if status == "FAILED":
                        return receipt(self.ledger.record(operation_id, "failed"), self.scope)
                    if status != "SUCCEEDED":
                        return diagnosed_receipt(self.ledger.record(operation_id, "uncertain"), self.scope,
                                                 {"stage": stage, "code": "EVENT_STATUS_UNRECOGNIZED"})
                # Read back the exact text, scope and operation, even for a sync
                # infer=False response. An empty search is not proof of failure.
                stage = "read_back"
                items = backend.list_facts()
                found = [item for item in items if self.matches(item, row)]
                if len(found) == 1:
                    row = self.ledger.record(operation_id, "completed", memory_id=found[0]["id"])
                else:
                    row = self.ledger.record(operation_id, "uncertain")
                    same_operation = [item for item in items if isinstance(item, dict)
                                      and isinstance(item.get("metadata"), dict)
                                      and item["metadata"].get("operation_id") == operation_id]
                    diagnostic = {
                        "stage": stage,
                        "code": "MULTIPLE_EXACT_MATCHES" if found else "NO_EXACT_MATCH",
                        "returned_count": len(items),
                        "same_operation_count": len(same_operation),
                        "exact_text_count": sum(item.get("memory") == row["fact"] for item in same_operation),
                        "exact_metadata_count": sum(item.get("metadata") == self.metadata(row)
                                                    for item in same_operation),
                    }
                    if same_operation:
                        # Counts for fixed field names only; no remote values or
                        # provider-added metadata are returned or persisted.
                        diagnostic["metadata_field_match_counts"] = {
                            field: sum(self.metadata_checks(item["metadata"], row)[field]
                                       for item in same_operation)
                            for field in self.metadata(row)
                        }
                        forms = [synthetic_marker_form(item["metadata"]) for item in same_operation]
                        diagnostic["synthetic_marker_forms"] = {form: forms.count(form) for form in sorted(set(forms))}
                    return diagnosed_receipt(row, self.scope, diagnostic)
                return receipt(row, self.scope)
        except Exception as error:
            # Hide SDK error text. A claimed operation remains non-retryable.
            row = self.ledger.get(operation_id)
            if row["state"] in {"in_flight", "pending_remote", "uncertain"}:
                return diagnosed_receipt(self.ledger.record(operation_id, "uncertain"), self.scope,
                                         error_diagnostic(error, stage))
            diagnostic = error_diagnostic(error, stage)
            raise MemorySafetyError("Memory service unavailable; no save was started. "
                                    f"Diagnostic: {diagnostic['code']}; HTTP status: {diagnostic.get('http_status', 'unavailable')}. "
                                    "Raw errors are hidden.") from None

    def recall(self, topic="preferences"):
        # Catalog topics prevent raw configs, keys or code entering remote search.
        queries = {"preferences": "Preferred explanation style", "project": "Synthetic Spring caching project facts"}
        if topic not in queries:
            raise MemorySafetyError("Recall topic must be preferences or project.")
        try:
            with self.backend_factory() as backend:
                results = backend.search(queries[topic])
            facts = []
            for item in results:
                if not isinstance(item, dict) or not isinstance(item.get("metadata"), dict):
                    continue
                try:
                    row = self.ledger.get(item["metadata"].get("operation_id"))
                except MemorySafetyError:
                    continue
                if (row["state"] != "completed" or row["fact"] not in FACTS.values()
                        or row["memory_id"] != item.get("id") or not self.matches(item, row)):
                    continue
                score = item.get("score")
                if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                    score = None
                facts.append({"id": item["id"], "fact": row["fact"], "score": score,
                              "operation_id": row["operation_id"]})
            return {"status": "ok" if facts else "no_facts", "service_mode": self.scope.mode,
                    "user_id": self.scope.user_id, "project_id": self.scope.project_id, "facts": facts}
        except Exception:
            raise MemorySafetyError("Memory recall failed. No result was invented; raw errors are hidden.") from None
