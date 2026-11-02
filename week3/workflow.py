"""The Week 3 flow: recall preferences -> retrieve documents -> answer.

Read build_workflow() first. Each node returns changes to WorkflowState, like
updating a Java DTO. Response checks live in workflow_services.py; checkpoint
and start/resume helpers follow the graph below.
"""

from contextlib import asynccontextmanager
import os
from pathlib import Path
import re
from typing import TypedDict

# This step has no tracing integration. Disable inherited automatic tracing
# before importing LangGraph; credentials and SDK clients never enter state.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

from week3.retrieval import DOCUMENTATION_VERSION, validate_request
from week3.workflow_services import SECRET_PATTERN, WorkflowError, read_passages, read_preferences


class WorkflowState(TypedDict):
    # Like a Java DTO shared between steps. TypedDict describes the dictionary's fields.
    question: str
    documentation_version: str
    scope_key: str  # Binds saved progress to the intended user/project/destination.
    service_mode: str  # Distinguishes synthetic responses from live responses.
    preferences: list[str]  # Approved wording preferences, such as Java comparisons.
    memory_status: str
    passages: list[dict]  # Retrieved documentation: the factual evidence.
    answer: str
    citations: list[dict]  # Maps answer numbers such as [1] to document passages.
    usage: dict  # Token counts when the providers return them; otherwise None.
    warnings: list[str]
    phase: str  # Last completed step, useful when inspecting saved progress.


def build_workflow(tool_call, answerer, scope, checkpointer, *, pause_after_docs=False):
    """Define three steps, then connect them at the bottom of this function.

    tool_call sends an MCP request; answerer calls the answer model.
    scope identifies the user/project; checkpointer saves graph progress.
    These arguments are dependencies, like constructor arguments in Java.
    """

    # The nested functions share the supplied dependencies. LangGraph runs them later.
    # Step 1: recall an approved presentation preference through MCP.
    async def recall_preferences(state):
        try:
            # await waits for the MCP reply before this step continues.
            response = await tool_call("recall_project_facts", {"topic": "preferences"})
            preferences = read_preferences(response, scope)
            # Return only changed fields; LangGraph merges them into the shared state.
            return {"preferences": preferences, "memory_status": "ok" if preferences else "empty",
                    "phase": "preferences_recalled"}
        except Exception:
            # Memory is optional: keep answering from documentation if recall fails.
            return {"preferences": [], "memory_status": "unavailable", "phase": "preferences_recalled",
                    "warnings": state["warnings"] + ["MEMORY_UNAVAILABLE_CONTINUING_WITH_DOCUMENTATION"]}

    # Step 2: retrieve documentation through MCP, even when memory is empty.
    async def retrieve_documentation(state):
        try:
            response = await tool_call("search_caching_docs", {
                "question": state["question"], "documentation_version": state["documentation_version"],
            })
            passages = read_passages(response, state)
            # ** copies existing usage before updating its embedding entry.
            return {"passages": passages, "phase": "documentation_retrieved",
                    "usage": {**state["usage"], "embedding": response.get("embedding_usage")}}
        except Exception:
            # Documentation is required: stop here so resume can retry this step.
            raise WorkflowError("DOCUMENTATION_TOOL_FAILED") from None

    # Step 3: ask the answer model to use the retrieved documentation.
    async def generate_answer(state):
        try:
            result = await answerer(state)
            return {"answer": result["answer"], "citations": result["citations"], "phase": "answered",
                    "usage": {**state["usage"], "generation": result["generation_usage"]}}
        except WorkflowError:
            raise
        except Exception:
            raise WorkflowError("ANSWER_SERVICE_FAILED") from None

    def not_found(state):
        return {"answer": "I could not find this in the provided documentation.",
                "citations": [], "phase": "not_found"}

    def next_step(state):
        # Empty retrieval gets a fixed not-found answer without calling the model.
        if state["passages"]:
            return "generate_answer"
        return "not_found"

    # These edges are the complete decision flow. The model does not choose tools.
    builder = StateGraph(WorkflowState)
    # add_node registers a step's name and the Python function to execute.
    builder.add_node("recall_preferences", recall_preferences)
    builder.add_node("retrieve_documentation", retrieve_documentation)
    builder.add_node("generate_answer", generate_answer)
    builder.add_node("not_found", not_found)
    # add_edge means "after this step finishes, run the next one".
    builder.add_edge(START, "recall_preferences")
    builder.add_edge("recall_preferences", "retrieve_documentation")
    builder.add_conditional_edges("retrieve_documentation", next_step,
                                  {"generate_answer": "generate_answer", "not_found": "not_found"})
    builder.add_edge("generate_answer", END)
    builder.add_edge("not_found", END)
    # compile makes the graph runnable and attaches progress storage.
    # pause_after_docs demonstrates restart/resume before generation.
    return builder.compile(checkpointer=checkpointer,
                           interrupt_after=["retrieve_documentation"] if pause_after_docs else None)


def initial_state(question, scope, documentation_version=DOCUMENTATION_VERSION):
    validate_request(question, documentation_version)
    # These are documentation questions, not configuration/code submissions.
    # This guard catches common sensitive-input patterns, not every possible secret.
    if (any(mark in question for mark in ("\n", "\r", "```", "{", "}", "-----BEGIN"))
            or SECRET_PATTERN.search(question)
            or re.search(r"spring\.[\w.-]+\s*[:=]", question, re.I)):
        raise WorkflowError("USE_A_SYNTHETIC_DOCUMENTATION_QUESTION_WITHOUT_SECRETS_OR_CONFIG")
    return {
        "question": question.strip(),
        "documentation_version": documentation_version,
        "scope_key": scope.key,
        "service_mode": "mock_services" if scope.mode == "mock_services" else "live_services",
        "preferences": [],
        "memory_status": "not_checked",
        "passages": [],
        "answer": "",
        "citations": [],
        "usage": {"embedding": None, "generation": None},
        "warnings": [],
        "phase": "started",
    }


@asynccontextmanager
async def checkpointed_workflow(state_dir, tool_call, answerer, scope, *, pause_after_docs=False):
    # This SQLite file stores graph progress; hosted Mem0 stores long-term facts.
    path = Path(state_dir) / "workflow-checkpoints.sqlite3"
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    os.close(fd)
    os.chmod(path, 0o600)
    async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
        # Like Java try-with-resources: the database closes when the caller exits.
        yield build_workflow(tool_call, answerer, scope, saver, pause_after_docs=pause_after_docs)


async def prepare_run(graph, scope, thread_id, *, question=None, documentation_version=DOCUMENTATION_VERSION):
    # A thread ID identifies one workflow run. Reuse it to resume that run.
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", thread_id):
        raise WorkflowError("USE_A_NON_SECRET_THREAD_ID_OF_1_TO_64_LETTERS_DIGITS_OR_DASHES")
    if SECRET_PATTERN.search(thread_id):
        raise WorkflowError("USE_A_NON_SECRET_THREAD_ID_OF_1_TO_64_LETTERS_DIGITS_OR_DASHES")
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = await graph.aget_state(config)
    # snapshot.values is saved state; snapshot.next names the unfinished steps.
    if question is not None:
        if snapshot.values:
            raise WorkflowError("THREAD_ALREADY_EXISTS_USE_RESUME_OR_A_NEW_THREAD")
        inputs = initial_state(question, scope, documentation_version)
        next_nodes = ["recall_preferences"]
    else:
        if not snapshot.values:
            raise WorkflowError("WORKFLOW_THREAD_NOT_FOUND")
        if snapshot.values.get("scope_key") != scope.key:
            raise WorkflowError("WORKFLOW_SCOPE_OR_SERVICE_MODE_MISMATCH")
        if snapshot.values.get("documentation_version") != documentation_version:
            raise WorkflowError("WORKFLOW_DOCUMENTATION_VERSION_MISMATCH")
        # None tells LangGraph to continue its saved state instead of starting over.
        inputs, next_nodes = None, list(snapshot.next)
    return {"config": config, "inputs": inputs, "next_nodes": next_nodes,
            "state": snapshot.values}


async def execute_run(graph, plan):
    error = None
    # A completed plan has no pending nodes, so inspecting it makes no service call.
    if plan["next_nodes"]:
        try:
            # Persist each finished super-step before advancing. No retry policy.
            await graph.ainvoke(plan["inputs"], plan["config"], durability="sync")
        except WorkflowError as exception:
            error = str(exception)
    snapshot = await graph.aget_state(plan["config"])
    if error:
        status = "blocked"
    elif snapshot.next:
        status = "paused"
    else:
        status = "completed"
    result = {"status": status,
              "next_nodes": list(snapshot.next), "state": snapshot.values}
    if error:
        result["error_code"] = error
    return result
