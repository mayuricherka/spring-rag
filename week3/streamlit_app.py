"""Live Week 3 Streamlit entry point. Offline fixtures are supplied only by tests."""

import json
import os
from pathlib import Path

import streamlit as st

from week3.memory_ledger import FACTS, MemorySafetyError, STATE_DIR
from week3.retrieval import RetrievalError
from week3.ui_backend import UIBackend, ui_scope
from week3.workflow_services import WorkflowError


def main():
    # Streamlit reruns this function after interactions, rebuilding the screen.
    st.set_page_config(page_title="Spring caching — Week 3", page_icon="🌱")
    st.title("Spring caching assistant")
    st.caption("Week 3 · MCP tools, approved memory and resumable answers")
    # An old ?mode=mock URL cannot select fixtures in the user-facing app.
    st.query_params["mode"] = "live"
    st.info("Live services: Nebius, Pinecone and hosted Mem0. Actions may consume credits; review current limits and pricing first.")
    permit_live = st.checkbox("I approve live service requests for the actions I click.", key="week3_live_permission")

    # Runtime configuration stays on the server. No key fields are shown in the UI.
    secrets_getter = lambda name: st.secrets.get(name)
    try:
        scope = ui_scope("live", secrets_getter)
        backend = UIBackend(scope, Path(os.environ.get("SPRING_RAG_UI_STATE_DIR", STATE_DIR)),
                            secrets_getter=secrets_getter)
        # The URL's view ID restores local state after refresh; it is not a password.
        view = backend.store.resolve(st.query_params.get("view"))
        st.query_params["view"] = view["id"]
    except MemorySafetyError as error:
        st.error(f"Live setup is unavailable. {error}")
        st.caption("Set MEM0_EXPECTED_ORG and MEM0_EXPECTED_PROJECT in .streamlit/secrets.toml or the server environment. "
                   "Live actions also require MEM0_API_KEY. See .streamlit/secrets.toml.example, then restart Streamlit after configuring it.")
        st.stop()
    except Exception:
        st.error("Live setup is unavailable. Check server configuration and access to the local state directory.")
        st.stop()

    def action(function):
        try:
            return True, function()
        except (MemorySafetyError, WorkflowError, RetrievalError) as error:
            st.error(str(error))  # Only application-owned, fixed safe messages.
        except Exception:
            st.error("The action could not finish. Your saved state is retained; raw errors are hidden.")
        return False, None

    # Start a fresh conversation while retaining the same project memory identity.
    if st.button("New session and recall", key="week3_new_session"):
        ok, fresh = action(lambda: backend.new_session(view["id"], permit_live=permit_live))
        if ok:
            st.query_params["view"] = fresh["id"]
            st.rerun()

    if view["recall_json"]:
        recalled = json.loads(view["recall_json"])
        st.subheader("Recalled project memory")
        if recalled["status"] == "unavailable":
            st.warning("Memory recall is unavailable. You can still ask a documentation question.")
        elif recalled["facts"]:
            for fact in recalled["facts"]:
                st.write(fact["fact"])
        else:
            st.caption("No approved project facts were recalled.")

    # A form submits on Ask; typing alone does not send a model/tool request.
    with st.form("question_form_" + view["id"]):
        question = st.text_area("Question", value=view["question"] or "", key="question_" + view["id"],
                                placeholder="How do I enable caching?",
                                help="Use public documentation questions. Do not include keys, private configurations or sensitive code.")
        asked = st.form_submit_button("Ask", key="ask_" + view["id"])
    if asked:
        with st.spinner("Preparing your answer…"):
            ok, _ = action(lambda: backend.ask(view["id"], question, permit_live=permit_live))
        if ok:
            st.rerun()

    # Read checkpoints for display. A rerun must not automatically execute the graph.
    ok, result = action(lambda: backend.result(view["id"]))
    if ok and result:
        state = result["state"]
        if result["status"] == "pending":
            st.warning("This answer has unfinished steps. Resume only when you want to retry the remaining requests.")
            if st.button("Resume answer", key="resume_" + view["id"]):
                with st.spinner("Resuming…"):
                    resumed, _ = action(lambda: backend.resume(view["id"], permit_live=permit_live))
                if resumed:
                    st.rerun()
        if state.get("answer"):
            st.subheader("Answer")
            st.caption("Question: " + state["question"])
            st.markdown(state["answer"])
            if state.get("preferences"):
                with st.expander("Presentation preferences used"):
                    for preference in state["preferences"]:
                        st.write(preference)
            if state.get("warnings"):
                st.warning("Project memory was unavailable. This answer used documentation without saved preferences.")
        if state.get("passages"):
            st.subheader("Source passages")
            for number, passage in enumerate(state["passages"], 1):
                with st.expander(f"[{number}] Spring Boot {passage['documentation_version']} · {passage['id']}"):
                    st.code(passage["text"], language=None)
                    st.markdown(f"[Open documentation]({passage['source']})")

    st.divider()
    st.subheader("Propose a project memory")
    # The catalog limits the preview to known safe examples; preview is local only.
    fact_key = st.selectbox("Fact to propose", list(FACTS), format_func=lambda key: FACTS[key],
                            key="fact_" + view["id"])
    if st.button("Preview memory", key="preview_" + view["id"]):
        ok, _ = action(lambda: backend.preview(view["id"], fact_key))
        if ok:
            st.rerun()
    ok, proposal = action(lambda: backend.proposal(view["id"]))
    if ok and proposal:
        st.write("Exact proposed fact:")
        st.code(proposal["fact"], language=None)
        st.caption(f"Destination: Hosted Mem0 · user {scope.user_id} · project {scope.project_id}")
        st.caption(f"Cloud destination: {scope.cloud_org} / {scope.cloud_project}")
        st.caption("Status: " + proposal["state"])
        # Buttons refer to this exact proposal; the backend checks its durable ledger.
        operation_id = proposal["operation_id"]
        left, right = st.columns(2)
        if left.button("Approve", key="approve_" + operation_id + view["id"], disabled=proposal["state"] != "proposed"):
            with st.spinner("Saving the approved fact…"):
                approved, _ = action(lambda: backend.approve(view["id"], operation_id, permit_live=permit_live))
            if approved:
                st.rerun()
        if right.button("Reject", key="reject_" + operation_id + view["id"], disabled=proposal["state"] != "proposed"):
            rejected, _ = action(lambda: backend.reject(view["id"], operation_id))
            if rejected:
                st.rerun()
        if proposal["state"] == "completed":
            st.success("Save completed. Start a new session to demonstrate recall.")
        elif proposal["state"] == "rejected":
            st.info("Proposal rejected.")
        elif proposal["state"] in {"approved", "in_flight", "pending_remote", "uncertain"}:
            st.warning("Save completion is pending. The same operation will be checked; an uncertain write will not be repeated.")
            if st.button("Check save status", key="check_" + operation_id + view["id"]):
                checked, _ = action(lambda: backend.approve(view["id"], operation_id, permit_live=permit_live))
                if checked:
                    st.rerun()


if __name__ == "__main__":
    main()
