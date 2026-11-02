from time import perf_counter

import streamlit as st
from openai import OpenAI
from pinecone import Pinecone

from app import build_app


st.set_page_config(page_title="Spring Caching Assistant", page_icon="📚")
st.title("Spring Caching Assistant")
st.write("Ask a question about the Spring Boot 4.0 caching documentation.")

# Read credentials on the server so they never appear in page input fields.
try:
    pinecone_key = st.secrets.get("PINECONE_API_KEY", "").strip()
    nebius_key = st.secrets.get("NEBIUS_API_KEY", "").strip()
except FileNotFoundError:
    pinecone_key = nebius_key = ""

if not pinecone_key or not nebius_key:
    st.info(
        "Before your demo, add PINECONE_API_KEY and NEBIUS_API_KEY to "
        ".streamlit/secrets.toml, then restart the app. "
        "The keys will load automatically without being displayed."
    )
    st.stop()

with st.form("question_form"):
    question = st.text_area(
        "Your question",
        placeholder="How do I enable caching and configure Redis cache expiration?",
        max_chars=2000,
    )
    submitted = st.form_submit_button("Ask", type="primary")

st.caption("Each question is independent. Answers include numbered source citations.")

if submitted:
    # Clear the previous answer so failed requests never display stale results.
    st.session_state.pop("last_result", None)
    question = question.strip()
    if not question or len(question) > 2000:
        st.warning("Enter a question of 1–2000 characters.")
    else:
        try:
            with st.spinner("Searching the documentation and preparing an answer..."):
                start = perf_counter()
                pc = Pinecone(api_key=pinecone_key)
                index = pc.Index(host=pc.describe_index("spring-rag").host)
                client = OpenAI(
                    base_url="https://api.tokenfactory.nebius.com/v1/",
                    api_key=nebius_key,
                    max_retries=0,
                    timeout=60.0,
                )
                result = build_app(client, index).invoke({"question": question})
                st.session_state["last_result"] = {
                    "question": question,
                    "answer": result["answer"],
                    "passages": result["passages"],
                    "seconds": perf_counter() - start,
                }
        except Exception:
            # Do not expose API responses or credentials in the page.
            st.error(
                "Could not get an answer. Check your local API-key setup and connection, "
                "and make sure the spring-rag index has been populated. Then try again."
            )

if "last_result" in st.session_state:
    result = st.session_state["last_result"]
    st.subheader("Answer")
    st.caption(f"Question: {result['question']}")
    st.markdown(result["answer"])
    st.caption(f"Answered in {result['seconds']:.2f} seconds")

    if result["passages"]:
        st.subheader("Sources")
        for number, passage in enumerate(result["passages"], start=1):
            with st.expander(f"[{number}] {passage['id']}"):
                st.caption(f"Similarity: {passage['score']:.4f}")
                st.write(passage["source"])
                st.text(passage["text"])
