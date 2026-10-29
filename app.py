"""
Streamlit chat UI for the course notes assistant.

    streamlit run app.py
"""

import streamlit as st

import config
import rag

st.set_page_config(page_title="Course Notes Assistant", page_icon="📚")
st.title("📚 Course Notes Assistant")

# ---------------------------------------------------------------------------
# Startup checks: show a clear message instead of a stack trace.
# ---------------------------------------------------------------------------
try:
    courses = rag.list_courses()
except Exception as e:
    st.error(f"Could not open the index: {e}")
    st.stop()
if not courses:
    st.warning("Nothing is indexed yet. Add notes under `notes/<course>/` and run `python index.py`.")
    st.stop()
# Without the LLM we can still search: show the retrieved chunks instead of
# an answer, so the app stays useful while Ollama is missing or the model is pulling.
try:
    rag.check_ollama()
    llm_ok = True
except RuntimeError as e:
    st.warning(f"{e}\n\nRunning in **search-only mode**: you'll see the matching chunks, but no written answer.")
    llm_ok = False

# ---------------------------------------------------------------------------
# Sidebar: course filter and settings
# ---------------------------------------------------------------------------
ALL = "All courses"
with st.sidebar:
    choice = st.selectbox("Course", [ALL] + courses)
    course = None if choice == ALL else choice
    st.caption(f"Model: `{config.LLM_MODEL}`  \nChunks: `{config.CHUNK_STRATEGY}`, top-k {config.TOP_K}  \n"
               f"Follow-up rewriting: {'on' if config.REWRITE_FOLLOWUPS else 'off'}")
    if st.button("Clear chat"):
        st.session_state.messages = []

# Streamlit re-runs this whole script on every interaction, so the chat
# history has to live in session_state to survive between questions.
if "messages" not in st.session_state:
    st.session_state.messages = []


def show_sources(chunks):
    with st.expander(f"Sources ({len(chunks)} chunks)"):
        for c in chunks:
            st.markdown(f"**{c['file']}, p. {c['page']}** · {c['course']} · score `{c['score']:.3f}`")
            st.text(c["text"])


def show_rewrite(original, standalone):
    if standalone != original:
        st.caption(f"🔁 Searched for: *{standalone}*")


# Re-draw the conversation so far.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            show_rewrite(msg["question"], msg["standalone"])
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            show_sources(msg["chunks"])

# ---------------------------------------------------------------------------
# New question
# ---------------------------------------------------------------------------
if question := st.chat_input("Ask about your notes..."):
    with st.chat_message("user"):
        st.markdown(question)

    # The rewriter only needs role + text, not the stored sources.
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

    with st.chat_message("assistant"):
        with st.spinner("Understanding the question..." if history and llm_ok else "Searching your notes..."):
            # Rewriting follow-ups needs the LLM, so it's skipped in search-only mode.
            standalone = rag.rewrite_question(question, history) if llm_ok else question
            chunks = rag.retrieve(standalone, course=course)
        show_rewrite(question, standalone)
        if llm_ok:
            # Stream the answer: on a CPU it takes a while, so show words as they come.
            reply = st.write_stream(rag.generate(rag.build_prompt(standalone, chunks), stream=True))
        else:
            reply = "_Search-only mode: no LLM answer. The best-matching chunks are under Sources._"
            st.markdown(reply)
        show_sources(chunks)

    st.session_state.messages += [
        {"role": "user", "content": question},
        {"role": "assistant", "content": reply, "question": question,
         "standalone": standalone, "chunks": chunks},
    ]
