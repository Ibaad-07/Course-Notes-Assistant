"""
The core question-answering logic, shared by ask.py (CLI), app.py (Streamlit)
and evaluate.py (retrieval only).

    question -> [rewrite follow-up] -> retrieve chunks -> build prompt -> LLM answer
"""

import ollama

import config
from index import all_payloads, count_chunks, embed, get_client, get_collection, match

NOT_FOUND = "I couldn't find that in your notes."


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def list_courses(collection=None):
    """All course labels currently in the index (for the --course flag / dropdown)."""
    collection = collection or get_collection()
    return sorted({p["course"] for p in all_payloads(collection)})


def retrieve(question, course=None, k=None, collection=None):
    """Return the k most similar chunks, optionally only from one course.

    `collection` is a collection name (default: the configured strategy's).
    Each result is {"text", "course", "file", "page", "id", "score"}, best first.
    """
    collection = collection or get_collection()
    if count_chunks(collection) == 0:
        raise RuntimeError("Nothing is indexed yet. Run `python index.py` first.")

    # bge expects this instruction in front of queries (but not documents).
    query_vec = embed([config.QUERY_INSTRUCTION + question], show_progress=False)[0]

    hits = get_client().query_points(
        collection,
        query=query_vec,
        limit=k or config.TOP_K,
        # The filter runs inside Qdrant, so we get the top k chunks OF THAT
        # COURSE, not the top k overall with other courses thrown away.
        query_filter=match("course", course) if course else None,
        with_payload=True,
    ).points

    return [{
        "id": h.payload["chunk_id"], "text": h.payload["text"],
        "course": h.payload["course"], "file": h.payload["file"], "page": h.payload["page"],
        # With Distance.COSINE, Qdrant's score is already cosine SIMILARITY
        # (1 = same direction, 0 = unrelated), so higher = more relevant and no
        # distance-to-similarity conversion is needed (Chroma would need 1 - distance).
        "score": h.score,
    } for h in hits]


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
# The rules go in a separate SYSTEM message: small models follow instructions
# much more reliably when they aren't buried in the same text as the notes.
ANSWER_RULES = f"""You are a study assistant that answers questions using ONLY the student's course notes provided in the user message.

Rules:
1. Use only facts stated in the notes. Never use outside knowledge or guess.
2. If the notes do not contain the answer, reply with exactly this sentence and nothing else: {NOT_FOUND}
3. After each fact, cite the source of the <note> it came from, copied exactly, for example (deep_learning_notes.pdf, p. 3).
4. If notes from different courses give different answers, give each one and say which course it belongs to.
5. Be concise."""


def build_prompt(question, chunks):
    """Label each chunk with its source so the model can cite it.

    The label is written in EXACTLY the citation format we want back, so the
    model only has to copy it (it got file names and pages wrong when the label
    looked different from the citation). It appears at the start AND end of
    each note: with a label only at the start, the model sometimes credited a
    note's last sentence to the NEXT note's label."""
    context = "\n\n".join(
        f"<note source=\"({c['file']}, p. {c['page']})\" course=\"{c['course']}\">\n"
        f"{c['text']}\n"
        f"</note source=\"({c['file']}, p. {c['page']})\">"
        for c in chunks
    )
    return f"Notes:\n\n{context}\n\nQuestion: {question}"


# ---------------------------------------------------------------------------
# LLM (Ollama)
# ---------------------------------------------------------------------------
def check_ollama():
    """Fail early with a clear message if Ollama isn't running or the model isn't pulled."""
    try:
        models = ollama.list().models
    except Exception:
        raise RuntimeError("Can't reach Ollama. Install it from https://ollama.com and make sure "
                           "it is running (open the Ollama app, or run `ollama serve`).")
    names = {m.model for m in models}
    if config.LLM_MODEL not in names:
        raise RuntimeError(f"Model '{config.LLM_MODEL}' is not pulled. "
                           f"Run: ollama pull {config.LLM_MODEL}")


def _chat(messages, stream=False):
    return ollama.chat(
        model=config.LLM_MODEL,
        messages=messages,
        options={"temperature": config.LLM_TEMPERATURE},
        # Keep the model in RAM between questions: loading it from disk takes
        # minutes on a CPU laptop, and Ollama's default unloads after 5 idle minutes.
        keep_alive=config.LLM_KEEP_ALIVE,
        stream=stream,
    )


def generate(prompt, stream=False, system=ANSWER_RULES):
    """Run the prompt through the LLM. With stream=True, yields text pieces as
    they are produced (so a slow CPU still shows progress)."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    if not stream:
        return _chat(messages)["message"]["content"].strip()
    return (part["message"]["content"] for part in _chat(messages, stream=True))


# ---------------------------------------------------------------------------
# Follow-up questions
# ---------------------------------------------------------------------------
def rewrite_question(question, history):
    """Turn a follow-up like "What value does it use?" into a standalone question,
    using the chat history. Retrieval only sees the question, so without this
    step "it" would match nothing useful.

    history: list of {"role": "user"|"assistant", "content": str}, oldest first.
    """
    if not config.REWRITE_FOLLOWUPS or not history:
        return question

    recent = history[-2 * config.HISTORY_TURNS:]
    convo = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in recent)

    def task(conversation, follow_up):
        return f"Conversation:\n{conversation}\n\nFollow-up question: {follow_up}"

    # The worked example is given as a real user/assistant exchange: small
    # models copy the PATTERN of a previous turn far better than they follow
    # an example written inside the instructions.
    messages = [
        {"role": "system", "content": REWRITE_RULES},
        {"role": "user", "content": task(
            "User: What is batch normalization?\n"
            "Assistant: It normalizes each layer's inputs over the mini-batch.",
            "Which week covers it?")},
        {"role": "assistant", "content": "Which week covers batch normalization?"},
        {"role": "user", "content": task(convo, question)},
    ]
    lines = _chat(messages)["message"]["content"].strip().splitlines() or [""]
    rewritten = lines[0].strip().strip('"').strip()

    # Small models sometimes echo the instructions or return nothing useful:
    # fall back to the original question rather than searching for garbage.
    if len(rewritten) < 4 or any(w in rewritten.lower() for w in ("standalone", "rewrite", "follow-up")):
        return question
    return rewritten


REWRITE_RULES = """You turn a follow-up question into a standalone question.
Replace words like "it", "that", "this" or "they" with what they refer to in the conversation.
Output ONLY the standalone question, on one line. Do not answer it. Do not explain."""


def answer(question, course=None, history=None, stream=False):
    """Full pipeline. Returns {"standalone", "chunks", "answer"}.
    With stream=True, "answer" is a generator of text pieces."""
    standalone = rewrite_question(question, history or [])
    chunks = retrieve(standalone, course=course)
    return {
        "standalone": standalone,
        "chunks": chunks,
        "answer": generate(build_prompt(standalone, chunks), stream=stream),
    }
