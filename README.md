# Course Notes Assistant (RAG Level 2)

Ask questions across all your lecture slides and notes. Answers cite the file
and page they came from, and you can limit a question to one course.

Everything runs locally on a CPU: Llama 3.2 3B through Ollama for answers,
`BAAI/bge-small-en-v1.5` for embeddings, Qdrant (local mode, saved to `./qdrant_db`,
no server needed) for the vector index.
No LangChain or LlamaIndex: loading, chunking and the pipeline are plain Python.

> **One program at a time:** Qdrant's local mode locks the `qdrant_db` folder.
> Close the Streamlit app before running `index.py`, `ask.py` or `evaluate.py`
> (and vice versa), or you'll get an "already in use" message.

## Setup

1. **Install Ollama** from https://ollama.com and start it (the Ollama app, or `ollama serve`).
2. **Pull the model** (about 2 GB):
   ```
   ollama pull llama3.2:3b
   ```
   If answers are too slow, use the 1B model instead: `ollama pull llama3.2:1b`
   and set `LLM_MODEL = "llama3.2:1b"` in `config.py`.
3. **Install the Python packages**:
   ```
   pip install -r requirements.txt
   ```
   The embedding model (~130 MB) downloads automatically the first time you index.

## Adding notes

Put files in one folder per course. **The folder name is the course label.**

```
notes/
  deep_learning/
    deep_learning_notes.pdf
  nlp/
    nlp_notes.pdf
    week3_slides.pptx
```

Supported: `.pdf`, `.pptx`, `.txt`, `.md`. Other files are skipped with a warning.
Scanned PDFs (images only) have no text to extract and are skipped; run OCR on them first.

## Indexing

```
python index.py
```

Re-run it whenever you add, change or delete notes. Unchanged files are skipped
(each file's hash is stored with its chunks), changed files are re-indexed and
deleted files are removed from the index. Changing `CHUNK_SIZE`, `CHUNK_OVERLAP`
or the strategy also triggers a re-index.

## Asking questions

**Terminal:**
```
python ask.py
python ask.py --course nlp
```
The retrieved chunks are printed with their scores and sources before the answer.
Type `clear` to forget the conversation, `quit` to exit.

**Web app:**
```
streamlit run app.py
```
Pick a course in the sidebar (or "All courses"). Open **Sources** under an answer
to see the chunks it was based on. Follow-up questions ("What value does it use?")
are rewritten into standalone questions first; the rewritten question is shown
above the answer. Turn this off with `REWRITE_FOLLOWUPS = False` in `config.py`.

### Speed on a CPU laptop

- The **first** question after starting (or after 30 idle minutes) takes 2-4 minutes:
  Ollama is loading the model into RAM. After that, answers take about 15-25 seconds.
  `LLM_KEEP_ALIVE` in `config.py` controls how long the model stays loaded.
- A follow-up question makes **two** LLM calls (rewrite, then answer), so it is
  slower. Set `REWRITE_FOLLOWUPS = False` if you don't need follow-ups.
- Still too slow? Switch to `llama3.2:1b` (see Setup).

### Known limitation

For questions that are genuinely ambiguous across courses (e.g. "What does
Assignment 2 require?" with **All courses** selected), the 3B model sometimes
mixes up which file a fact came from. Pick the course in the dropdown /
`--course` for these questions.

## Evaluation

`eval_set.json` holds 12 test questions with their expected source pages
(built from `answer_key.md`).

```
python evaluate.py                  # hit rate @1/3/5, MRR and a per-question table
python evaluate.py --filter-demo    # also: "What does Assignment 2 require?" per course
python compare_chunking.py          # fixed vs recursive chunking, side by side
```

These only test retrieval (no LLM calls), so they run in seconds.

## Files

| File | What it does |
|---|---|
| `config.py` | All settings: paths, models, chunking, top-k |
| `loaders.py` | Reads PDF / PPTX / TXT / MD into pages; strips repeated PDF footers |
| `chunking.py` | `fixed` and `recursive` chunking |
| `index.py` | notes/ -> chunks -> embeddings -> Qdrant, incrementally |
| `rag.py` | retrieve, build_prompt, generate, follow-up rewriting |
| `ask.py` | Command-line chat |
| `app.py` | Streamlit chat |
| `evaluate.py` | Retrieval metrics on `eval_set.json` |
| `compare_chunking.py` | Runs the evaluation for both chunking strategies |
