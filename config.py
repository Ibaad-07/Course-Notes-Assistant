"""
All settings for the course notes assistant live here.
Every other file imports from this one, so you only ever change values in one place.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Resolve paths relative to this file, so scripts work no matter which
# folder you run them from.
PROJECT_DIR = Path(__file__).resolve().parent

NOTES_DIR = PROJECT_DIR / "notes"        # one subfolder per course, e.g. notes/nlp/
QDRANT_DIR = PROJECT_DIR / "qdrant_db"   # where Qdrant (local mode) saves the index on disk
EVAL_SET_PATH = PROJECT_DIR / "eval_set.json"

# Base name for the Qdrant collection. The chunking strategy is appended
# (e.g. "course_notes_recursive"), so both strategies can live side by side
# without overwriting each other.
COLLECTION_BASE_NAME = "course_notes"

# File types we know how to read. Anything else is skipped with a warning.
SUPPORTED_EXTENSIONS = {".pdf", ".pptx", ".txt", ".md"}

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
# Switch to "llama3.2:1b" if answers are too slow on your CPU.
LLM_MODEL = "llama3.2:3b"

# Small (~130 MB) embedding model that runs fine on a CPU.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384  # vector length bge-small produces; Qdrant needs it to create a collection

# How many chunks to embed at once. Larger is faster but uses more RAM.
EMBED_BATCH_SIZE = 32

# bge models were trained with this prefix on the QUESTION side (not on the
# documents). Adding it slightly improves retrieval for short questions.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
# "fixed"     -> fixed-size character windows with overlap (Level 1 style)
# "recursive" -> split on paragraphs, then sentences, then words, then merge
CHUNK_STRATEGY = "recursive"

CHUNK_SIZE = 500      # max characters per chunk
CHUNK_OVERLAP = 100   # characters repeated between neighbouring chunks

# Footer/header stripping for PDFs: a line counts as "repeating" if it appears
# on at least this fraction of a file's pages (e.g. a course title footer).
REPEATED_LINE_THRESHOLD = 0.6

# ---------------------------------------------------------------------------
# Retrieval and generation
# ---------------------------------------------------------------------------
TOP_K = 4   # how many chunks to retrieve and give to the LLM

# Low temperature keeps the model close to the context instead of being creative.
LLM_TEMPERATURE = 0.1

# How long Ollama keeps the model loaded after the last question. Loading takes
# a few minutes on a CPU, so keep it warm; it frees ~3 GB of RAM when it expires.
LLM_KEEP_ALIVE = "30m"

# Follow-up questions: rewrite "What value does it use?" into a standalone
# question using the chat history before retrieving. Set False to turn off.
REWRITE_FOLLOWUPS = True

# How many past chat turns (question + answer pairs) the rewriter sees.
HISTORY_TURNS = 3


def collection_name(strategy=None):
    """Name of the Qdrant collection for a chunking strategy (default: the configured one)."""
    return f"{COLLECTION_BASE_NAME}_{strategy or CHUNK_STRATEGY}"
