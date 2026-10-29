"""
Split pages into chunks small enough to embed and to fit in the LLM prompt.

Two strategies (pick one with CHUNK_STRATEGY in config.py):
  fixed     -> cut every CHUNK_SIZE characters, stepping back CHUNK_OVERLAP each time.
               Simple, but happily cuts sentences (and facts) in half.
  recursive -> split on the biggest natural boundary first (paragraphs), only
               going finer (lines, sentences, words) when a piece is still too
               big, then merge small pieces back up to CHUNK_SIZE.

Every chunk keeps its page's course/file/page, so answers can cite sources.
"""

import re

import config


# ---------------------------------------------------------------------------
# Strategy 1: fixed-size windows
# ---------------------------------------------------------------------------
def fixed_chunks(text, size, overlap):
    step = size - overlap  # each window starts `step` chars after the previous one
    chunks = []
    for start in range(0, len(text), step):
        chunk = text[start:start + size].strip()
        if chunk:
            chunks.append(chunk)
        if start + size >= len(text):  # this window already reached the end
            break
    return chunks


# ---------------------------------------------------------------------------
# Strategy 2: recursive splitting
# ---------------------------------------------------------------------------
# Separators from coarsest to finest. Each level is only used on pieces that
# are still bigger than the chunk size after the previous level.
SEPARATORS = [
    r"\n\s*\n",           # blank line = paragraph break
    r"\n",                # line break (PDF text has no blank lines, so this matters)
    r"(?<=[.!?])\s+",     # end of sentence
    r"\s+",               # between words
]


def _split(text, size, level=0):
    """Break text into pieces that are each <= size (if at all possible)."""
    if len(text) <= size:
        return [text]
    if level == len(SEPARATORS):
        # A single "word" longer than size (e.g. a URL): hard-cut it.
        return [text[i:i + size] for i in range(0, len(text), size)]

    pieces = []
    for part in re.split(SEPARATORS[level], text):
        if part.strip():
            # Only parts that are STILL too big go one level finer.
            pieces.extend(_split(part.strip(), size, level + 1))
    return pieces


def _tail(text, overlap):
    """The last ~overlap characters of text, starting at a sentence or line
    start if one falls inside that window, otherwise at a word boundary,
    so the overlap doesn't begin with half a sentence or half a word."""
    if len(text) <= overlap:
        return text
    tail = text[-overlap:]
    boundary = re.search(r"(?<=[.!?])\s+|\n", tail)
    if boundary and tail[boundary.end():].strip():
        return tail[boundary.end():]
    space = tail.find(" ")
    return tail[space + 1:] if space != -1 else tail


def recursive_chunks(text, size, overlap):
    pieces = _split(text, size)

    # Merge neighbouring small pieces (e.g. a heading and its paragraph) until
    # adding the next one would go over the size limit.
    chunks = []
    current = ""
    for piece in pieces:
        joined = f"{current}\n{piece}" if current else piece
        if len(joined) <= size:
            current = joined
            continue
        chunks.append(current)
        # Start the next chunk with the end of the previous one, so a fact that
        # sits on a boundary still appears whole in at least one chunk.
        carry = _tail(current, overlap)
        current = f"{carry}\n{piece}" if len(carry) + 1 + len(piece) <= size else piece
    if current:
        chunks.append(current)
    return chunks


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def chunk_pages(pages, strategy=None, size=None, overlap=None):
    """Turn a list of loader pages into a list of chunks.

    Each chunk is the page dict plus "chunk" (its index within the page) and
    "strategy". Chunks never cross page boundaries, so every chunk has exactly
    one page number to cite.
    """
    strategy = strategy or config.CHUNK_STRATEGY
    size = size or config.CHUNK_SIZE
    overlap = config.CHUNK_OVERLAP if overlap is None else overlap

    chunks = []
    for page in pages:
        text = page["text"]
        is_slide = page["file"].lower().endswith(".pptx")

        if is_slide and len(text) <= size:
            # A slide is already a natural, self-contained unit: keep it whole.
            pieces = [text]
        elif strategy == "fixed":
            pieces = fixed_chunks(text, size, overlap)
        elif strategy == "recursive":
            pieces = recursive_chunks(text, size, overlap)
        else:
            raise ValueError(f"Unknown chunk strategy {strategy!r} (use 'fixed' or 'recursive')")

        for n, piece in enumerate(pieces):
            chunks.append({**page, "text": piece, "chunk": n, "strategy": strategy})
    return chunks
