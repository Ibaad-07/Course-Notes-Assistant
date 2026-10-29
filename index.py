"""
Build (or update) the search index:  notes/ -> load -> chunk -> embed -> Qdrant.

Run:  python index.py              (uses CHUNK_STRATEGY from config.py)
      python index.py --strategy fixed

Indexing is incremental: each chunk stores a hash of the file it came from.
On the next run, files whose hash hasn't changed are skipped, so only new or
edited files are re-embedded (embedding is the slow part on a CPU).
"""

import argparse
import atexit
import hashlib
import sys
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (Distance, FieldCondition, Filter, FilterSelector,
                                  MatchValue, PointStruct, VectorParams)

import config
from chunking import chunk_pages
from loaders import load_file

_embedder = None  # loaded once, on first use (it takes a few seconds)
_client = None    # one Qdrant client per process (local mode locks the folder)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
def get_embedder():
    global _embedder
    if _embedder is None:
        # Imported here so scripts that never embed don't pay the slow torch import.
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedding model {config.EMBEDDING_MODEL} ...")
        _embedder = SentenceTransformer(config.EMBEDDING_MODEL, device="cpu")
    return _embedder


def embed(texts, show_progress=True):
    """Embed a list of texts in batches. Vectors are normalised to length 1,
    which is what cosine similarity expects."""
    return get_embedder().encode(
        texts,
        batch_size=config.EMBED_BATCH_SIZE,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
    ).tolist()


# ---------------------------------------------------------------------------
# Qdrant helpers
# ---------------------------------------------------------------------------
def get_client():
    """Open the on-disk Qdrant database (local mode: no server needed)."""
    global _client
    if _client is None:
        try:
            _client = QdrantClient(path=str(config.QDRANT_DIR))
        except RuntimeError as e:
            # Local mode lets only ONE program use the folder at a time.
            raise RuntimeError(f"{config.QDRANT_DIR.name} is already in use by another program "
                               "(is the Streamlit app or ask.py still running?). Close it and retry.") from e
        # Close it ourselves on exit; otherwise Qdrant tries during interpreter
        # shutdown and prints a harmless but confusing traceback.
        atexit.register(_client.close)
    return _client


def get_collection(strategy=None):
    """Return the collection name for a strategy, creating the collection if needed."""
    name = config.collection_name(strategy)
    client = get_client()
    if not client.collection_exists(name):
        client.create_collection(
            name,
            # Cosine: compares the direction of vectors, the usual choice for
            # sentence embeddings. Qdrant then returns cosine SIMILARITY as the score.
            vectors_config=VectorParams(size=config.EMBEDDING_DIM, distance=Distance.COSINE),
        )
    return name


def count_chunks(collection):
    return get_client().count(collection).count


def all_payloads(collection):
    """Metadata of every chunk in the collection. Qdrant returns results in
    pages, so keep scrolling until there is no next page."""
    payloads, offset = [], None
    while True:
        points, offset = get_client().scroll(collection, limit=256, offset=offset,
                                             with_payload=True, with_vectors=False)
        payloads += [p.payload for p in points]
        if offset is None:
            return payloads


def match(key, value):
    """A Qdrant filter meaning: payload[key] == value."""
    return Filter(must=[FieldCondition(key=key, match=MatchValue(value=value))])


def point_id(chunk_id):
    """Qdrant IDs must be integers or UUIDs. uuid5 turns our readable ID into a
    UUID that is always the same for the same chunk ID (so re-adding replaces)."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


# ---------------------------------------------------------------------------
# Finding and hashing files
# ---------------------------------------------------------------------------
def find_note_files(notes_dir):
    """Return {source: (path, course)} for every file under notes/<course>/.
    `source` is "<course>/<path inside the course folder>", unique per file."""
    files = {}
    for path in sorted(notes_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(notes_dir)
        if len(rel.parts) < 2:
            print(f"  WARNING: {rel} is not inside a course folder (notes/<course>/). Skipping.")
            continue
        if path.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
            print(f"  WARNING: unsupported file type, skipping: {rel}")
            continue
        course = rel.parts[0]  # the folder name is the course label
        files[rel.as_posix()] = (path, course)
    return files


def file_hash(path, strategy):
    """Hash of the file's bytes PLUS the chunking settings, so changing the
    chunk size or overlap in config also counts as "changed" and re-indexes."""
    h = hashlib.sha256(path.read_bytes())
    h.update(f"{strategy}|{config.CHUNK_SIZE}|{config.CHUNK_OVERLAP}".encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------
def index_notes(strategy=None, notes_dir=None, verbose=True):
    strategy = strategy or config.CHUNK_STRATEGY
    notes_dir = notes_dir or config.NOTES_DIR
    log = print if verbose else (lambda *a, **k: None)

    if not notes_dir.exists():
        sys.exit(f"Notes folder not found: {notes_dir}\nCreate it and add notes/<course>/<files>.")

    client = get_client()
    collection = get_collection(strategy)
    log(f"Indexing {notes_dir} -> collection '{collection}' (strategy: {strategy})")

    on_disk = find_note_files(notes_dir)
    if not on_disk:
        print(f"No supported notes found in {notes_dir}. Add files like "
              f"notes/<course>/lecture1.pdf ({', '.join(sorted(config.SUPPORTED_EXTENSIONS))}).")
    in_db = {p["source"]: p["file_hash"] for p in all_payloads(collection)}
    summary = {"indexed": [], "skipped": [], "removed": []}

    # 1. Files deleted from notes/: remove their chunks so they can't be cited.
    for source in sorted(set(in_db) - set(on_disk)):
        client.delete(collection, points_selector=FilterSelector(filter=match("source", source)))
        summary["removed"].append(source)
        log(f"  removed  {source}")

    # 2. New or changed files: (re)load, chunk, embed, add.
    for source, (path, course) in on_disk.items():
        new_hash = file_hash(path, strategy)
        if in_db.get(source) == new_hash:
            summary["skipped"].append(source)
            log(f"  skipped  {source} (unchanged)")
            continue

        if source in in_db:
            # Delete old chunks first: the new version may have fewer chunks,
            # and upsert alone would leave the extra old ones behind.
            client.delete(collection, points_selector=FilterSelector(filter=match("source", source)))

        chunks = chunk_pages(load_file(path, course), strategy=strategy)
        if not chunks:
            continue  # loader already printed why (no text, unreadable, ...)

        log(f"  indexing {source}: {len(chunks)} chunks")
        vectors = embed([c["text"] for c in chunks], show_progress=verbose)
        points = []
        for c, vector in zip(chunks, vectors):
            chunk_id = f"{c['course']}/{c['file']}_p{c['page']}_c{c['chunk']}"
            points.append(PointStruct(
                id=point_id(chunk_id),
                vector=vector,
                # The payload is Qdrant's word for metadata. The text lives here too.
                payload={
                    "chunk_id": chunk_id, "text": c["text"],
                    "course": c["course"], "file": c["file"], "page": c["page"],
                    "chunk": c["chunk"], "strategy": strategy,
                    "source": source, "file_hash": new_hash,
                },
            ))
        client.upsert(collection, points=points)
        summary["indexed"].append(source)

    summary["total_chunks"] = count_chunks(collection)
    log(f"\nSummary: {len(summary['indexed'])} indexed, {len(summary['skipped'])} skipped "
        f"(unchanged), {len(summary['removed'])} removed, {summary['total_chunks']} chunks in total.")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index your course notes into Qdrant.")
    parser.add_argument("--strategy", choices=["fixed", "recursive"], help="override CHUNK_STRATEGY")
    try:
        index_notes(strategy=parser.parse_args().strategy)
    except RuntimeError as e:
        sys.exit(str(e))
