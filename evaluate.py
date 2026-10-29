"""
Retrieval-only evaluation: did the right page come back for each test question?

    python evaluate.py                     (uses CHUNK_STRATEGY from config.py)
    python evaluate.py --strategy fixed
    python evaluate.py --filter-demo       (also show the Assignment 2 course-filter demo)

No LLM calls, so it runs in seconds on a CPU. If retrieval misses the right
page, the LLM can't answer correctly no matter how good it is, so this is the
first thing to measure.
"""

import argparse
import json
import sys

import config
import rag
from index import count_chunks, get_collection

KS = [1, 3, 5]  # report hit rate at these cut-offs


def load_eval_set():
    with open(config.EVAL_SET_PATH, encoding="utf-8") as f:
        return json.load(f)


def first_hit_rank(retrieved, expected):
    """1-based rank of the first retrieved chunk whose (file, page) is one of the
    expected sources, or None if none of them appear."""
    wanted = {(s["file"], s["page"]) for s in expected}
    for rank, c in enumerate(retrieved, 1):
        if (c["file"], c["page"]) in wanted:
            return rank
    return None


def run_eval(collection=None):
    """Retrieve max(KS) chunks per question. Returns one result dict per question."""
    collection = collection or get_collection()
    results = []
    for item in load_eval_set():
        retrieved = rag.retrieve(item["question"], k=max(KS), collection=collection)
        results.append({
            **item,
            "retrieved": retrieved,
            "rank": first_hit_rank(retrieved, item["expected_sources"]),
            "in_scope": bool(item["expected_sources"]),
        })
    return results


def metrics(results):
    """Hit rate at each k and MRR, over the in-scope questions only (the
    out-of-scope question has no right answer to find)."""
    scored = [r for r in results if r["in_scope"]]
    out = {f"hit@{k}": sum(1 for r in scored if r["rank"] and r["rank"] <= k) / len(scored)
           for k in KS}
    # Reciprocal rank: 1 if the right page is first, 1/2 if second, ... 0 if missing.
    out["MRR"] = sum(1 / r["rank"] for r in scored if r["rank"]) / len(scored)
    return out


def _src(c):
    return f"{c['file'].replace('_notes.pdf', '')} p{c['page']} ({c['score']:.2f})"


def print_report(results, title=""):
    if title:
        print(f"\n=== {title} ===")
    print(f"\n{'#':>2}  {'Question':<44} {'Expected':<22} {'Top-3 retrieved (score)':<66} Result")
    print("-" * 150)
    for i, r in enumerate(results, 1):
        q = r["question"][:43]
        expected = ", ".join(f"{s['file'].replace('_notes.pdf', '')} p{s['page']}"
                             for s in r["expected_sources"]) or "(none)"
        top3 = " | ".join(_src(c) for c in r["retrieved"][:3])
        if r["in_scope"]:
            result = f"HIT @{r['rank']}" if r["rank"] else "MISS"
        else:
            # No right answer exists: show how high the best score still is.
            result = f"top score {r['retrieved'][0]['score']:.3f}"
        print(f"{i:>2}  {q:<44} {expected:<22} {top3:<66} {result}")

    m = metrics(results)
    in_scope = sum(1 for r in results if r["in_scope"])
    print(f"\nOver {in_scope} in-scope questions: "
          + "  ".join(f"{name} = {value:.2f}" for name, value in m.items()))
    return m


def filter_demo(collection=None):
    """The same ambiguous question with no filter and with each course filter."""
    question = "What does Assignment 2 require?"
    print(f"\n=== Course filter demo: {question!r} ===")
    for course in [None, *rag.list_courses(collection)]:
        chunks = rag.retrieve(question, course=course, collection=collection)
        print(f"\ncourse={course or 'None (all courses)'}")
        for c in chunks:
            text = " ".join(c["text"].split())
            hint = "  <- mentions Assignment 2" if "assignment 2" in text.lower() else ""
            print(f"  {c['score']:.3f}  [{c['course']}] {c['file']} p{c['page']}{hint}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    parser.add_argument("--strategy", choices=["fixed", "recursive"], help="override CHUNK_STRATEGY")
    parser.add_argument("--filter-demo", action="store_true", help="also run the course-filter demo")
    args = parser.parse_args()

    try:
        collection = get_collection(args.strategy)
    except RuntimeError as e:
        sys.exit(str(e))
    if count_chunks(collection) == 0:
        sys.exit(f"Collection '{collection}' is empty. "
                 f"Run `python index.py{' --strategy ' + args.strategy if args.strategy else ''}` first.")
    print_report(run_eval(collection), title=f"Retrieval evaluation ({collection}, "
                                             f"{count_chunks(collection)} chunks)")
    if args.filter_demo:
        filter_demo(collection)
