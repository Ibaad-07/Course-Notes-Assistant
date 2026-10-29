"""
Compare the two chunking strategies on the same test questions.

    python compare_chunking.py

Each strategy has its own Qdrant collection (course_notes_fixed and
course_notes_recursive), so indexing one never overwrites the other.
"""

import config
from evaluate import metrics, print_report, run_eval
from index import get_collection, index_notes

STRATEGIES = ["fixed", "recursive"]


def main():
    results = {}
    for strategy in STRATEGIES:
        print(f"\n##### Indexing with strategy '{strategy}' #####")
        index_notes(strategy=strategy)
        results[strategy] = run_eval(get_collection(strategy))
        print_report(results[strategy], title=f"Strategy: {strategy}")

    # Side-by-side metrics
    a, b = STRATEGIES
    ma, mb = metrics(results[a]), metrics(results[b])
    print(f"\n\n===== Side by side (chunk size {config.CHUNK_SIZE}, overlap {config.CHUNK_OVERLAP}) =====")
    print(f"{'metric':<8} {a:>10} {b:>10}")
    for name in ma:
        better = a if ma[name] > mb[name] else b if mb[name] > ma[name] else "tie"
        print(f"{name:<8} {ma[name]:>10.2f} {mb[name]:>10.2f}   ({better})")

    # Per-question differences: where the first correct page came at a different
    # rank. A miss (None) is treated as rank 99 so it compares as "worse".
    print(f"\nQuestions where the strategies differ (rank of first correct source, top-5):")
    differ = False
    for ra, rb in zip(results[a], results[b]):
        if not ra["in_scope"] or ra["rank"] == rb["rank"]:
            continue
        differ = True
        winner = a if (ra["rank"] or 99) < (rb["rank"] or 99) else b
        fmt = lambda r: f"@{r}" if r else "MISS"
        print(f"  {ra['question'][:55]:<56} {a}: {fmt(ra['rank']):<5} {b}: {fmt(rb['rank']):<5} -> {winner} better")
    if not differ:
        print("  (none: both strategies found the right page at the same rank for every question)")

    # The out-of-scope question: lower top score = easier to tell "not in notes".
    for ra, rb in zip(results[a], results[b]):
        if not ra["in_scope"]:
            print(f"\nOut-of-scope '{ra['question']}': top score "
                  f"{a} {ra['retrieved'][0]['score']:.3f}, {b} {rb['retrieved'][0]['score']:.3f}")


if __name__ == "__main__":
    main()
