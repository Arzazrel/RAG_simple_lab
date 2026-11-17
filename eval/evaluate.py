"""
A/B evaluation of the RAG system.

Goal
    Compare two conditions on the SAME questions with the SAME generator model, changing only ONE variable: the presence of retrieved context.
        A (baseline) : the model answers alone, no context.
        B (RAG)      : the model answers with the retrieved chunks in the prompt.


Why this is meaningful
    The definitions in the context files are INVENTED, so they cannot be in the model's weights. 
    In condition A the model can only fail or hallucinate; if it answers correctly in condition B, that knowledge can only come from the retrieved documents.
    The accuracy gap B - A is therefore a clean measure of the value of retrieval.

Metrics:
1. Answer accuracy  : fraction of expected key_terms found in the answer (>= THRESHOLD -> correct). Automatic, no human judgement.

2. Retrieval hit    : (condition B only) was the right source_file among the retrieved chunks? Separates retriever errors from generator errors.

3. Hallucination    : on control questions (expected_answer == NOT_IN_CORPUS), did the model admit it has no information, or invent one?

Usage
    python eval/evaluate.py --questions eval/questions.json --docs documenti \
                            --retriever semantic --k 3

Author: Alessandro Diana
Version: November 17, 2026
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

# make 'src' importable whatever the working directory is
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model import LocalLLM                                   
from retrieval import NameRetriever, SemanticRetriever, HybridRetriever

# ----------------------- costant -----------------------
THRESHOLD = 0.5     # a key-term match counts as "correct" if at least this fraction appears
# phrases that count as an honest "I don't know" on control questions
REFUSAL_MARKERS = [
    "don't have", "dont have", "do not have", "no information",
    "not in the context", "cannot find", "can't find", "cant find",
    "not provided", "not mentioned", "does not contain", "doesn't contain",
    "doesnt contain", "unable to", "no mention", "i don't know", "i dont know",
]


# ----------------------- Scoring helpers methods -----------------------

def key_term_score(answer: str, key_terms: list[str]) -> float:
    """Fraction of expected key terms that appear in the answer (case-insensitive)."""
    
    if not key_terms:
        return 0.0
    low = answer.lower()
    hits = sum(1 for t in key_terms if t.lower() in low)
    return hits / len(key_terms)


def looks_like_refusal(answer: str) -> bool:
    """True if the answer admits it has no information (good on control questions)."""
    low = answer.lower()
    return any(m in low for m in REFUSAL_MARKERS)


def build_prompt(query: str, retrieved: list[dict]) -> str:
    """Augmented prompt for condition B: context + question."""
    ctx = "\n\n".join(f"[{c['source']}] {c['text']}" for c in retrieved)
    return (f"Context:\n{ctx}\n\n"
            f"Question: {query}\n"
            f"Answer using only the context above.")


# ----------------------- Retriever factory methods -----------------------


def make_retriever(kind: str, docs: str):
    if kind == "name":
        return NameRetriever(docs)
    if kind == "semantic":
        return SemanticRetriever(docs, cache_dir="models")
    if kind == "hybrid":
        return HybridRetriever(docs, cache_dir="models")
    raise ValueError(f"unknown retriever: {kind}")


# ----------------------- Main evaluation loop -----------------------


def run(questions_path: str, docs: str, retriever_kind: str, k: int, verbose: bool):
    data = json.loads(Path(questions_path).read_text(encoding="utf-8"))
    items = data["items"]

    llm = LocalLLM()
    retriever = make_retriever(retriever_kind, docs)

    # counters, split into normal questions and control (NOT_IN_CORPUS) questions
    normal = [it for it in items if it["expected_answer"] != "NOT_IN_CORPUS"]
    control = [it for it in items if it["expected_answer"] == "NOT_IN_CORPUS"]

    stats = {
        "A_correct": 0, "B_correct": 0,         # answer accuracy
        "retrieval_hit": 0,                     # right source_file retrieved (B)
        "A_hallucinate": 0, "B_hallucinate": 0, # control questions
    }
    rows = []

    for it in normal:
        q = it["question"]
        retrieved = retriever.retrieve(q, k=k)

        # condition A: no context
        ans_a = llm.generate(q)
        # condition B: with retrieved context
        ans_b = llm.generate(build_prompt(q, retrieved))

        score_a = key_term_score(ans_a, it["key_terms"])
        score_b = key_term_score(ans_b, it["key_terms"])
        a_ok = score_a >= THRESHOLD
        b_ok = score_b >= THRESHOLD
        stats["A_correct"] += a_ok
        stats["B_correct"] += b_ok

        # retrieval diagnostic: is the right file among the retrieved sources?
        hit = any(it["source_file"] and it["source_file"] in c["source"] for c in retrieved)
        stats["retrieval_hit"] += hit

        rows.append({"id": it["id"], "term": it["term"], "A": round(score_a, 2),
                     "B": round(score_b, 2), "retrieval_hit": hit})

        if verbose:
            print(f"\n[{it['id']}] {it['term']}  (retrieval_hit={hit})")
            print(f"  A (no RAG) score={score_a:.2f} -> {ans_a[:120]}")
            print(f"  B (RAG)    score={score_b:.2f} -> {ans_b[:120]}")

    # control questions: the model should refuse, not invent
    for it in control:
        q = it["question"]
        retrieved = retriever.retrieve(q, k=k)
        ans_a = llm.generate(q)
        ans_b = llm.generate(build_prompt(q, retrieved))
        # hallucination = did NOT refuse
        if not looks_like_refusal(ans_a):
            stats["A_hallucinate"] += 1
        if not looks_like_refusal(ans_b):
            stats["B_hallucinate"] += 1
        if verbose:
            print(f"\n[{it['id']}] CONTROL {it['term']}")
            print(f"  A refusal={looks_like_refusal(ans_a)} -> {ans_a[:120]}")
            print(f"  B refusal={looks_like_refusal(ans_b)} -> {ans_b[:120]}")

    report(stats, len(normal), len(control), rows, retriever_kind, k)

# ----------------------- UI methods -----------------------


def report(stats, n_normal, n_control, rows, kind, k):
    print("\n" + "=" * 60)
    print(f"RAG A/B REPORT  (retriever={kind}, k={k})")
    print("=" * 60)
    if n_normal:
        print(f"Answer accuracy  A (no RAG): {stats['A_correct']}/{n_normal} "
              f"= {stats['A_correct']/n_normal:.0%}")
        print(f"Answer accuracy  B (RAG)   : {stats['B_correct']}/{n_normal} "
              f"= {stats['B_correct']/n_normal:.0%}")
        print(f"Retrieval hit rate (B)     : {stats['retrieval_hit']}/{n_normal} "
              f"= {stats['retrieval_hit']/n_normal:.0%}")
        gain = (stats['B_correct'] - stats['A_correct']) / n_normal
        print(f"Accuracy gain B - A        : {gain:+.0%}   <-- value of retrieval")
    if n_control:
        print(f"\nControl questions (should refuse):")
        print(f"  Hallucinations A (no RAG): {stats['A_hallucinate']}/{n_control}")
        print(f"  Hallucinations B (RAG)   : {stats['B_hallucinate']}/{n_control}")
    print("\nPer-question (key-term coverage):")
    print(f"  {'id':<5}{'term':<16}{'A':>6}{'B':>6}  hit")
    for r in rows:
        print(f"  {r['id']:<5}{r['term']:<16}{r['A']:>6}{r['B']:>6}  {r['retrieval_hit']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="A/B evaluation of the RAG system")
    ap.add_argument("--questions", default="eval/questions.json")
    ap.add_argument("--docs", default="docs")
    ap.add_argument("--retriever", default="semantic", choices=["name", "semantic", "hybrid"])
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--verbose", action="store_true", help="print every answer")
    args = ap.parse_args()
    run(args.questions, args.docs, args.retriever, args.k, args.verbose)
