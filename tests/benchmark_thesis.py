"""Thesis benchmarking suite (spec section 8).

Runs the REAL services in-process (Whisper, ChromaDB, Groq) to produce the quantitative
evidence for the thesis results chapter. Not collected by pytest (name does not match test_*).

Sub-commands
------------
wer      Word/Character Error Rate over a folder of ``<name>.wav|mp3|webm|...`` + ``<name>.txt``
         reference pairs.                                   Target: mean WER < 12 %
rag      Recall@K and NDCG@K over a JSON file of canonical queries:
         ``[{"query": "...", "relevant_sources": ["sharding.md", "system-design-primer"]}, ...]``
         A hit is any retrieved chunk whose ``source`` contains one of the substrings.
                                                            Target: Recall@3 >= 85 %
latency  End-to-end latency profile (ASR / vector search / LLM) over repeated runs of one
         audio file.                                        Target: total < 2.5 s

Examples
--------
    python tests/benchmark_thesis.py wer --audio-dir benchmark_data/audio
    python tests/benchmark_thesis.py rag --queries benchmark_data/queries.json --ks 3 5
    python tests/benchmark_thesis.py latency --audio benchmark_data/audio/sample.wav --runs 10

Every run also writes a JSON report to ``benchmark_results/``.
"""

import argparse
import asyncio
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

AUDIO_EXTENSIONS = (".wav", ".mp3", ".webm", ".ogg", ".m4a")
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "benchmark_results"))

WER_TARGET = 0.12
RECALL_TARGET = 0.85
LATENCY_TARGET_MS = 2500.0


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(pct / 100.0 * len(ordered)) - 1))
    return ordered[index]


def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 2),
        "p50": round(percentile(values, 50), 2),
        "p95": round(percentile(values, 95), 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
    }


def dcg(relevances: List[int]) -> float:
    return sum(rel / math.log2(rank + 2) for rank, rel in enumerate(relevances))


def ndcg_at_k(relevances: List[int], k: int) -> float:
    """Binary-relevance NDCG@K; ideal ranking puts all relevant hits first."""
    top = relevances[:k]
    ideal = sorted(relevances, reverse=True)[:k]
    ideal_dcg = dcg(ideal)
    return dcg(top) / ideal_dcg if ideal_dcg > 0 else 0.0


def save_report(name: str, report: Dict[str, Any]) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{name}_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"\nReport saved to {path}")
    return path


def verdict(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


# ------------------------------------------------------------------------------ WER
def run_wer(args: argparse.Namespace) -> Dict[str, Any]:
    from app.services.asr_service import get_asr_service

    asr = get_asr_service()
    if not asr.is_ready():
        raise SystemExit("ASR model is not loaded.")

    rows = []
    for name in sorted(os.listdir(args.audio_dir)):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in AUDIO_EXTENSIONS:
            continue
        ref_path = os.path.join(args.audio_dir, stem + ".txt")
        if not os.path.isfile(ref_path):
            print(f"  skip {name}: no reference {stem}.txt")
            continue
        with open(ref_path, encoding="utf-8") as fh:
            reference = fh.read().strip()
        with open(os.path.join(args.audio_dir, name), "rb") as fh:
            audio = fh.read()

        started = time.perf_counter()
        metrics = asr.transcribe_audio_stream(audio, expected_transcript=reference)
        elapsed = (time.perf_counter() - started) * 1000
        rows.append(
            {
                "file": name,
                "wer": metrics.wer_score,
                "cer": metrics.cer_score,
                "asr_ms": round(elapsed, 1),
                "hypothesis": metrics.transcription,
            }
        )
        print(f"  {name}: WER={metrics.wer_score} CER={metrics.cer_score} ({elapsed:.0f} ms)")

    wers = [r["wer"] for r in rows if r["wer"] is not None]
    cers = [r["cer"] for r in rows if r["cer"] is not None]
    mean_wer = statistics.fmean(wers) if wers else float("nan")
    print(f"\nSamples: {len(rows)} | mean WER: {mean_wer:.2%} (target < {WER_TARGET:.0%}) "
          f"-> {verdict(wers != [] and mean_wer < WER_TARGET)}")
    return {
        "benchmark": "wer",
        "samples": len(rows),
        "wer": summarize(wers),
        "cer": summarize(cers),
        "target_wer": WER_TARGET,
        "passed": bool(wers) and mean_wer < WER_TARGET,
        "rows": rows,
    }


# ------------------------------------------------------------------------------ RAG
def run_rag(args: argparse.Namespace) -> Dict[str, Any]:
    from app.services.rag_service import get_rag_service

    rag = get_rag_service()
    if not rag.is_ready():
        raise SystemExit("RAG service is not ready.")
    print(f"Collection size: {rag.collection.count()} vectors")

    with open(args.queries, encoding="utf-8") as fh:
        queries = json.load(fh)

    max_k = max(args.ks)
    hits_at = {k: 0 for k in args.ks}
    ndcg_at = {k: [] for k in args.ks}
    latencies: List[float] = []
    rows = []

    for item in queries:
        started = time.perf_counter()
        results = rag.query_corpus(item["query"], top_k=max_k)
        latencies.append((time.perf_counter() - started) * 1000)

        needles = [s.lower() for s in item["relevant_sources"]]
        relevance = [int(any(n in str(r["source"]).lower() for n in needles)) for r in results]
        for k in args.ks:
            hits_at[k] += int(any(relevance[:k]))
            ndcg_at[k].append(ndcg_at_k(relevance, k))
        rows.append({"query": item["query"], "relevance": relevance,
                     "top_sources": [r["source"] for r in results]})

    total = len(queries) or 1
    recall = {f"recall@{k}": round(hits_at[k] / total, 4) for k in args.ks}
    ndcg = {f"ndcg@{k}": round(statistics.fmean(v), 4) if v else 0.0 for k, v in ndcg_at.items()}
    primary = 3 if 3 in args.ks else args.ks[0]
    passed = recall[f"recall@{primary}"] >= RECALL_TARGET
    print(f"\nQueries: {len(queries)} | {recall} | {ndcg}")
    print(f"Recall@{primary} target >= {RECALL_TARGET:.0%} -> {verdict(passed)}")
    return {
        "benchmark": "rag",
        "queries": len(queries),
        **recall,
        **ndcg,
        "search_latency_ms": summarize(latencies),
        "target_recall": RECALL_TARGET,
        "passed": passed,
        "rows": rows,
    }


# -------------------------------------------------------------------------- Latency
async def _latency_once(audio: bytes, scenario: str) -> Dict[str, float]:
    from app.schemas.request_schemas import TargetScenario
    from app.services.asr_service import get_asr_service
    from app.services.llm_service import get_llm_service
    from app.services.rag_service import get_rag_service

    asr, rag, llm = get_asr_service(), get_rag_service(), get_llm_service()
    scenario_enum = TargetScenario(scenario)

    t0 = time.perf_counter()
    metrics = await asr.atranscribe_audio_stream(audio)
    t1 = time.perf_counter()
    context = await rag.aquery_corpus(metrics.transcription, top_k=3, scenario=scenario_enum.value)
    t2 = time.perf_counter()
    await llm.evaluate_technical_english(
        candidate_text=metrics.transcription,
        target_scenario=scenario_enum.value,
        rag_context_items=context,
        asr_metrics=metrics,
        start_time_monotonic=t2,
    )
    t3 = time.perf_counter()
    return {
        "asr_ms": (t1 - t0) * 1000,
        "vector_search_ms": (t2 - t1) * 1000,
        "llm_ms": (t3 - t2) * 1000,
        "total_ms": (t3 - t0) * 1000,
    }


def run_latency(args: argparse.Namespace) -> Dict[str, Any]:
    with open(args.audio, "rb") as fh:
        audio = fh.read()

    async def runner() -> List[Dict[str, float]]:
        await _latency_once(audio, args.scenario)  # warm-up (not counted)
        return [await _latency_once(audio, args.scenario) for _ in range(args.runs)]

    samples = asyncio.run(runner())
    breakdown = {key: summarize([s[key] for s in samples]) for key in samples[0]}
    p95_total = breakdown["total_ms"]["p95"]
    passed = p95_total < LATENCY_TARGET_MS
    for key, stats in breakdown.items():
        print(f"  {key:>17}: mean={stats['mean']} p50={stats['p50']} p95={stats['p95']}")
    print(f"\nTotal p95 {p95_total} ms (target < {LATENCY_TARGET_MS:.0f} ms) -> {verdict(passed)}")
    return {
        "benchmark": "latency",
        "runs": args.runs,
        "breakdown": breakdown,
        "target_total_ms": LATENCY_TARGET_MS,
        "passed": passed,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tech Tutor Engine thesis benchmarks")
    sub = parser.add_subparsers(dest="command", required=True)

    wer = sub.add_parser("wer", help="Word/Character Error Rate")
    wer.add_argument("--audio-dir", required=True)
    wer.set_defaults(func=run_wer)

    rag = sub.add_parser("rag", help="Recall@K and NDCG@K")
    rag.add_argument("--queries", required=True)
    rag.add_argument("--ks", type=int, nargs="+", default=[3, 5])
    rag.set_defaults(func=run_rag)

    lat = sub.add_parser("latency", help="End-to-end latency profile")
    lat.add_argument("--audio", required=True)
    lat.add_argument("--runs", type=int, default=10)
    lat.add_argument("--scenario", default="system_design_interview")
    lat.set_defaults(func=run_latency)
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    save_report(parsed.command, parsed.func(parsed))
