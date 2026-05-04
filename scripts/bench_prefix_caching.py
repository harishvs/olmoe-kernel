"""Measure the effect of prefix caching on TTFT.

For each prefix length P in {128, 512, 2048}, build a shared prefix and 30
requests each consisting of (prefix || unique 64-token suffix). Measure TTFT
twice: once with prefix caching off (baseline), once with it on. The delta
per P is the lever's impact for that workload shape.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer

from scripts.harness.client import Request
from scripts.harness.run import make_synthetic_prompt, run_experiment

MODEL = "allenai/OLMoE-1B-7B-0924-Instruct"
SUFFIX_LEN = 64
N_REQUESTS = 30
WARMUP = 3
PREFIX_LENGTHS = [128, 512, 2048]
RESULTS_DIR = Path("results/harness/prefix_caching")


def build_requests(prefix_len: int, suffix_len: int, n: int) -> list[Request]:
    """Build n requests sharing the same P-token prefix but with unique S-token suffixes."""
    prefix = make_synthetic_prompt(prefix_len, model=MODEL, seed=0)

    reqs: list[Request] = []
    for i in range(n):
        # Unique suffix per request by seeding on the iteration index.
        suffix = make_synthetic_prompt(suffix_len, model=MODEL, seed=1000 + i)
        full = prefix + " " + suffix
        reqs.append(Request(prompt=full, max_tokens=1, label=f"P{prefix_len}_iter{i}"))
    return reqs


def run_one(*, cache_on: bool, prefix_len: int, timestamp: str):
    label = (
        f"prefix_{'on' if cache_on else 'off'}_P{prefix_len}_S{SUFFIX_LEN}_"
        f"{timestamp}"
    )
    engine_args = [
        "--dtype", "bfloat16",
        "--gpu-memory-utilization", "0.85",
    ]
    if cache_on:
        engine_args += ["--enable-prefix-caching"]
    else:
        engine_args += ["--no-enable-prefix-caching"]

    reqs = build_requests(prefix_len, SUFFIX_LEN, N_REQUESTS)
    return run_experiment(
        engine="vllm",
        model=MODEL,
        engine_args=engine_args,
        requests=reqs,
        warmup=WARMUP,
        seed=42,
        label=label,
        results_dir=RESULTS_DIR,
    )


def main():
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    summary: dict = {}
    for P in PREFIX_LENGTHS:
        print(f"\n=== P={P}, cache OFF ===")
        off = run_one(cache_on=False, prefix_len=P, timestamp=timestamp)
        print(f"\n=== P={P}, cache ON ===")
        on = run_one(cache_on=True, prefix_len=P, timestamp=timestamp)
        summary[P] = {
            "off_ttft_ms": off.ttft_ms,
            "on_ttft_ms": on.ttft_ms,
            "delta_ms": off.ttft_ms["mean"] - on.ttft_ms["mean"],
            "delta_pct": 100.0 * (off.ttft_ms["mean"] - on.ttft_ms["mean"]) / off.ttft_ms["mean"],
        }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / f"summary_{timestamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[summary] wrote {summary_path}\n")

    print("=" * 66)
    print(f"{'Prefix':>8}  {'Cache OFF':>12}  {'Cache ON':>12}  {'Saved':>10}")
    print("=" * 66)
    for P, s in summary.items():
        off = s["off_ttft_ms"]["mean"]
        on_ = s["on_ttft_ms"]["mean"]
        print(
            f"P={P:>5}  {off:>8.1f} ms    {on_:>8.1f} ms    "
            f"{s['delta_pct']:>6.1f}% ({s['delta_ms']:.1f} ms)"
        )


if __name__ == "__main__":
    main()
