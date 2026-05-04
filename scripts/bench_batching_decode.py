"""Measure batching on pure-decode workload (tiny prefill, long decode).

Same harness as bench_batching.py but uses synthetic 1-token prompts instead of
ShareGPT prompts. Isolates the MoE-batching question from prefill-mixing.

If OLMoE batching amortizes cleanly at all, it should amortize here. If
TPOT still grows badly, the culprit is MoE routing at high batch, not
prefill mixing.
"""
from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.harness.client import Request, run_concurrent_workload
from scripts.harness.run import make_synthetic_prompt
from scripts.harness.server import ServerConfig, serve

MODEL = "allenai/OLMoE-1B-7B-0924-Instruct"
CONCURRENCIES = [1, 4, 16, 32, 64, 128]
POOL_SIZE = 256
OUTPUT_TOKENS = 128
WARMUP_S = 10.0
MEASURE_S = 30.0
RESULTS_DIR = Path("results/harness/batching_decode")


def main():
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Each request gets a unique 1-token prompt (via different seeds) so
    # concurrent requests don't accidentally share a prefix.
    pool = [
        Request(
            prompt=make_synthetic_prompt(1, model=MODEL, seed=2000 + i),
            max_tokens=OUTPUT_TOKENS,
            label=f"decode_only_seed{2000 + i}",
        )
        for i in range(POOL_SIZE)
    ]

    cfg = ServerConfig(
        engine="vllm",
        model=MODEL,
        extra_args=[
            "--dtype", "bfloat16",
            "--gpu-memory-utilization", "0.85",
            "--no-enable-prefix-caching",
            "--max-num-seqs", "256",
        ],
    )

    summary: dict = {}
    with serve(cfg) as url:
        for B in CONCURRENCIES:
            print(f"\n[batching-decode] concurrency={B}")
            result = run_concurrent_workload(
                url=url,
                model=MODEL,
                request_pool=pool,
                concurrency=B,
                duration_s=MEASURE_S,
                warmup_s=WARMUP_S,
                label=f"batching_decode_B{B}",
            )
            summary[B] = {
                "n_completed": result.n_completed,
                "duration_s": result.duration_s,
                "requests_per_s": result.requests_per_s,
                "output_tokens_per_s": result.output_tokens_per_s,
                "ttft_ms": result.ttft_ms,
                "tpot_ms": result.tpot_ms,
                "total_ms": result.total_ms,
            }
            out_path = RESULTS_DIR / f"batching_decode_B{B}_{timestamp}.json"
            with open(out_path, "w") as f:
                json.dump(
                    {
                        "concurrency": B,
                        "n_completed": result.n_completed,
                        "duration_s": result.duration_s,
                        "requests_per_s": result.requests_per_s,
                        "output_tokens_per_s": result.output_tokens_per_s,
                        "ttft_ms": result.ttft_ms,
                        "tpot_ms": result.tpot_ms,
                        "total_ms": result.total_ms,
                        "raw": [dataclasses.asdict(r) for r in result.raw],
                    },
                    f,
                    indent=2,
                )
            print(f"  completed {result.n_completed} requests in "
                  f"{result.duration_s:.1f}s")
            print(f"  {result.requests_per_s:.2f} req/s, "
                  f"{result.output_tokens_per_s:.0f} tok/s")
            print(f"  TTFT mean {result.ttft_ms['mean']:.1f} ms, "
                  f"p90 {result.ttft_ms['p90']:.1f} ms")
            print(f"  TPOT mean {result.tpot_ms['mean']:.2f} ms, "
                  f"p90 {result.tpot_ms['p90']:.2f} ms")

    summary_path = RESULTS_DIR / f"summary_{timestamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[summary] wrote {summary_path}\n")

    print("=" * 76)
    print(f"{'B':>4}  {'req/s':>8}  {'tok/s':>8}  {'TTFT mean':>11}  {'TTFT p90':>10}  {'TPOT':>8}")
    print("=" * 76)
    for B, s in summary.items():
        print(
            f"{B:>4}  {s['requests_per_s']:>7.2f}  "
            f"{s['output_tokens_per_s']:>7.0f}  "
            f"{s['ttft_ms']['mean']:>9.1f}    "
            f"{s['ttft_ms']['p90']:>8.1f}    "
            f"{s['tpot_ms']['mean']:>6.2f}"
        )


if __name__ == "__main__":
    main()
