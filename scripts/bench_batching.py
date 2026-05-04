"""Measure the effect of concurrent batching on per-request latency and throughput.

Uses ShareGPT prompts filtered to a target token-length band so real prompts
don't accidentally share prefixes (which would make prefix caching merge the
requests and inflate throughput).

Sweeps concurrency B in {1, 4, 16, 32}. For each B, maintains B in-flight
requests for a fixed measurement window and records:
  - requests/sec (throughput)
  - output tokens/sec (throughput)
  - per-request TTFT and TPOT (latency distribution)
"""
from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer

from scripts.harness.client import run_concurrent_workload
from scripts.harness.datasets import load_sharegpt_prompts
from scripts.harness.server import ServerConfig, serve

MODEL = "allenai/OLMoE-1B-7B-0924-Instruct"
CONCURRENCIES = [1, 4, 16, 32]
PROMPT_MIN_TOKENS = 128
PROMPT_MAX_TOKENS = 512
POOL_SIZE = 256  # enough distinct prompts that we never repeat within a run
OUTPUT_TOKENS = 128
WARMUP_S = 10.0
MEASURE_S = 30.0
RESULTS_DIR = Path("results/harness/batching")


def main():
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    print(f"[batching] loading {POOL_SIZE} ShareGPT prompts in "
          f"[{PROMPT_MIN_TOKENS}, {PROMPT_MAX_TOKENS}] tokens...")
    pool = load_sharegpt_prompts(
        n=POOL_SIZE,
        min_tokens=PROMPT_MIN_TOKENS,
        max_tokens=PROMPT_MAX_TOKENS,
        tokenizer=tok,
        seed=42,
        max_output_tokens=OUTPUT_TOKENS,
    )
    print(f"[batching] got {len(pool)} prompts")

    cfg = ServerConfig(
        engine="vllm",
        model=MODEL,
        extra_args=[
            "--dtype", "bfloat16",
            "--gpu-memory-utilization", "0.85",
            "--no-enable-prefix-caching",
            "--max-num-seqs", "64",
        ],
    )

    summary: dict = {}
    # Single server launch, sweep concurrencies against it
    with serve(cfg) as url:
        for B in CONCURRENCIES:
            print(f"\n[batching] concurrency={B}")
            result = run_concurrent_workload(
                url=url,
                model=MODEL,
                request_pool=pool,
                concurrency=B,
                duration_s=MEASURE_S,
                warmup_s=WARMUP_S,
                label=f"batching_B{B}",
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
            out_path = RESULTS_DIR / f"batching_B{B}_{timestamp}.json"
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
