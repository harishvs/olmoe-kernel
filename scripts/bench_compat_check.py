"""Compatibility check: does the harness produce numbers close to vllm bench latency?

Compares harness TTFT/TPOT to our committed baseline (bf16 OLMoE, L=1024, O=128:
TTFT ~94 ms, TPOT ~10.7 ms).
"""
from pathlib import Path

from scripts.harness.run import make_synthetic_prompt, run_experiment
from scripts.harness.client import Request


def main():
    model = "allenai/OLMoE-1B-7B-0924-Instruct"
    prompt = make_synthetic_prompt(n_tokens=1024, model=model)

    # Sanity-check the token count before launching a 5-min run.
    from transformers import AutoTokenizer
    verified = len(AutoTokenizer.from_pretrained(model)(prompt, add_special_tokens=False)["input_ids"])
    print(f"[compat] prompt tokenizes to {verified} tokens (target: 1024)")

    reqs = [Request(prompt=prompt, max_tokens=128, label=f"iter_{i}") for i in range(30)]

    result = run_experiment(
        engine="vllm",
        model=model,
        engine_args=[
            "--dtype", "bfloat16",
            "--gpu-memory-utilization", "0.85",
            "--no-enable-prefix-caching",
        ],
        requests=reqs,
        warmup=3,
        seed=42,
        label="compat_check_bf16_L1024_O128",
        results_dir=Path("results/harness"),
    )

    print("\n=== Compat check results ===")
    print(f"TTFT mean:  {result.ttft_ms['mean']:.2f} ms  (baseline: ~94 ms)")
    print(f"TTFT p90:   {result.ttft_ms['p90']:.2f} ms")
    print(f"TPOT mean:  {result.tpot_ms['mean']:.2f} ms  (baseline: ~10.7 ms)")
    print(f"TPOT p90:   {result.tpot_ms['p90']:.2f} ms")


if __name__ == "__main__":
    main()
