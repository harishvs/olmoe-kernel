# Is it fast? Profiling and optimizing OLMoE inference on one L4

A reproducible measurement of OLMoE-1B-7B-Instruct served by vLLM on a single NVIDIA L4 GPU (AWS g6.4xlarge). Benchmarks, profiles, accuracy evals, and all raw JSON checked in — clone and rerun against your own hardware.

## What's measured

| Lever | Result | Where |
|---|---|---|
| Baseline TTFT / TPOT | 94 ms / 10.7 ms at L=1024 | `results/baseline_*.json` |
| Kernel-level profile | `fused_moe_kernel` = 80% of TTFT; router = 0.09% | `results/profile/` |
| FP8 quantization | TTFT -36%, TPOT -33%, end-to-end -32% | `results/fp8_*.json` |
| FP8 accuracy (4 tasks) | MMLU/HellaSwag/ARC within noise; GSM8K -1.9 pp | `results/eval/` |
| Prefix caching | TTFT -67% at 2048-token shared prefix | `results/harness/prefix_caching/` |
| Batching | 2055 tok/s peak at B=128; MoE tax below B=32 | `results/harness/batching_decode/` |
| TensorRT-LLM | Not applicable — OLMoE not supported in TRT-LLM 1.2 | (verified, not measured) |

Full writeup: [`talk/story.md`](talk/story.md).

## Requirements

- **Hardware:** NVIDIA GPU with compute capability 8.9+ (tested on L4). Most scripts need ~24 GB VRAM.
- **OS:** Linux with recent NVIDIA driver + CUDA runtime (tested on Ubuntu 24.04, driver 580+).
- **Python:** 3.12.
- **Disk:** ~20 GB for the model download, another ~1 GB for ShareGPT dataset if you run batching.

## Setup

This repo uses [uv](https://docs.astral.sh/uv/) for Python environment management.

```bash
git clone https://github.com/harishvs/olmoe-kernel
cd olmoe-kernel
uv sync
```

`uv sync` creates `.venv/` and installs all pinned dependencies. The first run will also pull the OLMoE model (~14 GB) the first time you hit it.

## Reproducing the main results

Run these in order. Each writes JSON under `results/` with a UTC timestamp, so reruns don't clobber old data.

### 1. Smoke test — confirm vLLM + OLMoE + GPU work

```bash
uv run python scripts/smoke_test.py
```

Loads OLMoE, generates one token. Takes ~60 seconds on first run (weight download + CUDA graph capture).

### 2. Baseline TTFT/TPOT sweep

```bash
uv run python scripts/benchmark_ttft.py
```

Sweeps `input_len ∈ {32, 128, 512, 1024} × output_len ∈ {1, 128}`, 30 iterations per cell, seed 42. Writes `results/baseline_<timestamp>_L<N>_O<M>_seed42.json`. Runtime ~20 minutes.

### 3. FP8 sweep (same grid, `--quantization fp8`)

```bash
uv run python scripts/benchmark_fp8.py
```

### 4. Kernel-level profile with torch.profiler

```bash
uv run python scripts/profile_torch.py   # bf16
uv run python scripts/profile_fp8.py     # fp8
uv run python scripts/profile_decode.py  # decode-only, both bf16 and fp8
```

Each writes a Chrome-trace JSON plus a text kernel summary under `results/profile/<run_dir>/`. Open the trace at [perfetto.dev](https://perfetto.dev) to visualize.

### 5. Accuracy evals via lm-evaluation-harness

```bash
uv run python scripts/eval_suite.py
```

Runs MMLU, GSM8K, HellaSwag, ARC-Challenge against bf16 and FP8 back-to-back. ~2 hours. Writes to `results/eval/`.

### 6. Engine-agnostic HTTP harness (prefix caching, batching)

```bash
# Verify the harness produces numbers matching vllm bench latency
uv run python -m scripts.bench_compat_check

# Prefix caching experiment
uv run python -m scripts.bench_prefix_caching

# Batching sweep with ShareGPT prompts (needs data/sharegpt.json)
uv run python -m scripts.bench_batching

# Pure-decode batching sweep (1-tok prompt, isolates MoE tax from prefill-mixing)
uv run python -m scripts.bench_batching_decode
```

The harness launches `vllm serve` in a subprocess, hits its OpenAI-compatible endpoint, and tears it down. Engine-agnostic: point at any OpenAI-compatible URL (TRT-LLM, SGLang) with one line change.

**For batching:** download ShareGPT V3 first. `data/` is gitignored (license ambiguity + size).

```bash
mkdir -p data
curl -L -o data/sharegpt.json \
  https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

## Regenerating the plots

All plots under `talk/figures/` are reproducible from the committed JSON:

```bash
uv run python scripts/plots/ttft_vs_length.py
uv run python scripts/plots/tpot_vs_length.py
uv run python scripts/plots/prefix_caching.py
uv run python scripts/plots/batching.py
uv run python scripts/plots/batching_decode_compare.py
```

## Repo layout

```
scripts/
  benchmark_ttft.py       # TTFT/TPOT grid sweep (bf16)
  benchmark_fp8.py        # same grid with --quantization fp8
  profile_torch.py        # torch.profiler on prefill at L=1024
  profile_fp8.py, profile_decode.py
  eval_suite.py           # lm-eval-harness, bf16 vs FP8
  smoke_test.py           # end-to-end sanity check
  bench_compat_check.py   # harness vs vllm-bench-latency reconciliation
  bench_prefix_caching.py # prefix caching experiment
  bench_batching.py       # ShareGPT concurrent workload
  bench_batching_decode.py # pure-decode concurrent workload
  harness/
    server.py             # launches vLLM subprocess, waits for /v1/models
    client.py             # streaming HTTP client, captures TTFT + TPOT
    datasets.py           # ShareGPT loader
    run.py                # experiment runner (server + client + JSON writer)
  plots/                  # matplotlib generators for talk/figures
results/
  baseline_*.json         # vllm bench latency outputs
  fp8_*.json              # same with FP8 quantization
  eval/                   # lm-evaluation-harness outputs
  harness/                # HTTP-harness outputs
  profile/                # torch.profiler traces + summaries
talk/
  story.md                # full narrative writeup of the experiment
  figures/                # embedded plots and diagrams
tasks/
  requirements.md         # user stories
  design.md               # methodology + lever board
  kernel_learning.md      # 12-week Triton / cuTile learning roadmap
  todo.md                 # progress checklist
```

## Method, in one paragraph

Measure before you optimize. Model the hypothesis with back-of-envelope FLOP math. Profile to verify. Enumerate the levers. Measure each one against the same baseline with the same seed. Report tradeoffs, not decrees. The kernel — if you ever write one — is the easy part; the hard part is knowing whether you should. For this workload on this engine, the profile said don't.

## License

MIT for the code in this repo. Model weights and datasets referenced have their own licenses — check the upstream sources before redistributing.
