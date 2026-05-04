# Design: Profiling and optimizing LLM inference for OLMoE on L4

## The question we're answering

*A user is running OLMoE-1B-7B-Instruct on a single-L4 instance (AWS g6.4xlarge) with vLLM. Are they getting good performance for prefill and decode? What can they do to improve it?*

Everything below is the measurement pipeline, the analysis framework, and the set of optimization levers we evaluate in order to answer that question with data.

## Methodology

```
1. Baseline:  measure prefill (TTFT) and decode (TPOT) with default vLLM config
2. Profile:   torch.profiler on the target shape to see the kernel breakdown
3. Levers:    apply one optimization at a time, re-measure, attribute the delta
4. Verdict:   compare measured deltas to expected upside; recommend which levers matter
```

No lever gets evaluated until the profile justifies looking at it. No kernel work happens unless the profile shows a kernel-level opportunity that isn't already covered by vLLM's production kernels.

## Components

### 1. Benchmark harness (`scripts/benchmark_ttft.py`)
Drives `vllm bench latency` across a grid of input/output lengths.

- **Input lengths:** 32, 128, 512, 1024 — a realistic prefill span.
- **Output lengths:** 1 (prefill-only → TTFT) and 128 (prefill + decode → end-to-end).
- **30 timed iterations per cell**, 3 warmup iterations discarded.
- **Seed pinned** so the synthetic token sequences are identical across runs.
- **Output:** per-run JSON under `results/` with percentile latencies.

Decode per-token (TPOT) is derived from the two output-length runs:
`TPOT ≈ (mean_latency(L, O=128) − mean_latency(L, O=1)) / 127`

### 2. Profiler (`scripts/profile_torch.py`)
Drives `vllm bench latency` with `--profile` and `--profiler-config.profiler=torch`.
Produces a Chrome-trace file for Perfetto/`chrome://tracing` plus a text summary of
top kernels by CUDA time.

Target configuration for profiling: `input-len=1024, output-len=1, batch-size=1, seed=42`
— the longest-prefill cell of the baseline sweep.

### 3. Optimization levers

Each lever gets a dedicated benchmark run with otherwise-identical configuration.
The name and status are recorded here as they are attempted.

| Lever | Expected magnitude | Measured delta | Status |
|---|---|---|---|
| vLLM defaults (baseline) | — | — | measured |
| Prefix caching | 0–90% (workload-dependent) | -18% to -67% across P in {128, 512, 2048} | ✅ measured |
| FP8 quantization | 15–30% on L4 | TTFT −36%, TPOT −33% @ L=1024 | ✅ measured |
| TensorRT-LLM backend | 10–30% | not applicable — OLMoE not supported in TRT-LLM 1.2 | ❌ verified |
| H100 upgrade (same stack) | 40–60% | — | not measured |
| Custom kernel (if profile justifies) | workload-specific | — | not applicable — profile shows no gap |

Each lever's measurement goes next to the baseline in `results/` with a lever suffix
in the filename.

## What we are *not* doing, and why

We are not writing a custom kernel for MoE routing. The profile at `input-len=1024`
shows `fused_moe_kernel` at ~80% of TTFT and the router at ~0.09%. vLLM's production
CUDA kernels (grouped GEMM on tensor cores via CUTLASS-style implementations) already
occupy the hot path. A solo custom kernel cannot close the tensor-core gap in
the time budget of this project and would deliver no measurable TTFT improvement
even if it matched vLLM's performance on the specific op it targets.

The talk's technical conclusion is that for this workload on this hardware with
this engine, the levers that move performance are configuration, quantization,
and hardware — not custom kernels.

## Data flow (MoE layer, for reference)

```
hidden_states (num_tokens, 2048)
  → router linear (2048 → 64) → logits (num_tokens, 64)
  → softmax → top-K (K=8) → expert assignments
  → dispatch (permute tokens to experts)
  → grouped expert GEMM (where ~80% of FFN time lives)
  → combine (weighted sum back to original positions)
```

This is kept in the design doc because the FLOP analysis in the talk references
these shapes. It is not the target of optimization.
