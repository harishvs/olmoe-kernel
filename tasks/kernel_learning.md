# Kernel Learning Path

A 12-week plan for going from "can read a profiler" to "can write a
production-grade GPU kernel in Triton for an underserved workload." Grounded
in the measurements already collected in this repo — OLMoE-1B-7B on L4 under
vLLM, with bf16 and FP8 baselines profiled at the kernel level.

## Assumptions

- **Starting point:** fluent in Python and PyTorch. Comfortable reading
  `torch.profiler` output. Has run and profiled OLMoE under vLLM. No prior
  CUDA or Triton kernel-writing experience.
- **Hardware:** single L4 GPU on `g6.4xlarge` (Ada, compute capability 8.9,
  24 GB VRAM, 120 bf16 TFLOPS, 300 GB/s HBM).
- **Time budget:** ~8 hours/week of focused work. 100 hours total across 12
  weeks. Skippable to ~50 hours via the compressed track at the bottom.
- **Tools already installed in the venv:** vLLM, Triton 3.6, CUDA toolkit
  13.0, cuda-tile 1.3 (with the tileiras compiler), torch profiler.
- **Missing and needed:** Nsight Compute (install in week 4).

## Why this path, in one paragraph

Our profile at `L=1024, output-len=1` showed `fused_moe_kernel` consuming 80%
of TTFT, `topk_softmax` at 0.09%, and attention at ~12%. The lesson is that
mature stacks leave almost no room on the hot path. The learning path is not
"write a faster X than vLLM." It is "develop the skills to recognize where
there *is* headroom — underserved shapes, new architectures, ancillary
operations — and write the kernel that captures it." Triton is the right
first tool because the Python-to-tile programming model is the industry
standard for this work. cuTile is the second tool because it is NVIDIA's
first-party answer and worth fluency in.

## Phase 1 — Triton fundamentals (weeks 1-3)

**Goal:** write basic Triton kernels, internalize the tile programming model,
get a feel for the autotuning loop.

### Week 1 — the 8 official tutorials

Work through all of them in order. Each takes ~1-3 hours depending on depth.

1. Vector addition
2. Fused softmax
3. Matrix multiplication
4. Low-memory dropout
5. Layer normalization
6. Fused attention (this is the hardest; allocate 3+ hours)
7. Libdevice math functions
8. Grouped GEMM

Do not skim. For each: run it, change tile sizes, run the benchmark at
different shapes, note where the curve bends. The matmul and attention
tutorials are load-bearing for everything later.1

**Deliverable:** `kernels/week1/` directory with one modified version of each
tutorial and a one-paragraph writeup of something surprising you learned.

### Week 2 — RMSNorm from scratch

RMSNorm is a bandwidth-bound kernel with clean shapes. Perfect first original
kernel.

- Read the RMSNorm equation. Hand-compute for a small example.
- Write a Triton kernel that takes `(batch, hidden)` input and returns
  normalized output. Use one program per row.
- Benchmark against `torch.nn.functional.rms_norm` across
  `hidden in {512, 1024, 2048, 4096, 8192}`.
- Target: match or beat PyTorch.

Expected outcome: you will match PyTorch within a few percent. If you beat
it, examine why. If you lose, examine why (usually block size choice or
failure to vectorize loads).

**Deliverable:** `kernels/week2/rmsnorm.py` with benchmarks.

### Week 3 — SiLU × Mul (the gated activation)

OLMoE's expert FFNs use gated activations: the up-projection produces two
tensors, one goes through SiLU, the other stays linear, then they're
multiplied. vLLM fuses this into `silu_and_mul` (saw it in our profile: 2.3 ms
at L=1024 baseline, 2.3 ms under FP8 — bandwidth-limited).

- Write the fused Triton kernel. Input: two tensors of shape `(tokens, d)`.
  Output: `silu(x) * y`.
- Benchmark against `torch.nn.functional.silu(x) * y` (unfused reference).
- Then benchmark against vLLM's implementation:
  `vllm.model_executor.layers.activation.SiluAndMul`.

Expected outcome: you beat the unfused reference by ~2×. You will probably
match but not beat vLLM's version. That's fine.

**Deliverable:** `kernels/week3/silu_mul.py` with benchmarks vs both
references.

## Phase 2 — Reading the hardware (weeks 4-6)

**Goal:** stop guessing why kernels are fast or slow. Use Nsight Compute to
read rooflines, identify bottlenecks from data, not instinct.

### Week 4 — Install Nsight Compute; profile vLLM's fused_moe_kernel

Install `nsight-compute` (`ncu`). Profile vLLM at three batch sizes:

- Batch 1: what we measured. Bandwidth-bound.
- Batch 8: a typical small-serving batch.
- Batch 64: saturating batch.

For each, capture `ncu --set full` output on `fused_moe_kernel`. Produce a
roofline chart. Identify:

- Peak bandwidth utilization (what % of L4's 300 GB/s)
- Peak compute utilization (what % of 120 TFLOPS bf16)
- Where the kernel sits on the roofline curve

**Deliverable:** `notes/week4_roofline.md` with three roofline charts and a
paragraph explaining the transition from bandwidth-bound to compute-bound as
batch size grows.

### Week 5 — Profile your own kernels

Take the RMSNorm and SiLU×Mul kernels from weeks 2-3. Profile them with
Nsight Compute. For each:

- Is it bandwidth-bound or compute-bound?
- What % of peak are you achieving?
- Where's the gap? Bank conflicts? Register pressure? Launch overhead?

If you're below 70% of the appropriate peak, iterate on tile sizes until you
hit 70%+. Document what changed and why.

**Deliverable:** updated versions of the week 2-3 kernels with roofline
analysis and a "why it's as fast as it is" writeup.

### Week 6 — Read FlashAttention-2 source

No kernel writing this week. Reading.

Source: `flash-attn` on GitHub. Read `flash_attn/flash_attn_triton.py` (the
Triton version, much more approachable than the C++ one). Produce a writeup
answering:

- How does FlashAttention avoid materializing the `N × N` attention matrix?
- What are the tile shapes and why those specifically?
- How does online softmax work? Walk through the math for a 2-block example.
- Where are the warp-level primitives used?

This is the single most important kernel to understand in modern inference.
Don't skip it.

**Deliverable:** `notes/week6_flashattn.md` answering the above questions in
your own words. 500-1500 words.

## Phase 3 — Port a real kernel (weeks 7-9)

**Goal:** write one kernel that's actually useful. Deliver a portfolio piece
that demonstrates end-to-end competence.

### Week 7 — pick the target and study the baseline

**Recommended target: shape-specialized MoE expert GEMM for batch=1 on L4.**

Why this target:
- vLLM's `fused_moe_kernel` is tuned for generic shapes. Batch=1 is on the
  edge of its tuning envelope.
- OLMoE's exact shapes (num_experts=64, top_k=8, d_model=2048, d_ff=1024)
  are known and fixed.
- L4 (sm_89) is not the primary tuning target for the kernel authors —
  Hopper is.

Study the baseline:

- Read vLLM's `fused_moe_kernel` source (in
  `vllm/model_executor/layers/fused_moe/`).
- Run it at batch=1 with our OLMoE config. Capture Nsight Compute output.
- Identify: is it bandwidth-bound or compute-bound at batch=1? What's the
  gap to roofline peak?

**Deliverable:** `notes/week7_target.md` with the bottleneck analysis and a
concrete hypothesis about where headroom exists.

### Week 8 — write v1

Write a Triton kernel specialized for:

- num_experts = 64 (hardcoded, not a parameter)
- top_k = 8 (hardcoded)
- d_model = 2048, d_ff = 1024 (hardcoded)
- batch = 1, seq_len variable (actual use case)
- bf16 precision

Do not try to be general. The whole point is that specialization is where the
win comes from.

Benchmark against vLLM's `fused_moe_kernel` at sequence lengths 32, 128, 512,
1024. Report raw numbers with no spin.

**Deliverable:** `kernels/week8/moe_gemm_v1.py` + benchmark results.

### Week 9 — iterate v1 → v2

Based on Nsight Compute output for v1, iterate:

- Try different tile shapes (M, N, K tiles)
- Check for shared memory bank conflicts
- Check register pressure — if you're spilling, tile smaller
- Check for warp divergence in the expert-routing branch
- Try `num_warps` and `num_stages` autotune sweeps

Realistic outcome: v2 is 1.1-1.5× faster than v1 but still within 10-30% of
vLLM's generic kernel. If v2 is faster than vLLM at batch=1, investigate
carefully — make sure you're measuring correctly and there's no subtle
correctness bug.

If v2 is slower than vLLM, the gap analysis is valuable: write up why. This
is a realistic outcome and the writeup is what makes it a portfolio piece.

**Deliverable:** `kernels/week9/moe_gemm_v2.py` + benchmarks + a
`notes/week9_writeup.md` explaining what worked, what didn't, and what the
final gap to vLLM is and why.

## Phase 4 — Stretch (weeks 10-12)

Choose one of two tracks depending on what interests you more.

### Track A — cuTile port

Port the week 9 kernel to cuTile. Requires rewriting, not translating —
cuTile's tile model is similar to Triton's but the APIs differ.

Week 10: learn cuTile enough to implement the kernel. There are no good
tutorials yet; read the module docstrings in `cuda.tile` and the examples in
this repo's `/tmp/cutile_smoke.py`.

Week 11: port the kernel. Benchmark against your Triton version.

Week 12: write up "Triton vs. cuTile on the same workload." Honest comparison
— does the compiler make different choices? Are the APIs a productivity win?

**Deliverable:** `kernels/week12/moe_gemm_cutile.py` + comparison doc. This
is genuinely novel content; very few people have published cuTile vs. Triton
comparisons as of this writing.

### Track B — novel target

Pick an operation that neither vLLM nor Triton has a strong kernel for yet.
Candidates drawn from recent architectures:

- **Tree-attention for speculative decoding:** attention with non-contiguous
  causal masks. Used in EAGLE-2 and SpecInfer. Triton implementations exist
  but quality varies.
- **Group-query attention with unusual head ratios:** the standard Triton
  attention kernel is tuned for specific head ratios (e.g., 32:4). Other
  ratios can be slower.
- **Sliding-window + attention-sink:** the Mistral/Gemma pattern. Most
  implementations are derived from FlashAttention; direct Triton versions
  are less tuned.

Weeks 10-12: implement, benchmark, write up.

**Deliverable:** a kernel for a novel target plus benchmarks against any
existing implementations.

## Compressed track (50 hours)

If 100 hours isn't realistic, skip Phase 2 entirely. The weeks become:

- **Weeks 1-3:** Phase 1 as described (fundamentals)
- **Weeks 4-6:** Phase 3 as described (port a real kernel) — but expect to
  struggle more without the profiling foundation
- **Skip Phase 4**

You get one real kernel and less depth. Still worth doing.

## Repo layout

All work lives in this repo:

```
kernels/
  week1/           # modified tutorials
  week2/rmsnorm.py
  week3/silu_mul.py
  week8/moe_gemm_v1.py
  week9/moe_gemm_v2.py
  week12/moe_gemm_cutile.py  (if track A)
notes/
  week4_roofline.md
  week6_flashattn.md
  week7_target.md
  week9_writeup.md
```

Every kernel gets:
- The kernel code itself
- A benchmark script
- Results on L4 as a JSON or CSV
- A short `README.md` in its directory explaining what and why

## Progress tracking

Check off weeks as they're completed. Keep `notes/` updated weekly — writing
while fresh is worth more than trying to reconstruct what you learned later.

- [ ] Week 1 — Triton tutorials
- [ ] Week 2 — RMSNorm
- [ ] Week 3 — SiLU × Mul
- [ ] Week 4 — Roofline analysis of fused_moe_kernel
- [ ] Week 5 — Profile own kernels
- [ ] Week 6 — Read FlashAttention-2
- [ ] Week 7 — Target selection and baseline study
- [ ] Week 8 — MoE GEMM v1
- [ ] Week 9 — MoE GEMM v2 + writeup
- [ ] Week 10 — Phase 4 kickoff (choose Track A or B)
- [ ] Week 11 — Phase 4 middle
- [ ] Week 12 — Phase 4 writeup

## What this path does not cover

Being honest about the scope:

- **Distributed kernels (multi-GPU, tensor parallelism, expert parallelism):**
  not addressed. Requires hardware we don't have on a single g6.4xlarge.
- **Training kernels (backward passes):** not addressed. Inference only.
- **Quantization authoring (writing your own FP8 or INT4 kernels):** not
  addressed directly, though the MoE GEMM work in Phase 3 touches quantized
  paths.
- **CUDA C directly:** not addressed. Triton and cuTile are deliberately the
  abstraction layer we operate at.

These are follow-on paths after this one lands.
