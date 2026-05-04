# TODO: OLMoE TTFT Baseline & Custom Numba Kernel

## Phase 1: Baseline TTFT

- [x] Step 1: Environment setup (uv venv, install vllm + numba + deps, verify GPU)
- [x] Step 2: Smoke test (load OLMoE via vLLM, generate 1 token end-to-end)
- [ ] Step 3: Benchmark TTFT (scripts/benchmark_ttft.py — sweep input lengths, record to CSV)
- [ ] Step 4: Profile (scripts/profile_torch.py — torch.profiler + nsys, identify top kernels)
- [ ] Step 5: Record baseline (document top-10 kernels, routing overhead %, target kernel)

## Phase 2: Custom Numba Kernel

- [ ] Step 1: Confirm target operation from profiling data
- [ ] Step 2: Write kernel (kernels/fused_topk_softmax.py — fused softmax + top-K)
- [ ] Step 3: Test correctness (tests/test_fused_topk.py — compare vs torch reference)
- [ ] Step 4: Integrate (scripts/inference_custom.py — patch HF OlmoeTopKRouter)
- [ ] Step 5: Compare (re-run benchmark, compute delta, generate comparison)
