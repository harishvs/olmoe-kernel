# Requirements

## User Stories

### US-001: Baseline TTFT Measurement
**As a** ML engineer, **I want** to measure the time-to-first-token for OLMoE on my L4 GPU, **so that** I have a reproducible baseline to optimize against.

#### Acceptance Criteria
- [ ] Given vLLM with OLMoE-1B-7B loaded, when I run the benchmark with batch_size=1, then I get TTFT measurements at input lengths 32, 128, 512, 1024
- [ ] Given the benchmark results, when I inspect the output, then I see p50, p90, p99, and mean TTFT in milliseconds
- [ ] Given repeated runs, when I compare results, then variance is low (consistent measurements)

### US-002: Profile TTFT Bottleneck
**As a** ML engineer, **I want** to profile the prefill path at the CUDA kernel level, **so that** I know which operation to target with a custom kernel.

#### Acceptance Criteria
- [ ] Given a profiling run, when I inspect the trace, then I can see the top-10 CUDA kernels by time
- [ ] Given the profiling results, when I analyze them, then I can identify the MoE routing overhead as a percentage of total TTFT

### US-003: Custom Numba Kernel for MoE Routing
**As a** ML engineer, **I want** a fused Numba CUDA kernel for softmax + top-K selection, **so that** the MoE routing step is faster.

#### Acceptance Criteria
- [ ] Given the custom kernel, when I compare its output to torch.softmax + torch.topk, then results match within float32 tolerance
- [ ] Given the custom kernel integrated into inference, when I re-run the TTFT benchmark, then TTFT is reduced (or I understand why not)
- [ ] Given edge cases (uniform logits, single dominant expert), when I run the kernel, then it produces correct results

## Non-Functional Requirements

- All Python work uses a virtual environment managed by `uv`
- Measurements must be reproducible
- Kernel must work on NVIDIA L4 (compute capability 8.9)
