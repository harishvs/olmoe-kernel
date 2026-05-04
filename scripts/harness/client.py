"""Engine-agnostic benchmark client.

Speaks OpenAI-compatible HTTP. Captures per-request TTFT (time to first
streaming chunk) and TPOT (time per subsequent chunk) for any backend
that exposes an OpenAI-compatible /v1/completions or /v1/chat/completions
endpoint.
"""
from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from typing import Literal

import httpx


@dataclass
class Request:
    prompt: str
    max_tokens: int
    label: str = ""  # free-form, for per-request identification


@dataclass
class RequestResult:
    label: str
    ttft_ms: float
    tpot_ms: float
    total_ms: float
    num_output_tokens: int


@dataclass
class BenchResult:
    label: str
    n_iters: int
    ttft_ms: dict[str, float]   # {mean, p50, p90, p99}
    tpot_ms: dict[str, float]
    total_ms: dict[str, float]
    raw: list[RequestResult]


def _percentiles(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0}
    xs_sorted = sorted(xs)
    return {
        "mean": statistics.fmean(xs),
        "p50": statistics.median(xs),
        "p90": xs_sorted[min(int(len(xs_sorted) * 0.9), len(xs_sorted) - 1)],
        "p99": xs_sorted[min(int(len(xs_sorted) * 0.99), len(xs_sorted) - 1)],
    }


def _send_one(
    url: str,
    model: str,
    req: Request,
    temperature: float,
    seed: int | None,
) -> RequestResult:
    """Send a streaming completion request, capture first-chunk and last-chunk times."""
    payload = {
        "model": model,
        "prompt": req.prompt,
        "max_tokens": req.max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    if seed is not None:
        payload["seed"] = seed

    t_submit = time.perf_counter()
    t_first: float | None = None
    t_last = t_submit
    n_tokens = 0

    with httpx.stream("POST", f"{url}/v1/completions", json=payload, timeout=600.0) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            data = line[6:]
            if data.strip() == "[DONE]":
                break
            chunk = json.loads(data)
            text = chunk.get("choices", [{}])[0].get("text", "")
            if not text:
                continue
            now = time.perf_counter()
            if t_first is None:
                t_first = now
            t_last = now
            n_tokens += 1

    if t_first is None:
        raise RuntimeError(f"No tokens received for request '{req.label}'")

    ttft_ms = (t_first - t_submit) * 1000.0
    decode_ms = (t_last - t_first) * 1000.0
    tpot_ms = decode_ms / max(n_tokens - 1, 1)
    total_ms = (t_last - t_submit) * 1000.0
    return RequestResult(
        label=req.label,
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        total_ms=total_ms,
        num_output_tokens=n_tokens,
    )


def run_workload(
    url: str,
    model: str,
    requests: list[Request],
    warmup: int = 3,
    temperature: float = 0.0,
    seed: int | None = 42,
    label: str = "workload",
) -> BenchResult:
    """Run `warmup` throwaway requests, then time `requests` in order."""
    for i in range(warmup):
        _send_one(url, model, requests[0], temperature, seed)

    results = [_send_one(url, model, req, temperature, seed) for req in requests]

    return BenchResult(
        label=label,
        n_iters=len(results),
        ttft_ms=_percentiles([r.ttft_ms for r in results]),
        tpot_ms=_percentiles([r.tpot_ms for r in results]),
        total_ms=_percentiles([r.total_ms for r in results]),
        raw=results,
    )


@dataclass
class ConcurrentResult:
    """Aggregate metrics for a steady-state concurrent run."""
    label: str
    concurrency: int
    n_completed: int
    duration_s: float
    requests_per_s: float
    output_tokens_per_s: float
    ttft_ms: dict[str, float]
    tpot_ms: dict[str, float]
    total_ms: dict[str, float]
    raw: list[RequestResult]


def run_concurrent_workload(
    url: str,
    model: str,
    request_pool: list[Request],
    concurrency: int,
    duration_s: float,
    warmup_s: float = 5.0,
    temperature: float = 0.0,
    seed: int | None = 42,
    label: str = "concurrent",
) -> ConcurrentResult:
    """Maintain `concurrency` in-flight requests for `duration_s` seconds.

    Each worker loops: pop a request from the pool (round-robin), send it,
    record the result, repeat. Results collected during the first `warmup_s`
    seconds are discarded; the measurement window is the rest.
    """
    import threading

    n_workers = concurrency
    results_lock = threading.Lock()
    all_results: list[tuple[float, RequestResult]] = []  # (completion_time, result)
    stop = threading.Event()
    pool_idx = [0]
    pool_lock = threading.Lock()

    def worker():
        while not stop.is_set():
            with pool_lock:
                req = request_pool[pool_idx[0] % len(request_pool)]
                pool_idx[0] += 1
            try:
                r = _send_one(url, model, req, temperature, seed)
            except Exception as e:
                print(f"[harness.client] worker error: {e}")
                continue
            t_complete = time.perf_counter()
            with results_lock:
                all_results.append((t_complete, r))

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(n_workers)]
    for t in threads:
        t.start()

    t_start = time.perf_counter()
    t_measure_start = t_start + warmup_s
    t_end = t_measure_start + duration_s

    while time.perf_counter() < t_end:
        time.sleep(0.1)

    stop.set()
    for t in threads:
        t.join(timeout=60)

    # Filter to results completed within the measurement window.
    measured = [r for t_c, r in all_results if t_measure_start <= t_c <= t_end]
    measured_window = max(
        (t_c for t_c, r in all_results if t_measure_start <= t_c <= t_end),
        default=t_end,
    ) - t_measure_start

    if not measured:
        raise RuntimeError(f"No requests completed in measurement window for {label}")

    total_output_tokens = sum(r.num_output_tokens for r in measured)

    return ConcurrentResult(
        label=label,
        concurrency=concurrency,
        n_completed=len(measured),
        duration_s=measured_window,
        requests_per_s=len(measured) / measured_window,
        output_tokens_per_s=total_output_tokens / measured_window,
        ttft_ms=_percentiles([r.ttft_ms for r in measured]),
        tpot_ms=_percentiles([r.tpot_ms for r in measured]),
        total_ms=_percentiles([r.total_ms for r in measured]),
        raw=measured,
    )
