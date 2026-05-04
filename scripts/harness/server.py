"""Engine-agnostic server launcher.

Supports vLLM today, TRT-LLM stubbed. Each engine adapter knows how to
launch its own server process and expose an OpenAI-compatible endpoint.
"""
from __future__ import annotations

import contextlib
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx


@dataclass
class ServerConfig:
    engine: str  # "vllm" | "trtllm"
    model: str
    port: int = 8000
    extra_args: list[str] = field(default_factory=list)
    startup_timeout_s: int = 600


def _vllm_cmd(cfg: ServerConfig) -> list[str]:
    vllm = shutil.which("vllm") or str(Path(sys.executable).parent / "vllm")
    return [
        vllm, "serve", cfg.model,
        "--port", str(cfg.port),
        *cfg.extra_args,
    ]


def _trtllm_cmd(cfg: ServerConfig) -> list[str]:
    raise NotImplementedError(
        "TRT-LLM server launcher is not implemented yet. "
        "We will add it when we actually benchmark that engine."
    )


_CMD_BY_ENGINE = {
    "vllm": _vllm_cmd,
    "trtllm": _trtllm_cmd,
}


def _wait_for_healthy(url: str, timeout_s: int) -> None:
    deadline = time.monotonic() + timeout_s
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{url}/v1/models", timeout=2.0)
            if r.status_code == 200:
                return
        except Exception as e:
            last_err = e
        time.sleep(1.0)
    raise TimeoutError(f"Server at {url} not healthy after {timeout_s}s (last err: {last_err})")


@contextlib.contextmanager
def serve(cfg: ServerConfig, log_path: Path | None = None):
    """Context manager: launches the server, yields its base URL, tears it down on exit.

    Server stdout/stderr go to `log_path` if provided, otherwise to /tmp with a
    predictable name so a failed startup can be inspected.
    """
    cmd = _CMD_BY_ENGINE[cfg.engine](cfg)
    url = f"http://127.0.0.1:{cfg.port}"

    if log_path is None:
        log_path = Path("/tmp") / f"harness_server_{cfg.engine}_{cfg.port}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[harness.server] launching: {' '.join(cmd)}")
    print(f"[harness.server] logs -> {log_path}")
    log_fh = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
    try:
        try:
            _wait_for_healthy(url, cfg.startup_timeout_s)
        except TimeoutError:
            print(f"[harness.server] startup failed. Last 40 lines of {log_path}:")
            log_fh.flush()
            try:
                with open(log_path) as f:
                    tail = f.readlines()[-40:]
                print("".join(tail))
            except Exception:
                pass
            raise
        print(f"[harness.server] ready at {url}")
        yield url
    finally:
        print(f"[harness.server] stopping (pid={proc.pid})")
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        log_fh.close()
