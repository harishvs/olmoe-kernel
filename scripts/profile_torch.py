import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

MODEL = "allenai/OLMoE-1B-7B-0924-Instruct"
INPUT_LEN = 1024
SEED = 42
RESULTS_DIR = Path("results")
PROFILE_DIR = RESULTS_DIR / "profile"

VLLM = shutil.which("vllm") or str(Path(sys.executable).parent / "vllm")

PROFILE_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
run_dir = PROFILE_DIR / f"baseline_{timestamp}_L{INPUT_LEN}_seed{SEED}"
run_dir.mkdir()

subprocess.run(
    [
        VLLM, "bench", "latency",
        "--model", MODEL,
        "--input-len", str(INPUT_LEN),
        "--output-len", "1",
        "--batch-size", "1",
        "--num-iters-warmup", "3",
        "--seed", str(SEED),
        "--profile",
        "--profiler-config.profiler", "torch",
        "--profiler-config.torch_profiler_dir", str(run_dir),
        "--profiler-config.torch_profiler_with_stack", "false",
        "--profiler-config.torch_profiler_record_shapes", "true",
    ],
    check=True,
)

print(f"\nProfile trace(s) written to: {run_dir}")
