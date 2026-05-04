import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

MODEL = "allenai/OLMoE-1B-7B-0924-Instruct"
INPUT_LENS = [32, 128, 512, 1024]
OUTPUT_LENS = [1, 128]
SEED = 42
RESULTS_DIR = Path("results")

VLLM = shutil.which("vllm") or str(Path(sys.executable).parent / "vllm")

RESULTS_DIR.mkdir(exist_ok=True)
timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

for input_len in INPUT_LENS:
    for output_len in OUTPUT_LENS:
        output_json = RESULTS_DIR / f"baseline_{timestamp}_L{input_len}_O{output_len}_seed{SEED}.json"
        subprocess.run(
            [
                VLLM, "bench", "latency",
                "--model", MODEL,
                "--input-len", str(input_len),
                "--output-len", str(output_len),
                "--batch-size", "1",
                "--num-iters-warmup", "3",
                "--num-iters", "30",
                "--seed", str(SEED),
                "--output-json", str(output_json),
            ],
            check=True,
        )
