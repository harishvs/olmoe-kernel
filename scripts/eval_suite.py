import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

MODEL = "allenai/OLMoE-1B-7B-0924-Instruct"
TASKS = ["mmlu", "gsm8k", "hellaswag", "arc_challenge"]
SEED = 42
RESULTS_DIR = Path("results/eval")

LM_EVAL = shutil.which("lm_eval") or str(Path(sys.executable).parent / "lm_eval")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

RUNS = [
    ("baseline", ""),
    ("fp8", ",quantization=fp8"),
]

for name, extra_args in RUNS:
    model_args = f"pretrained={MODEL},gpu_memory_utilization=0.85{extra_args}"
    for task in TASKS:
        out_dir = RESULTS_DIR / f"{name}_{timestamp}_{task}"
        out_dir.mkdir()
        result = subprocess.run(
            [
                LM_EVAL, "run",
                "--model", "vllm",
                "--model_args", model_args,
                "--tasks", task,
                "--batch_size", "auto",
                "--seed", str(SEED),
                "--output_path", str(out_dir),
            ],
        )
        status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
        print(f"[{name}] {task}: {status}")
