# /// script
# dependencies = [
#   "modal",
#   "huggingface-hub",
# ]
# ///

"""
Modal (https://modal.com) script to run the evaluation on the cloud.

Run `modal run scripts/run_modal.py --models <model1> <model2> ... --config <config.yaml>`
"""

from datetime import datetime
from pathlib import Path
from typing import List

import modal
from modal import FilePatternMatcher

app = modal.App("encoder-fast-eval")

image = (
    modal.Image.debian_slim()
    .run_commands("pip install --upgrade uv")
    .uv_sync()
    .env({"TQDM_MININTERVAL": "5"})
    .add_local_dir(
        ".",
        remote_path="/root",
        ignore=FilePatternMatcher.from_file("./.gitignore"),
    )
)

models_volume = modal.Volume.from_name(
    "encoder-fast-eval-models", create_if_missing=True
)
results_volume = modal.Volume.from_name(
    "encoder-fast-eval-results", create_if_missing=True
)

RESULTS_PATH = Path("/results")
MODELS_PATH = Path("/models")


@app.function(
    image=image,
    volumes={MODELS_PATH: models_volume},
    timeout=60,
)
def check_if_model_in_volume(model_dir: str) -> bool:
    from pathlib import Path

    model_path = Path(model_dir)
    return (
        model_path.exists() and model_path.is_dir() and bool(list(model_path.iterdir()))
    )


@app.function(
    image=image,
    gpu="a100",
    timeout=86400,
    volumes={RESULTS_PATH: results_volume, MODELS_PATH: models_volume},
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def run_evaluation(model_paths: List[str], config_path: str, results_path: str):
    """Run evaluation for all models in parallel."""
    import subprocess
    from pathlib import Path
    from queue import Queue
    from threading import Thread

    def run_model_eval(model_path, output_queue):
        print(f"Starting evaluation for model: {model_path}")
        cmd = [
            "python",
            "encoder_eval.py",
            config_path,
            "--model",
            model_path,
            "--output-dir",
            results_path,
        ]

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd="/root",
            bufsize=1,
            universal_newlines=True,
        )

        for line in proc.stdout:
            print(f"[{model_path}] {line}", end="")

        proc.wait()
        output_queue.put((model_path, proc.returncode))

    output_queue = Queue()
    threads = []

    for model_path in model_paths:
        thread = Thread(target=run_model_eval, args=(model_path, output_queue))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    while not output_queue.empty():
        model_path, returncode = output_queue.get()
        if returncode != 0:
            print(f"Evaluation failed for {model_path} with exit code {returncode}")
        else:
            print(f"Evaluation completed for {model_path}")

    formats = ["console", "excel", "markdown", "csv", "json"]
    output_dir = Path(results_path)

    for fmt in formats:
        if fmt == "excel":
            output_file = output_dir / "results.xlsx"
            cmd = [
                "python",
                "scripts/analyze_results.py",
                fmt,
                "--results-dir",
                results_path,
                "-o",
                str(output_file),
            ]
        else:
            output_file = output_dir / f"results.{fmt}"
            cmd = [
                "python",
                "scripts/analyze_results.py",
                fmt,
                "--results-dir",
                results_path,
            ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd="/root")

        if fmt not in ["console", "excel"]:
            with open(output_file, "w") as f:
                f.write(result.stdout)

    print(f"Results saved in {results_path}")


@app.local_entrypoint()
def main(*args):
    import argparse

    from huggingface_hub import model_info

    parser = argparse.ArgumentParser(description="Run encoder evaluation on Modal")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="List of model names or paths (space-separated)",
    )
    parser.add_argument("--config", required=True, help="Path to config YAML file")

    parsed_args = parser.parse_args(args)

    model_paths = []
    local_model_paths = []
    for model_path in parsed_args.models:
        try:
            _ = model_info(model_path)
            print(f"Model {model_path} found on Hugging Face Hub.")
            model_paths.append(model_path)
            continue
        except Exception as e:
            print(f"Model {model_path} not found on Hugging Face Hub: {e}")

        local_model_path = Path(model_path)
        if not local_model_path.exists():
            raise ValueError(f"Model path {model_path} does not exist locally.")

        local_model_paths.append(local_model_path)
        model_paths.append(MODELS_PATH / local_model_path.name)

    for local_model_path in local_model_paths:
        remote_model_path = MODELS_PATH / local_model_path.name
        is_model_uploaded = check_if_model_in_volume.remote(remote_model_path)

        if is_model_uploaded:
            print(f"Model {local_model_path.name} is already uploaded.")
            continue

        print(f"Uploading model {local_model_path.name} to Modal...")

        with models_volume.batch_upload() as upload:
            upload.put_directory(str(local_model_path), str(local_model_path.name))

        print(f"Model {local_model_path.name} uploaded successfully.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = RESULTS_PATH / timestamp

    with results_volume.batch_upload() as upload:
        upload.put_file(parsed_args.config, f"{timestamp}/config.yaml")

    run_evaluation.remote(
        model_paths=model_paths,
        config_path=str(RESULTS_PATH / timestamp / "config.yaml"),
        results_path=str(results_path),
    )
