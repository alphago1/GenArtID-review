"""Replay one recorded initialization at its fixed training step; explicit --run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
ADAPTERS = ROOT.parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slot", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    slots = json.loads((ADAPTERS / "manifest.json").read_text(encoding="utf-8"))["adapters"]
    slot = next(s for s in slots if s["slot"] == args.slot)
    metadata = json.loads((ADAPTERS / args.slot / "metadata.json").read_text(encoding="utf-8"))
    for entry in json.loads((ROOT / "data_manifest.json").read_text(encoding="utf-8"))["training"]:
        if entry["domain"] == metadata["domain"]:
            dataset = ROOT / "artifacts/sft_corpus_only" / metadata["domain"] / "ratio_1f_1r.json"
            if hashlib.sha256(dataset.read_bytes()).hexdigest() != entry["dataset_file_sha256"]:
                raise ValueError("Prepare the frozen corpus-only dataset first")
    script = ROOT / "scripts" / metadata["training_entry"]
    if hashlib.sha256(script.read_bytes()).hexdigest() != metadata["training_entry_sha256"]:
        raise ValueError("Training entry hash mismatch")
    cmd = [sys.executable, str(script), "--domain", metadata["domain"], "--ratio", "1f_1r",
           "--seed", str(metadata["training_seed"]), "--model", str(args.model.resolve()),
           "--tokenizer", str(args.model.resolve()), "--profile",
           str(ROOT / "configs/glm0414/base_ft_eval_profile.json"), "--target-modules", "qvo",
           "--max-length", "1024", "--stop-after-step", str(metadata["target_step"]),
           "--save-total-limit", "1", "--out", str(args.out.resolve())]
    if metadata["training_entry"] == "train_recovery.py":
        cmd += ["--initialization-seed", str(metadata["initialization_seed"])]
    print(json.dumps({"command": cmd, "executed": args.run}, ensure_ascii=False), flush=True)
    if args.run:
        # Same explicit process environment as the recorded training runs.
        env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8", TOKENIZERS_PARALLELISM="false",
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="4",
                   MKL_NUM_THREADS="4", PYTHONHASHSEED=str(metadata["training_seed"]))
        subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
