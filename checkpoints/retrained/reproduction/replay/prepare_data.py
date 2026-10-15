"""Rebuild the two frozen corpus-only datasets from externally obtained inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from genartid.build_sft import build_ratio_dataset
from genartid.corpus import load_corpus


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True,
                        help="Directory containing the two normalized *_articles.json files")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/sft_corpus_only")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "data_manifest.json").read_text(encoding="utf-8"))
    pending = []
    for entry in manifest["training"]:
        source = args.inputs / entry["file"]
        if sha(source) != entry["sha256"]:
            raise ValueError("Input hash mismatch: " + entry["file"])
        articles = load_corpus(source, entry["domain"])
        dataset = build_ratio_dataset(entry["domain"], articles, (1, 1))
        payload = json.dumps(dataset, ensure_ascii=False, indent=2).encode("utf-8")
        if hashlib.sha256(payload).hexdigest() != entry["dataset_file_sha256"]:
            raise ValueError("Rebuilt dataset differs from frozen bytes")
        pending.append((entry, payload, dataset["stats"]))
    build = {"schema": "genartid-sft-build-manifest/1", "inputs": {}, "outputs": []}
    for entry, payload, stats in pending:
        path = args.output / entry["domain"] / "ratio_1f_1r.json"
        if path.exists() and path.read_bytes() != payload:
            raise ValueError("Refusing to overwrite a different dataset: " + str(path))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        build["inputs"][entry["file"]] = entry["sha256"]
        build["outputs"].append({
            "domain": entry["domain"], "kind": "ratio", "ratio_f_r": [1, 1],
            "path": "artifacts/sft_corpus_only/" + entry["domain"] + "/ratio_1f_1r.json",
            "sha256": sha(path), "bytes": len(payload),
            "f_slots": stats["f_slots"], "r_slots": stats["r_slots"]})
    (args.output / "build_manifest.json").write_text(
        json.dumps(build, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": "PASS", "domains": len(pending),
                      "samples": [s["total_slots"] for _, _, s in pending],
                      "models_loaded": False}))


if __name__ == "__main__":
    main()
