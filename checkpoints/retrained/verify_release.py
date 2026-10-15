"""Audit retrained adapter archives and ID-level outputs without loading models."""
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import struct
import zipfile

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    files = read(ROOT / "files.json")["files"]
    for f in files:
        path = ROOT / f["path"]
        if path.stat().st_size != f["bytes"] or digest(path) != f["sha256"]:
            raise ValueError("File verification failed: " + f["path"])
    manifest = read(ROOT / "manifest.json")
    total = 0
    for slot in manifest["adapters"]:
        folder = ROOT / slot["slot"]
        meta, config = read(folder / "metadata.json"), read(folder / "adapter_config.json")
        if config["base_model_name_or_path"] != manifest["base_model"]["repo_id"] or config["revision"] != manifest["base_model"]["revision"]:
            raise ValueError("Wrong base snapshot")
        if (config["r"], config["lora_alpha"], set(config["target_modules"])) != (32, 64, {"q_proj", "v_proj", "o_proj"}):
            raise ValueError("Wrong LoRA configuration")
        with zipfile.ZipFile(ROOT / slot["archive"]) as z:
            if set(z.namelist()) != {m["path"] for m in slot["members"]}:
                raise ValueError("Unexpected archive member")
            for member in slot["members"]:
                h = hashlib.sha256()
                with z.open(member["path"]) as stream:
                    if member["path"] == slot["weight"]:
                        length_bytes = stream.read(8)
                        length = struct.unpack("<Q", length_bytes)[0]
                        if not 0 < length < 1024 * 1024:
                            raise ValueError("Invalid tensor header")
                        header_bytes = stream.read(length)
                        header = json.loads(header_bytes)
                        tensors = [v for k, v in header.items() if k != "__metadata__"]
                        if len(tensors) != 240:
                            raise ValueError("Wrong tensor count")
                        offsets = sorted(t["data_offsets"] for t in tensors)
                        if offsets[0][0] != 0 or offsets[-1][1] != member["bytes"] - 8 - length or any(a[1] != b[0] for a, b in zip(offsets, offsets[1:])):
                            raise ValueError("Incomplete tensor data")
                        for t in tensors:
                            width = {"F32": 4, "BF16": 2, "F16": 2}[t["dtype"]]
                            if 32 not in t["shape"] or math.prod(t["shape"]) * width != t["data_offsets"][1] - t["data_offsets"][0]:
                                raise ValueError("Wrong tensor shape/size")
                        h.update(length_bytes + header_bytes)
                    for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                        h.update(chunk)
                if h.hexdigest() != member["sha256"] or z.getinfo(member["path"]).file_size != member["bytes"]:
                    raise ValueError("Archive member SHA/size differs")
                if member["path"] != slot["weight"] and digest(ROOT / member["path"]) != member["sha256"]:
                    raise ValueError("Archive metadata differs from accompanying file")
        if meta["weight_sha256"] != slot["weight_sha256"]:
            raise ValueError("Adapter identity mismatch")
        rows = read(folder / "per_question.json")
        ids = read(ROOT / "reproduction/replay/splits" / (meta["domain"] + ".json"))["test_ids"]
        if [r["item_id"] for r in rows] != ids:
            raise ValueError("Question membership/order differs")
        for r in rows:
            gold, pred = set(r["gold_ids"]), r["predicted_ids"]
            hits = len(gold & set(pred))
            if len(pred) > 5 or Fraction(r["R@5_exact"]) != Fraction(hits, len(gold)) or abs(r["P@5_fixed"] - hits / 5) > 1e-15:
                raise ValueError("Invalid ID-level score")
        total += len(rows)
    print(json.dumps({"status": "PASS", "adapters": len(manifest["adapters"]),
                      "id_level_outputs": total, "model_weights_loaded": False}))


if __name__ == "__main__":
    main()
