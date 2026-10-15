"""Check the six adapter archives; write weights only with --extract (no models loaded)."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--slot", help="One slot from manifest.json; default: all six")
    p.add_argument("--extract", action="store_true")
    args = p.parse_args()
    slots = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))["adapters"]
    if args.slot:
        slots = [s for s in slots if s["slot"] == args.slot]
        if not slots:
            raise ValueError("Unknown adapter slot")
    for slot in slots:
        archive = ROOT / slot["archive"]
        if digest(archive) != slot["archive_sha256"]:
            raise ValueError("Archive checksum mismatch")
        with zipfile.ZipFile(archive) as z:
            if set(z.namelist()) != {m["path"] for m in slot["members"]}:
                raise ValueError("Unexpected archive member")
            for entry in slot["members"]:
                h = hashlib.sha256()
                size = 0
                with z.open(entry["path"]) as f:
                    for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
                        size += len(chunk)
                        h.update(chunk)
                if h.hexdigest() != entry["sha256"] or size != entry["bytes"]:
                    raise ValueError("Archive member verification failed")
            if args.extract:
                # All archive members have been checked. Only the recorded weight is written;
                # small configuration/metadata files already accompany each slot directory.
                target = ROOT / slot["weight"]
                target.resolve().relative_to(ROOT.resolve())
                if target.exists():
                    if digest(target) != slot["weight_sha256"]:
                        raise ValueError("Refusing to overwrite a different weight file")
                else:
                    temp = target.with_suffix(".safetensors.part")
                    with z.open(slot["weight"]) as src, temp.open("wb") as dst:
                        shutil.copyfileobj(src, dst, 4 * 1024 * 1024)
                    if digest(temp) != slot["weight_sha256"]:
                        raise ValueError("Extracted weight verification failed")
                    temp.replace(target)
        print(slot["slot"], "EXTRACTED" if args.extract else "VERIFIED")


if __name__ == "__main__":
    main()
