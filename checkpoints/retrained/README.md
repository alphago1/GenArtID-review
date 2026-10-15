# Adapter files

Six adapter archives are listed in `manifest.json`. Each ZIP contains one complete PEFT adapter folder; accompanying metadata records the base revision, training/data/initialization seeds, fixed step and file hashes. See `NOTICE.txt` for provenance. Historical paper outputs are in `results/`.

From the repository root:

```sh
# CPU-only checks; no extraction, generation or training.
python checkpoints/retrained/verify_release.py

# Unpack the supplied adapters for inference; no model is loaded by this command.
python checkpoints/retrained/unpack_adapters.py --extract
```

Use the exact base snapshot and tokenizer in [reproduction/README.md](reproduction/README.md). Evaluation/loading instructions and optional fixed-step training commands are in [reproduction/replay/README.md](reproduction/replay/README.md). Training is not required to audit the released paper results. Base/scorer weights and full installed environments are obtained separately using the pinned revisions, SHA256 files and version lock. Preserve upstream model licensing.
