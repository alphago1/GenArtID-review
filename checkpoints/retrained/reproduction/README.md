# Dependencies

Model versions, checksums, and environment setup for the adapter files.

## Frozen identities

| Component | Official repository | Immutable revision |
| --- | --- | --- |
| GLM base model and tokenizer | [zai-org/GLM-4-9B-0414](https://huggingface.co/zai-org/GLM-4-9B-0414) | `645b8482494e31b6b752272bf7f7f273ef0f3caf` |
| Text2vec memory scorer | [shibing624/text2vec-base-chinese](https://huggingface.co/shibing624/text2vec-base-chinese) | `183bb99aa7af74355fb58d16edf8c13ae7c5433e` |

`model_assets.json` lists all 24 required files, exact byte sizes, and SHA256 digests, including tokenizer/configuration files. GLM files total about 18.8 GB; the scorer adds about 0.41 GB. Download the scorer only when running the paper's memory scoring, not for Recall-only evaluation. Use the full commit hashes rather than a mutable `main` branch. Preserve the original licenses when using or redistributing upstream artifacts.

## Recreate the software environment

Use Linux x86_64 and Python **3.12.15**. From the repository root, create an isolated environment and install the recorded CUDA build and 74 pinned distributions:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r checkpoints/retrained/reproduction/environment.lock.txt --index-url https://pypi.org/simple
python -m pip check
python -c "import sys,torch; print(sys.version); print(torch.__version__,torch.version.cuda)"
```

Check that Python is 3.12.15 and PyTorch reports `2.8.0+cu128` with CUDA build `12.8`. GPU, driver, and core package versions are recorded in `environment.json`. The lock specifies package versions; the host supplies the NVIDIA driver separately.

PyTorch's official [previous-version installation guide](https://pytorch.org/get-started/previous-versions/) documents the CUDA 12.8 build of 2.8.0.

## Download and verify

The helper downloads only named files from the official Hugging Face endpoint at the frozen revisions and checks every file's size and SHA256. It does not instantiate a model or access saved account credentials.

```sh
# Metadata validation only: no network access and no weight downloads.
python checkpoints/retrained/reproduction/acquire_assets.py --asset all --check-manifest

# Explicit acquisition of the base model and tokenizer (about 18.8 GB).
python checkpoints/retrained/reproduction/acquire_assets.py --asset glm --download --output .

# Optional: needed for memory scoring only (about 0.41 GB).
python checkpoints/retrained/reproduction/acquire_assets.py --asset text2vec --download --output .

# Independently verify a completed download without network access.
python checkpoints/retrained/reproduction/acquire_assets.py --asset glm --verify --output .
```

The directories are `models/GLM-4-9B-0414/` and `models/text2vec-base-chinese/`. The official Hub supports [downloads at a specified commit revision](https://huggingface.co/docs/huggingface_hub/en/guides/download). A checksum mismatch fails verification; do not silently substitute the latest model/tokenizer.

Use the matching base model, adapter, and recorded evaluation configuration. Adapter metadata records the weight SHA256, seeds, target step, source hashes, and decoding/parser settings.
