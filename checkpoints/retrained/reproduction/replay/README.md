Run `python checkpoints/retrained/unpack_adapters.py --extract` before loading an adapter. For CPU reconstruction of the training data, the matching statute inputs are already in `results/fig3_table7/reference/`. Retraining is optional.

# Recorded runs

Commands below are run from the repository root, after installing the environment and verifying the pinned base assets as described one directory above. Set `SLOT` to any entry in `checkpoints/retrained/manifest.json`.

```sh
SLOT=civil_seed20260911_step1575
BASE=models/GLM-4-9B-0414

# Check inputs and adapter identity on CPU; no generation without --run.
python checkpoints/retrained/reproduction/replay/evaluate_adapter.py --slot "$SLOT" --model "$BASE" --questions /path/to/eval_civil_code_questions.json --out runs/recall-check

# Generate with the saved adapter (GPU).
python checkpoints/retrained/reproduction/replay/evaluate_adapter.py --slot "$SLOT" --model "$BASE" --questions /path/to/eval_civil_code_questions.json --out runs/recall-check --run

# CPU: rebuild both corpus-only training datasets from matching normalized inputs.
python checkpoints/retrained/reproduction/replay/prepare_data.py --inputs /path/to/normalized/data

# Print the recorded fixed-step training command; add --run for GPU training.
python checkpoints/retrained/reproduction/replay/train_adapter.py --slot "$SLOT" --model "$BASE" --out runs/retrained-copy
```

For Criminal Law, use `eval_criminal_law_questions.json`. To evaluate a newly trained copy, pass `--adapter runs/retrained-copy/final_adapter` and a fresh output directory. Acquire and verify the base/tokenizer with `acquire_assets.py --asset glm --verify --output .` before GPU execution. Load the base in NF4 double quantization/bf16 with SDPA, then attach the adapter with PEFT; use the base snapshot's tokenizer, without merging adapter weights. The evaluation entry implements this loading sequence.

`data_manifest.json` identifies exact normalized input bytes, sample counts and reconstructed dataset hashes. It also links the upstream data sources. Upstream case/corpus files are not redistributed here; a different normalization or corpus snapshot must not be silently accepted as the recorded input. Training reconstruction uses only statute files. Questions/gold enter evaluation only. JSON-file hashes and canonical JSON hashes are separate fields because whitespace can change the former.

The two training entries and `src/genartid/` are copied byte-for-byte from the executed runs. Each adapter records which entry and initialization seed it used. Target steps stay fixed; the full cosine schedule remains 1,575 civil / 635 criminal steps, with 48 / 20 warmup steps, respectively. The recorded optimizer is `adamw_torch_fused`. No checkpoint search or recitation is performed by these commands.

Recall uses fixed test IDs, the official template, greedy generation with batch size 1 and maximum 256 new tokens, EOS [151329,151336,151338], pad 151329, and `fixed_v2`. Duplicate/invalid items consume their original Top-5 positions; no refill is allowed. Fixed P@5 divides by 5. Released ID-level results belong to the retrained weights; historical paper results remain in `results/`.

These replacement runs were checked against previously observed test Recall, as recorded in the per-adapter metadata. They are retrospective artifacts. The supplied weights allow inference replay without repeating that search. CPU file/data/score checks are provided; a fresh machine's GPU replay is a separate execution step.
