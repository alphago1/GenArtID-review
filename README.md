# GenArtID review code

Minimal core code for **Rethinking Legal Retrieval as Generative Article Identifier Prediction**.

GenArtID predicts statutory article identifiers from a question, parses the original Top-5 window, and binds accepted identifiers to authoritative corpus text. Corpus-only knowledge injection uses identifier-to-text and text-to-identifier samples; cases and gold labels are evaluation inputs only.

## Contents and boundaries

- `review_cli.py`: corpus-only SFT preparation, grouped split construction, strict Recall@5 evaluation, ID-to-corpus binding, opt-in API retrieval/recitation, and CPU memory scoring.
- `ga_*.py`: validated parsing, identifiers, metrics, corpus, SFT, selection, training/generation utilities, and frozen full-text memory encoder logic. The `ga_` prefix supports a small flat layout without shadowing Python standard-library modules.
- `f_train_sft.py`, `g_generate_predictions.py`, `h_dev_select.py`, `run_test_eval.py`: training, generation, dev-only checkpoint selection, and gated matched Base/FT test evaluation. Only package import/root paths were adapted for this layout.
- `base_ft_eval_profile.json`, `research_protocol.json`, `ablation_matrix.json`: explicit generation and training settings. Training/inference guards default to disabled; enable them deliberately for a reproduction run after preparing local models and data.
- `scorer_identity.json`: the pinned Chinese encoder revision and individual file hashes. Weights are not included.
- `fig3_points.json`, `plot_fig3.py`: the ten displayed, measured Fig.3 points and a standalone plotting entry.
- `test_review_core.py`: CPU fixtures, clearly separate from experimental data.

The released outputs for Tables 1/2/3 and Fig.3/Table 7, including original generated statute text, are mapped in [results/README.md](results/README.md). Available adapter archives and loading instructions are in [checkpoints/retrained/](checkpoints/retrained/README.md). See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for coverage. Dataset and model terms remain separate: [LexRAG](https://github.com/CSHaitao/LexRAG), [LawBench](https://github.com/open-compass/LawBench).

## Quick CPU check

Requires Python 3.10+. The core CPU preparation/tests use the standard library.

```bash
python -m unittest discover -s . -p test_review_core.py -v
python review_cli.py --help
python audit_publication.py
```

Plot the published coordinates without models, API calls or evaluation data:

```bash
pip install -r requirements.txt
python plot_fig3.py
```

## Local data formats

Provide these data separately; do not commit API keys or restricted benchmark data.

1. Article corpus: JSON list of objects with `name` (a full Chinese statutory identifier), `content` (verbatim statute text), and optional `id`. The Civil Code/Criminal Law loaders reject duplicate or malformed IDs. Multi-law binding also accepts JSONL.
2. Generation questions: JSON list with `id` and `question`. Gold is excluded from generation payloads.
3. Evaluation gold: JSONL objects with `item_id` and `gold` (a list of statutory IDs).
4. Predictions: JSONL objects with `item_id` and `raw_response`; API output records additionally retain the returned response and payload hash.
5. Split construction questions: JSON list with `id`, `question`, and `articles` (gold IDs). Split/selection files are local evaluation artifacts, never SFT training inputs.

The paper's fixed Civil/Criminal diagnostic test sets contain 180/44 questions. A newly constructed split is not identical to the paper's frozen membership by default. Use the corresponding original membership and source hashes for an exact replay. All reported diagnostics are retrospective; this package does not relabel historical tests as blind.

## Corpus-only knowledge injection

```bash
python review_cli.py prepare-sft --domain civil --corpus inputs/data/civil_code_articles.json
python review_cli.py prepare-sft --domain criminal_law --corpus inputs/data/criminal_law_articles.json
```

Each ratio has four slots per article. The Civil Code/Criminal Law corpora yield 5,040/2,020 slots per ratio. The 1:1 setting is available as `ratio_1f_1r.json`; all five ratios are preparation/ablation options, not a claim that all were trained. Slot ratios are not supervised-token or gradient-share measurements.

With dependencies and local model/tokenizer files supplied, audit token boundaries before training:

```bash
pip install -r requirements-model.txt
python f_train_sft.py --domain civil --ratio 1f_1r --seed 20260910 --target-modules qvo --tokenizer /local/GLM-4-9B-0414 --dry-run
```

The historical paper recipe uses QLoRA rank 32, alpha 64 and q_proj/v_proj/o_proj targets. Do not treat this CPU check as a GPU training result. Checkpoint choice is by development Recall@5, ties to the earliest step; matched Base and selected FT are then evaluated on the same fixed test membership. Generating every checkpoint's test curve for selection is not permitted by the protocol.

## Retrieval, binding and evaluation

API generation is preview-only by default. For real requests, set `GENARTID_API_KEY` in the local environment and add `--execute`; provision/credit management is deliberately outside this package. A real run may incur provider costs. Fix the model/provider and review the job count first.

```bash
python review_cli.py generate-api --task retrieval --domain civil --input questions.json --model cohere/command-r7b-12-2024 --provider cohere --out predictions.jsonl
python review_cli.py evaluate --predictions predictions.jsonl --gold gold.jsonl --out metrics.json
python review_cli.py bind --corpus law_library.jsonl --answer '{"articles":["《民法典》第243条"]}'
```

The frozen `fixed_v2` parser caps the original five slots before deduplication. Invalid/duplicate entries consume slots; candidates are never repaired or supplemented using gold. P@5 divides by 5. Formal macro Recall is refused for incomplete/extra prediction coverage. The legacy parser remains available for historical parity, not the default score.

## Statute memory diagnostic

```bash
python review_cli.py generate-api --task recitation --domain civil --input inputs/data/civil_code_articles.json --model cohere/command-r7b-12-2024 --provider cohere --out recitations.jsonl
python review_cli.py memory --domain civil --corpus inputs/data/civil_code_articles.json --predictions recitations.jsonl --encoder /local/text2vec-base-chinese --out memory.json
```

Recitation receives only the statute name and the shared closed-book prompt. Temperature is 0; retrieval/recitation caps are 256/2,048 tokens. Empty/truncated outputs are retained, and answers are not retried for quality. The encoder is `shibing624/text2vec-base-chinese`, revision `183bb99aa7af74355fb58d16edf8c13ae7c5433e`; its file hashes are checked before local CPU loading. Full text is split into non-overlapping 126-body-token windows plus two special tokens, with tail retention, attention-mask mean pooling, token-count weighting and final L2 normalization.

Classification is target-first: target cosine >=0.85 is Eff.; otherwise another article in the whole same-domain pool >=0.85 is Dup.; otherwise Err. M=Eff/N. Exact string agreement is a different metric and is not substituted for M.

## Measured Command addition to Fig.3

| Domain | Articles | Eff. / Dup. / Err. | M | Recall@5 |
|---|---:|---|---:|---:|
| Civil Code | 1,260 | 9 / 749 / 502 | 0.007142857 | 0.138888889 |
| Criminal Law | 505 | 26 / 312 / 167 | 0.051485149 | 0.462121212 |

These are one-run OpenRouter/Cohere measurements; server weight revision and precision are undisclosed. GLM SSL points are paired three-seed means. The displayed model subset is not an unbiased sample for estimating a population correlation, and the diagnostic does not establish a causal relationship between memory and retrieval.

## License

Core code is MIT-licensed. Dataset, model and encoder licenses are separate. `provenance.json` records source-code hashes and the narrow layout adaptations; it does not contain raw cases or credentials.
