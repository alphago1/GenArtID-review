# GenArtID

Code, evaluation artifacts, and adapter checkpoints for **Rethinking Legal Retrieval as Generative Article Identifier Prediction**.

GenArtID formulates legal retrieval as the prediction of canonical statutory article identifiers. Given a legal question, a language model predicts article IDs, which are then bound to authoritative statute text. Corpus-only knowledge injection teaches the association between identifiers and article content without annotated case–article training pairs.

## Method

The retrieval pipeline has three steps:

1. **Predict identifiers.** Generate statutory article IDs from the legal question.
2. **Parse candidates.** Normalize and validate identifiers within the original Top-5 output window.
3. **Retrieve article text.** Look up accepted identifiers in the statute corpus to return the corresponding provisions.

Knowledge injection uses both identifier-to-text and text-to-identifier examples constructed from the statute corpus. Questions and gold labels are used for evaluation. The implementation covers data preparation, QLoRA fine-tuning, generation, development-set checkpoint selection, retrieval scoring, and statute-memory diagnostics.

## Getting started

The core CLI, data preparation, and release audit use Python 3.10+ and the standard library. Run commands from the repository root.

```sh
# Explore the available commands.
python review_cli.py --help

# Bind a predicted article identifier to the included statute text.
python review_cli.py bind --corpus results/fig3_table7/reference/civil_code_articles.json --answer '{"articles":["《民法典》第243条"]}'

# Check the released files and saved experimental results.
python audit_publication.py
```

Model setup, pinned versions, and environment installation are described in the [dependency guide](checkpoints/retrained/reproduction/README.md). See the [execution guide](checkpoints/retrained/reproduction/replay/README.md) for adapter loading and evaluation commands.

## Repository guide

| Component | Contents |
| --- | --- |
| [review_cli.py](review_cli.py) | Corpus preparation, retrieval evaluation, article-text binding, API generation, and memory scoring. |
| Core modules (`ga_*.py`) | Article identifiers, parsing, corpus handling, SFT construction, metrics, and memory-scoring utilities. |
| [Training](f_train_sft.py), [generation](g_generate_predictions.py), [selection](h_dev_select.py), and [evaluation](run_test_eval.py) | Experiment execution entries, with settings in [research_protocol.json](research_protocol.json) and [base_ft_eval_profile.json](base_ft_eval_profile.json). |
| [training/](training/), [splits/](splits/), and [selection/](selection/) | Recorded configurations, fixed diagnostic split membership, and historical development-set selection records. |
| [results/](results/README.md) | Experimental outputs, generated statute text, per-question predictions, per-article scores, and reported table summaries. |
| [checkpoints/](checkpoints/manifest.json) | Checkpoint configurations, downloadable adapter archives, metadata, and dependency records. |

## Adapter checkpoints

Six PEFT adapter archives are provided for **GLM-4-9B-0414**, covering three seeds in each legal domain. Each archive contains weights, configuration, and run metadata; individual ZIP files are approximately 42 MB.

| Seed | Civil Code | Criminal Law |
| --- | --- | --- |
| 20260910 | [Step 600](checkpoints/retrained/civil_seed20260910_step600.zip) | [Step 40](checkpoints/retrained/criminal_law_seed20260910_step40.zip) |
| 20260911 | [Step 1575](checkpoints/retrained/civil_seed20260911_step1575.zip) | [Step 220](checkpoints/retrained/criminal_law_seed20260911_step220.zip) |
| 20260912 | [Step 520](checkpoints/retrained/civil_seed20260912_step520.zip) | [Step 40](checkpoints/retrained/criminal_law_seed20260912_step40.zip) |

```sh
python checkpoints/retrained/verify_release.py
python checkpoints/retrained/unpack_adapters.py --extract
```

Use the matching base model and tokenizer specified in the [dependency guide](checkpoints/retrained/reproduction/README.md). Checksums, metadata, and [provenance](checkpoints/retrained/NOTICE.txt) are provided with the [adapter files](checkpoints/retrained/README.md).

## Experimental data and evaluation

The release includes retrieval outputs on LexRAG and LawBench, knowledge-injection results across three seeds, identifier/content ablation outputs, and paired statute-memory and retrieval records. Generated statute text and matching reference articles are included alongside the scored records. The [data guide](results/README.md) maps these files to the paper's experiments and describes the available records; [table exports](results/paper_tables/) preserve the reported summaries.

Evaluation follows each experiment's recorded protocol. The main retrieval comparison uses Hit@K and MRR; the knowledge-injection and memory diagnostics use macro Recall@5. The `fixed_v2` parser retains the original five slots, including invalid and duplicate entries, and P@5 divides by five. Memory scoring uses the pinned text2vec encoder and a cosine threshold of 0.85. See the [reproduction notes](REPRODUCIBILITY.md) for data identities, scoring details, and file verification.

Core CPU tests:

```sh
python -m unittest discover -s . -p test_review_core.py -v
```

## License and upstream resources

The core code is released under the [MIT License](LICENSE). Dataset, model, and encoder artifacts retain their respective upstream terms. The benchmarks are [LexRAG](https://github.com/CSHaitao/LexRAG) and [LawBench](https://github.com/open-compass/LawBench). Model revisions, official acquisition commands, file checksums, and the environment lock are recorded in the [dependency guide](checkpoints/retrained/reproduction/README.md).
