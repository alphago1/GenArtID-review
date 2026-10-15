# GenArtID

Code, evaluation data, and adapter checkpoints for **Rethinking Legal Retrieval as Generative Article Identifier Prediction**.

GenArtID retrieves legal provisions by predicting their canonical article identifiers and binding them to authoritative statute text. Its corpus-only knowledge injection associates article identifiers with their content through identifier-to-text and text-to-identifier examples, without using annotated case–article pairs for training.

## Paper results and data

The release includes per-question predictions, generated statute text, scoring records, and the reported table summaries.

| Paper result | Released materials |
| --- | --- |
| **Table 1: retrieval comparison** | [Predictions and index](results/table1_main/index.json) for five generative models on LexRAG and LawBench; [baseline prediction archives](results/table1_main/baseline_archives/) and [baseline aggregates](results/table1_main/aggregate_archives/). |
| **Table 2: knowledge injection** | [Three-seed results and statistics](results/knowledge_injection.json), [per-question records](results/per_question/), and the [raw-output index](results/table2_knowledge_injection/index.json). |
| **Table 3: identifier/content ablation** | [Original archived outputs and summaries](results/table3_ablation/), including generated statute content for the ID+Content setting. |
| **Fig. 3 and Table 7: memory and retrieval** | [Raw outputs and reference statutes](results/fig3_table7/index.json), [per-article scores](results/per_article/), [run summaries](results/memory_recall_runs.json), and the [ten plotted points](fig3_points.json). |
| **Other tables** | [CSV exports](results/paper_tables/) and [structured table data](results/paper_tables.json). |

The Fig. 3 / Table 7 release contains **12,355 statute generations and 1,568 retrieval answers across 14 model/domain runs**. The ten displayed points include the three-seed GLM SSL means. Table 3 retains its original archived files and scoring summaries.

See the [data guide](results/README.md) for file formats and coverage. Original response text is available for the Qwen, GLM, and DeepSeek Table 1 runs; the Claude/GPT archives retain parsed predictions. BM25, Hybrid, and Embedding baselines provide saved aggregate results.

## Verify the released results

From the repository root, with Python 3.10 or later:

```sh
python audit_publication.py
```

This standard-library audit checks release hashes, Table 1 generative-model ranking metrics, Table 2 seed results and statistics, Table 3 archive integrity, and the saved Fig. 3 / Table 7 records. It recomputes retrieval scores and checks memory classifications against stored similarities. No GPU or API access is needed.

To render Fig. 3 from the released coordinates:

```sh
python -m pip install -r requirements.txt
python plot_fig3.py
```

The plotter writes `fig3.pdf` and `fig3.svg`.

## Adapter checkpoints

Six PEFT adapter archives are provided for **GLM-4-9B-0414**, covering three seeds in each legal domain. Each archive contains adapter weights, configuration, and run metadata; individual ZIP files are approximately 42 MB.

| Seed | Civil Code | Criminal Law |
| --- | --- | --- |
| 20260910 | [Step 600](checkpoints/retrained/civil_seed20260910_step600.zip) | [Step 40](checkpoints/retrained/criminal_law_seed20260910_step40.zip) |
| 20260911 | [Step 1575](checkpoints/retrained/civil_seed20260911_step1575.zip) | [Step 220](checkpoints/retrained/criminal_law_seed20260911_step220.zip) |
| 20260912 | [Step 520](checkpoints/retrained/civil_seed20260912_step520.zip) | [Step 40](checkpoints/retrained/criminal_law_seed20260912_step40.zip) |

Verify the archives, then extract the adapter folders for use with the matching base model:

```sh
python checkpoints/retrained/verify_release.py
python checkpoints/retrained/unpack_adapters.py --extract
```

Both commands run on CPU without loading model weights. Use the pinned base model and tokenizer described in the [dependency guide](checkpoints/retrained/reproduction/README.md); it includes official download commands, file checksums, and the recorded environment lock. [Loading and evaluation commands](checkpoints/retrained/reproduction/replay/README.md) and [adapter provenance](checkpoints/retrained/NOTICE.txt) accompany the files.

## Evaluation conventions

**Table 1 reports Hit@K and MRR.** Hit@K measures whether at least one gold article appears in the first K predictions. Its evaluated subsets contain 954 LexRAG questions and 439 LawBench questions with nonempty gold labels.

**The knowledge-injection and memory diagnostics use macro Recall@5**, which averages the fraction of gold articles retrieved for each question. Their fixed test sets contain 180 Civil Code questions and 44 Criminal Law questions. The `fixed_v2` parser uses the original five output slots: invalid or duplicate entries consume a slot, and no additional candidates are used to refill it. P@5 uses a fixed denominator of five.

The statute-memory rate is **M = Eff. / N**, using a cosine threshold of 0.85 with the pinned text2vec scorer. Raw generations, reference statutes, and saved similarity scores are included for inspection. The [reproduction notes](REPRODUCIBILITY.md) describe the matching data, scoring settings, and artifact coverage.

## Code and configuration

| Entry | Purpose |
| --- | --- |
| [review_cli.py](review_cli.py) | Corpus preparation, article-ID parsing, retrieval evaluation, ID-to-text binding, API generation, and memory scoring. Run `python review_cli.py --help` for commands. |
| [research_protocol.json](research_protocol.json) and [base_ft_eval_profile.json](base_ft_eval_profile.json) | Recorded experiment settings and evaluation configuration. |
| [splits/](splits/) and [selection/](selection/) | Fixed diagnostic split membership and historical development-set checkpoint selection. |
| [training/](training/) | Recorded training configurations. |
| [release_manifest.json](release_manifest.json) | Published file sizes and SHA256 checksums. |

Run the core CPU tests with:

```sh
python -m unittest discover -s . -p test_review_core.py -v
```

## License and upstream resources

The core code is released under the [MIT License](LICENSE). Dataset, model, and encoder artifacts retain their respective upstream terms. The benchmarks are [LexRAG](https://github.com/CSHaitao/LexRAG) and [LawBench](https://github.com/open-compass/LawBench); model revisions and acquisition details are recorded in the [dependency guide](checkpoints/retrained/reproduction/README.md).
