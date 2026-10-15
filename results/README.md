# Released results

Start with the existing outputs, without requesting APIs or retraining:

```sh
python audit_publication.py
python plot_fig3.py
```

| Paper location | Files |
|---|---|
| Table 1 | `table1_main/`: all five generative backends' ID-level predictions/gold; original response text for Qwen, GLM and DeepSeek. Claude/GPT archives retain parsed predictions. `index.json` records denominators and printed ranking values. RAPTOR/LinearRAG per-sample archives and the available BM25/Hybrid/Embedding aggregate archives are also included. |
| Table 2 | `knowledge_injection.json`, `per_question/`, `table2_knowledge_injection/index.json` and shared raw GLM retrieval outputs. |
| Table 3 | `table3_ablation/`: original archived ID-Only/ID+Content files, generated statute text and summaries, retained byte-for-byte. `ARCHIVE_MANIFEST.json` records hashes and the original mixed JSONL/text format where present. No Table 1 parser is applied to these files. |
| Fig.3 / Table 7 | `fig3_table7/`: 14 runs, 12,355 original final statute generations, 1,568 retrieval answers and fixed reference statutes; `per_article/` and `per_question/` contain their scored records. |
| Other manuscript tables | `paper_tables/` and `paper_tables.json`, at the original reported precision. |

The three BM25/Hybrid/Embedding archives currently provide aggregates; per-question traces are not supplied for those methods. Archived baseline predictions retain their original scoring context. Table 1 Hit@K and MRR use nonempty-gold questions (954 LexRAG / 439 LawBench); the memory/SSL diagnostics use fixed 180/44 test questions and macro Recall@5. These are distinct metrics and protocols. For Qwen Table 1, the frozen first-record-per-ID rule is retained.

Raw recitation files preserve empty and truncated final outputs. Provider reasoning and administration records are omitted; final content is unchanged. To recompute encoder similarities, obtain the pinned text2vec scorer and use `review_cli.py memory` with the matching reference corpus and raw-recitation JSONL. The default audit checks saved similarities/classes and exact ID scores without loading an encoder.

The archived Table 3 files retain their original fields, including benchmark text; upstream dataset terms remain applicable. Evaluation cases/gold are never training inputs. The historical outputs here are preserved separately from adapter artifacts under `checkpoints/retrained/`.

Table 1 records use file-order `record_id` locators. The original `item_id` is retained, including null IDs in the archived Claude/GPT LawBench files; no question-to-record mapping is invented.
