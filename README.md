# GenArtID Paper Reproduction

Paper: *Rethinking Legal Retrieval as Generative Article Identifier Prediction*

## Structure

```
data/          — Shared data (questions, law library, training data)
table1/        — Main results: 10 methods on LexRAG + LawBench
table2/        — Knowledge injection: ranking quality (AvgRank, Hit@1/3/5)
table3/        — Ablation: ID-Only vs ID+Content + content fidelity
table4/        — Case study (see case_study/)
table5/        — Content fidelity: Correct/Partial/Wrong
table6/        — QLoRA hyperparameters (see paper Table 6)
table7/        — Cross-model memory and retrieval (Civil Code + CPL)
case_study/    — Table 4 Case 250 + Table 8 Case 207 raw data
```

## Data Flow

```
data/lexrag_1013_questions.json    → table1/ (retrieval eval)
data/lawbench_processed.json       → table1/ (retrieval eval)
data/law_library.jsonl             → table1/ table3/ table5/ (article matching)
data/civil_code_articles.json      → table2/ table7/ (CC recitation + retrieval)
data/cpl_articles.json             → table2/ table7/ (CPL recitation + retrieval)
data/eval_civil_code_questions     → table2/ table7/ (361 questions)
data/eval_cpl_questions.json       → table2/ table7/ (48 questions)
data/train_sft_*.json              → table7/ (LoRA fine-tuning)
```

## Key Scripts

| Experiment | Script | Runtime |
|------------|--------|---------|
| GenArtID retrieval | `table3/run_genartid_id_only.py` | API |
| Content fidelity | `table3/recompute_table2_cosine.py` | local CPU |
| Ranking quality | `table2/compute_ranking.py` | local CPU |
| BM25/Hybrid | `table1/run_bm25_hybrid.py` | local CPU |
| Embedding baseline | `table1/run_embedding_baseline.py` | API |
| RAPTOR | `table1/raptor_run_raptor.py` | local GPU |
| LoRA fine-tuning | `table7/*.ipynb` | AutoDL GPU |
| GLM-4-9B | `table7/run_table3_glm4.py` | API |
| SFT data construction | `data/make_sft_data.py` | local |

## Requirements

- DeepSeek API Key (GenArtID retrieval)
- SiliconFlow API Key (Qwen3-8B / GLM-4-9B / Embedding)
- AutoDL RTX 5090 (LoRA fine-tuning + recitation eval)
