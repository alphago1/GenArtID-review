# GenArtID: Generative Article Identifier Prediction for Legal Retrieval

Official reproduction package for *Rethinking Legal Retrieval as Generative Article Identifier Prediction*.

## Repository Structure

```
├── README.md
├── data/                          # All input datasets
│   ├── lexrag_1013_questions.json       # LexRAG 1013 consultation questions
│   ├── lawbench_processed.json          # LawBench 439 questions
│   ├── law_library.jsonl                # 222 laws, 17,228 articles
│   ├── civil_code_articles.json         # Civil Code 1,260 articles
│   ├── cpl_articles.json                # Criminal Procedure Law 307 articles
│   ├── eval_civil_code_questions.json   # Civil Code 361 evaluation questions
│   ├── eval_cpl_questions.json          # CPL 48 evaluation questions
│   ├── train_sft_civil_code.json        # Civil Code SFT training data
│   └── train_sft_cpl.json               # CPL SFT training data
│
├── scripts/                       # Executable scripts (by function)
│   ├── genartid_retrieve.py             # ID-Only retrieval (API)
│   ├── id_content_retrieve.py           # ID+Content retrieval (API)
│   ├── content_fidelity.py              # Corr/Part/Wrong evaluation
│   ├── compute_ranking.py               # AvgRank / Hit@k
│   ├── bm25_hybrid.py                   # BM25 + Hybrid baselines
│   ├── embedding_baseline.py            # Embedding retrieval baseline
│   ├── raptor_run.py                    # RAPTOR tree retrieval
│   ├── recite_laws.py                   # Statute recitation via API
│   ├── compare_civil_code.py            # Civil Code recitation evaluation
│   ├── compare_criminal_procedure.py    # CPL recitation evaluation
│   ├── run_glm4.py                      # GLM-4-9B experiments
│   ├── cpl_base_retrieval.py            # CPL base model retrieval
│   ├── make_sft_data.py                 # SFT data construction
│   └── prepare_stage1_data.py           # Stage-1 training data
│
├── notebooks/                     # Jupyter notebooks (AutoDL GPU)
│   ├── civilTrainAndEval.ipynb          # Civil Code LoRA training + eval
│   ├── cplTrainAndEval.ipynb            # CPL LoRA training + eval
│   ├── civilcode_checkpoint280_recite.ipynb  # CC LoRA memory test
│   ├── cpl_checkpoint-180_recite.ipynb       # CPL LoRA memory test
│   ├── civilBaseRecall.ipynb            # CC base recall evaluation
│   ├── cc_base_stage2_eval.ipynb        # CC base + Stage-2 eval
│   └── cpl_eval_all.ipynb               # CPL all-checkpoint eval
│
├── checkpoints/                   # Trained LoRA weights
│   ├── bestincivilcheckpoint.tar.gz     # Civil Code checkpoint-280
│   └── bestincplcheckpoint.tar.gz       # CPL checkpoint-180
│
├── results/                       # All output data (by paper table)
│   ├── table1_main/                     # Main results: 10 methods × 2 datasets
│   ├── table2_ranking/                  # Knowledge injection: ranking quality
│   ├── table3_ablation/                 # Ablation: ID-Only vs ID+Content
│   ├── table5_fidelity/                 # Content fidelity: Corr/Part/Wrong
│   └── table7_memory/                   # Cross-model memory + retrieval
│
└── case_study/                    # Table 4 & 8 qualitative case study evidence
```

## Paper Tables → Results Mapping

| Table | Content | Results Directory | Key Scripts |
|-------|---------|-------------------|-------------|
| Table 1 | Main retrieval comparison | `results/table1_main/` | `scripts/genartid_retrieve.py`, `scripts/bm25_hybrid.py`, `scripts/embedding_baseline.py`, `scripts/raptor_run.py` |
| Table 2 | Knowledge injection ranking | `results/table2_ranking/` | `scripts/compute_ranking.py` |
| Table 3 | ID-Only vs ID+Content ablation | `results/table3_ablation/` | `scripts/genartid_retrieve.py`, `scripts/id_content_retrieve.py`, `scripts/content_fidelity.py` |
| Table 4 | Case study (Civil Code) | `case_study/` | — |
| Table 5 | Content fidelity | `results/table5_fidelity/` | `scripts/content_fidelity.py` |
| Table 6 | QLoRA hyperparameters | See paper | `notebooks/civilTrainAndEval.ipynb` |
| Table 7 | Cross-model memory + retrieval | `results/table7_memory/` | `scripts/recite_laws.py`, `scripts/compare_*.py`, `notebooks/*.ipynb` |
| Table 8 | Case study (CPL) | `case_study/` | — |

## Getting Started

### Requirements

- Python 3.10+
- DeepSeek API key (for GenArtID retrieval)
- SiliconFlow API key (for Qwen3-8B, GLM-4-9B, Embedding baselines)
- GPT And Claude API key
- NVIDIA RTX 5090 (for LoRA fine-tuning notebooks)

### Quick Verification

```bash
# Verify Table 2 ranking metrics
python scripts/compute_ranking.py

# Verify Table 5 content fidelity
python scripts/content_fidelity.py
```

### Full Reproduction

1. **Retrieval baselines**: `python scripts/bm25_hybrid.py`, `scripts/embedding_baseline.py`
2. **GenArtID retrieval**: `python scripts/genartid_retrieve.py` (requires API keys)
3. **LoRA fine-tuning**: Run notebooks in `notebooks/` on AutoDL RTX 5090
4. **Memory evaluation**: `python scripts/recite_laws.py` → `scripts/compare_*.py`
5. **Ranking analysis**: `python scripts/compute_ranking.py`

## Citation

```bibtex
@inproceedings{genartid,
  title     = {Rethinking Legal Retrieval as Generative Article Identifier Prediction},
  author    = {},
  booktitle = {ACL},
  year      = {2025}
}
```
