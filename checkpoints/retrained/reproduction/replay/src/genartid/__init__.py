"""GenArtID corpus-lab CPU protocol package (standard library only).

Layers:
- identifiers: canonical Chinese article IDs (verbatim port of notebook helpers).
- parsing:     legacy (historical parity) and fixed (revision 2) output parsers.
- scoring:     Recall@5 family metrics, macro aggregation, paired bootstrap.
- corpus:      strict corpus loading, per-source newline repair, normalization.
- build_sft:   corpus-only training data construction (no eval/case data).
- split:       deterministic dev/test split over question groups.
- select:      dev-only checkpoint selection and gated test evaluation.
"""
__version__ = "1.0.0"
