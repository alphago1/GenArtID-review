# Reproduction materials

For the released paper outputs, start with `python audit_publication.py` and `python plot_fig3.py`. These use existing records without API requests, model inference or adapter training. `audit_results.py` checks original numerical evidence and release hashes; `audit_publication.py` additionally verifies Table 1's five generative backends, every Table 2 metric/seed mean/SD, original Table 3 file hashes, and raw-to-scored Fig.3/Table 7 coverage and parsing.

See [results/README.md](results/README.md) for table-to-file mappings and available baseline coverage. Table 3 retains its original archived protocol and data. Its files are not rescored using the Table 1 parser. The fixed membership, source hashes, original development selection, exact fractions and retrospective status are retained.

Six downloadable adapter archives, their loading instructions and provenance are in [checkpoints/retrained/](checkpoints/retrained/README.md). Historical selected-step configurations remain in `checkpoints/manifest.json`; the supplied archives have their own identities and metadata. Check all archives on CPU with `python checkpoints/retrained/verify_release.py`. Use `unpack_adapters.py --extract` to obtain the supplied weights for inference; retraining is optional.

The base is `zai-org/GLM-4-9B-0414` at `645b8482494e31b6b752272bf7f7f273ef0f3caf`; the memory scorer is `shibing624/text2vec-base-chinese` at `183bb99aa7af74355fb58d16edf8c13ae7c5433e`. Their exact file sizes/SHA256 and acquisition commands, plus the recorded Python/CUDA/package lock, are in `checkpoints/retrained/reproduction/`. Upstream model weights and full installed environments are not bundled.

Raw final-content recitations and matching statute references now allow separate encoder replay. The default audit checks saved scores and classifications, without claiming a fresh GPU/encoder replay. BM25/Hybrid/Embedding per-question traces and the fine-tuned embedding adapter are not supplied; their published aggregates remain in the package. Preserve upstream licenses and use the fixed data snapshots when performing further evaluation.

Refreshing an anonymous snapshot and testing its binary downloads is separate from verifying this Git repository.
