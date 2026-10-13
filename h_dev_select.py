"""Dev-side scoring + checkpoint selection (pure CPU, no GPU dependency).

Input: one Base prediction file + one or more FT checkpoint prediction files
(from scripts/g_generate_predictions.py --partition dev). Output: a frozen
selection artifact (tamper-evident, protocol-fingerprinted) under --out.

    python scripts/h_dev_select.py --domain cpl --base predictions/cpl/base \
        --ft predictions/cpl/ckpt-100 predictions/cpl/ckpt-180 ... \
        --out artifacts/selection/cpl_selection.json

Refuses on: unknown global_step in any FT file, dev coverage gaps, gold that
disagrees with the split manifest, or any FT file that carries test rows.
The selector sees dev rows only, by construction of this entry.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ga_pipeline import load_profile_and_protocol  # noqa: E402
from ga_scoring import macro_view  # noqa: E402
from ga_select import (Candidate, make_selection,  # noqa: E402
                             freeze_selection, select_checkpoint)
from ga_split import split_manifest_hash  # noqa: E402


def load_predictions(path: str, dev_ids: set[str], question_hashes: dict[str, str],
                     profile_sha: str, protocol_sha: str) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    run_id = data.get("run_id")
    if not run_id:
        raise SystemExit(f"{path}: missing run_id")
    if data.get("partition") != "dev":
        raise SystemExit(f"{path}: partition is {data.get('partition')!r}; the "
                         "selector only consumes dev runs")
    per_q = data.get("per_question")
    if not isinstance(per_q, dict) or not per_q:
        raise SystemExit(f"{path}: missing per_question")
    extra = sorted(set(per_q) - dev_ids)[:5]
    if extra:
        raise SystemExit(f"{path}: carries non-dev rows {extra}; refusing")
    missing = sorted(dev_ids - set(per_q))[:5]
    if missing:
        raise SystemExit(f"{path}: dev coverage gap {missing}; refusing")
    # G8 identity gates: stale predictions made under other configs must not
    # be re-stamped with the CURRENT selection fingerprint
    if data.get("profile_sha256") != profile_sha:
        raise SystemExit(f"{path}: profile_sha256 does not match the current "
                         "Base/FT profile; prediction is from a different config")
    if data.get("protocol_sha256") != protocol_sha:
        raise SystemExit(f"{path}: protocol_sha256 does not match the current "
                         "protocol; prediction is from a different protocol")
    if not isinstance(data.get("decode"), str) or not data.get("decode_params"):
        raise SystemExit(f"{path}: missing decode/decode_params identity")
    if not data.get("identity"):
        raise SystemExit(f"{path}: missing identity block (model/adapter/tokenizer)")
    for qid, row in per_q.items():
        if row.get("question_sha256") != question_hashes[qid]:
            raise SystemExit(f"{path}: question {qid} text hash differs from the "
                             "split manifest (question edited?); refusing")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=["civil", "cpl", "criminal_law"], required=True)
    parser.add_argument("--base", required=True, help="base dev prediction file")
    parser.add_argument("--split", default=None,
                        help="split manifest path (default artifacts/split/<domain>_split.json)")
    parser.add_argument("--profile", default=str(ROOT / "base_ft_eval_profile.json"),
                        help="Base/FT evaluation profile (explicit for the 0414 flow)")
    parser.add_argument("--protocol", default=str(ROOT / "research_protocol.json"))
    parser.add_argument("--ft", nargs="+", required=True, help="FT dev prediction files")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    profile, protocol = load_profile_and_protocol(args.profile, args.protocol)

    split_path = Path(args.split) if args.split else (
        ROOT / "artifacts" / "split" / f"{args.domain}_split.json")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split.get("domain") != args.domain:
        raise SystemExit(f"split domain {split.get('domain')!r} != --domain {args.domain!r}")
    if split_manifest_hash(split) != split["split_manifest_hash"]:
        raise SystemExit("split manifest tampered; regenerate before selecting")
    dev_ids = set(split["dev_ids"])
    manifest_gold = {q["qid"]: set(q["gold"]) for q in split["questions"]
                     if q["side"] == "dev"}

    import hashlib
    def sha(payload) -> str:
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False,
                                         sort_keys=True).encode("utf-8")).hexdigest()
    profile_sha, protocol_sha = sha(profile), sha(protocol)
    question_hashes = {q["qid"]: q["question_sha256"] for q in split["questions"]
                       if q["side"] == "dev"}
    base_file = load_predictions(args.base, dev_ids, question_hashes,
                                 profile_sha, protocol_sha)
    def check_gold(run_id: str, per_q: dict) -> None:
        for qid in sorted(dev_ids):
            if set(per_q[qid]["gold"]) != manifest_gold[qid]:
                raise SystemExit(f"{run_id}: gold mismatch on dev question {qid}; refusing")
    check_gold(base_file["run_id"], base_file["per_question"])

    # G8: decode variant + seed must be IDENTICAL across base and all FT files
    def check_decode_consistency(files: list[dict]) -> None:
        ref = (base_file["decode"], base_file["decode_params"], base_file["seed"])
        for f in files:
            if (f["decode"], f["decode_params"], f["seed"]) != ref:
                raise SystemExit(
                    f"{f['run_id']}: decode/seed differs from base "
                    f"({f['decode']}/{f['seed']} vs base {ref[0]}/{ref[2]}); "
                    "Base/FT must share one decode configuration")
    # G8: tokenizer/model identity must match base (adapter may differ)
    def check_model_consistency(files: list[dict]) -> None:
        ref_id = base_file["identity"]
        for f in files:
            if f["identity"]["tokenizer"] != ref_id["tokenizer"]:
                raise SystemExit(f"{f['run_id']}: tokenizer identity differs from base")
            if f["identity"]["model"] != ref_id["model"]:
                raise SystemExit(f"{f['run_id']}: base-model identity differs from base run")
            if not f["identity"].get("adapter"):
                raise SystemExit(f"{f['run_id']}: FT candidate lacks adapter identity")

    def dev_score(per_q: dict) -> float:
        rows = [{"id": q, "pred": per_q[q]["pred"], "gold": per_q[q]["gold"]}
                for q in sorted(dev_ids)]
        return macro_view(rows, args.domain, "full_gold_all")["R@5"]

    candidates = []
    ft_files = []
    for path in args.ft:
        data = load_predictions(path, dev_ids, question_hashes,
                                profile_sha, protocol_sha)
        step = data.get("global_step")
        if not isinstance(step, int):
            raise SystemExit(f"{path}: FT candidate has no integer global_step; "
                             "the selector refuses unknown-step candidates")
        check_gold(data["run_id"], data["per_question"])
        ft_files.append(data)
    check_decode_consistency(ft_files)
    check_model_consistency(ft_files)
    for data in ft_files:
        score = dev_score(data["per_question"])
        candidates.append(Candidate(data["run_id"], data["global_step"], score))
        print(f"[dev] {data['run_id']} (step {data['global_step']}): macro R@5 = {score:.6f}")

    def identity_of(data: dict) -> dict:
        return {"decode": data["decode"], "decode_params": data["decode_params"],
                "seed": data["seed"], "identity": data["identity"]}

    best = select_checkpoint(candidates)
    best_file = next(f for f in ft_files if f["run_id"] == best.run_id)
    base_score = dev_score(base_file["per_question"])
    print(f"[dev] base {base_file['run_id']}: macro R@5 = {base_score:.6f}")
    artifact = make_selection(args.domain, candidates,
                              Candidate(base_file["run_id"], 0, base_score),
                              split["split_manifest_hash"], protocol,
                              profile=profile,
                              selected_identity=identity_of(best_file),
                              base_identity=identity_of(base_file))
    freeze_selection(artifact, args.out)
    print(f"selected {artifact.selected_run_id} (step {artifact.selected_global_step}) "
          f"-> {args.out}")


if __name__ == "__main__":
    main()
