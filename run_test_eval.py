"""P2 entry: gated test evaluation of a frozen selection against its matched Base.

Usage (from the workspace root):

    python scripts/run_test_eval.py --domain cpl \
        --selection artifacts/selection/cpl_selection.json \
        --split artifacts/split/cpl_split.json \
        --base predictions/base.json --ft predictions/ckpt180.json \
        --out artifacts/test_eval/cpl

Prediction files are JSON: {"run_id": str,
  "per_question": {qid: {"pred": [...], "gold": [...]}}}
produced by the (future) GPU runner under configs/base_ft_eval_profile.json.
This entry refuses on: missing/edited selection, protocol fingerprint mismatch,
split hash mismatch, run identity mismatch, coverage gaps, or gold mismatch.
Historical per-checkpoint raw outputs are NOT part of this workspace; until
the GPU runner produces prediction files, this entry correctly refuses with
"missing inputs" rather than fabricating a re-analysis.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ga_select import (TestGateError, load_frozen_selection,  # noqa: E402
                             run_test_evaluation, verify_gold_match)


def load_predictions(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "run_id" not in data or "per_question" not in data:
        raise SystemExit(f"{path}: prediction file must have run_id and per_question")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=["civil", "cpl", "criminal_law"], required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--ft", required=True)
    parser.add_argument("--protocol", default=str(ROOT / "research_protocol.json"))
    parser.add_argument("--profile", default=str(ROOT / "base_ft_eval_profile.json"),
                        help="Base/FT evaluation profile; bound into the protocol fingerprint "
                             "(decode/prompt changes invalidate selections)")
    parser.add_argument("--out", required=True, help="new output directory (must not exist)")
    args = parser.parse_args()

    for label, path in (("selection", args.selection), ("split", args.split),
                        ("base", args.base), ("ft", args.ft)):
        if not Path(path).is_file():
            raise SystemExit(f"missing input ({label}): {path}\n"
                             "historical checkpoint raw outputs are not in this workspace; "
                             "produce prediction files with the GPU runner first")
    if Path(args.out).exists():
        raise SystemExit("output directory exists; use a new run name")

    selection = load_frozen_selection(args.selection)
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    profile = json.loads(Path(args.profile).read_text(encoding="utf-8"))
    base_file = load_predictions(Path(args.base))
    ft_file = load_predictions(Path(args.ft))

    if selection["domain"] != args.domain:
        raise SystemExit(f"selection domain {selection['domain']} != {args.domain}")
    if base_file["run_id"] != selection["base_run_id"]:
        raise SystemExit(f"base run_id {base_file['run_id']!r} does not match the matched "
                         f"base run {selection['base_run_id']!r}")
    if ft_file["run_id"] != selection["selected_run_id"]:
        raise SystemExit(f"ft run_id {ft_file['run_id']!r} is not the selected run "
                         f"{selection['selected_run_id']!r}; refusing to evaluate a different "
                         "checkpoint on test")

    base_pred = {qid: row["pred"] for qid, row in base_file["per_question"].items()}
    ft_pred = {qid: row["pred"] for qid, row in ft_file["per_question"].items()}
    base_gold = {qid: row["gold"] for qid, row in base_file["per_question"].items()}
    ft_gold = {qid: row["gold"] for qid, row in ft_file["per_question"].items()}
    verify_gold_match(split, base_gold, ft_gold)

    rows = [{"qid": qid, "gold": gold}
            for qid, gold in base_gold.items() if qid in set(split["test_ids"])]
    try:
        result = run_test_evaluation(selection, protocol, split, rows,
                                     base_pred, ft_pred, profile=profile,
                                     base_pred_file=base_file,
                                     ft_pred_file=ft_file)
    except TestGateError as error:
        raise SystemExit(f"test gate refused: {error}")

    out = Path(args.out)
    out.mkdir(parents=True)
    (out / "test_eval.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in result.items() if k != "schema"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
