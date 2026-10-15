"""Base-performance-balanced dev/test split for the GLM-4-9B-0414 experiment.

Frozen algorithm ``glm0414-balanced-stratified/1`` (PLAN_GLM0414 A §4 / B §4):

1. Group questions by whitespace-normalized question text (same group key as
   ``genartid.split``; identical normalized text = same case = one group that
   never straddles halves).
2. Stratify groups by Base correctness layer computed from ONE model's Base
   calibration only: layer 0 = group mean recall@5 == 0 (all miss),
   layer 1 = partial (0 < mean < 1), layer 2 = full hit (mean == 1).
   No FT result ever enters this module.
3. Within layers (processed 0, 1, 2 in order) groups are sorted by
   group_key_sha256, then shuffled by ONE ``random.Random(seed)`` stream
   (seeded Fisher-Yates, continuing across layers).
4. Alternating assignment inside each layer, first group -> dev; with an odd
   layer count dev receives the extra group (same half-up convention as
   ``e_make_split``).
5. Deterministic whole-group improvement swaps: scan candidate swaps in fixed
   order (dev groups by key x test groups by key), accept the FIRST swap that
   strictly reduces |dev_mean - test_mean| (eps 1e-12) while not increasing
   |dev_n - test_n|; repeat full passes until none is accepted (bounded).
6. Targets are checked honestly: a missed target is REPORTED, never faked.

Output manifests are ``genartid-split/1`` compatible so the existing
dev/test entries (k_eval_checkpoints / h_dev_select / g_generate_predictions /
run_test_eval) consume them unchanged via explicit --split paths.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from .split import normalized_question

EPS = 1e-12
MAX_SWAPS = 10_000


def group_key_sha256(text: str) -> str:
    return hashlib.sha256(normalized_question(text).encode("utf-8")).hexdigest()


def layer_of(mean: float) -> int:
    if mean <= 0:
        return 0
    if mean >= 1:
        return 2
    return 1


def build_balanced_split(domain: str, questions: list[dict], seed: int = 20260910,
                         balance_target: float = 0.01) -> dict:
    """Construct the balanced split manifest from Base calibration rows.

    ``questions``: one dict per question with qid, question (raw text), gold
    (canonical list), recall_at_5 (float in [0,1] from the Base calibration),
    question_sha256.  Pure function of its inputs; deterministic.
    """
    by_qid = {}
    for q in questions:
        qid = str(q["qid"])
        if qid in by_qid:
            raise ValueError(f"duplicate qid {qid}")
        r = q["recall_at_5"]
        if not isinstance(r, (int, float)) or isinstance(r, bool) or not 0 <= r <= 1:
            raise ValueError(f"question {qid}: recall_at_5 must be a number in [0,1]")
        by_qid[qid] = {
            "qid": qid, "question": q["question"], "gold": list(q["gold"]),
            "recall": float(r), "question_sha256": q["question_sha256"],
            "group": group_key_sha256(q["question"]),
        }

    groups: dict[str, list[dict]] = {}
    for row in by_qid.values():
        groups.setdefault(row["group"], []).append(row)
    for members in groups.values():
        members.sort(key=lambda m: m["qid"])            # deterministic membership order

    group_info = {}
    for gkey, members in groups.items():
        mean = sum(m["recall"] for m in members) / len(members)
        group_info[gkey] = {"mean": mean, "layer": layer_of(mean),
                            "qids": [m["qid"] for m in members]}

    # step 3+4: per-layer seeded shuffle then alternating assignment
    rng = random.Random(seed)
    side_of_group: dict[str, str] = {}
    strata_sizes = {}
    for layer in (0, 1, 2):
        members = sorted(k for k, info in group_info.items() if info["layer"] == layer)
        order = list(members)
        rng.shuffle(order)
        strata_sizes[layer] = len(order)
        for position, gkey in enumerate(order):
            side_of_group[gkey] = "dev" if position % 2 == 0 else "test"

    def side_rows(sides: dict[str, str], which: str) -> list[dict]:
        out = []
        for gkey, members in groups.items():
            if sides[gkey] == which:
                out.extend(members)
        return out

    def mean_of(rows: list[dict]) -> float:
        return sum(r["recall"] for r in rows) / len(rows) if rows else 0.0

    # step 5: deterministic first-improvement whole-group swaps
    swaps = 0
    while swaps < MAX_SWAPS:
        dev_keys = sorted(k for k, s in side_of_group.items() if s == "dev")
        test_keys = sorted(k for k, s in side_of_group.items() if s == "test")
        dev_rows, test_rows = side_rows(side_of_group, "dev"), side_rows(side_of_group, "test")
        base_gap = abs(mean_of(dev_rows) - mean_of(test_rows))
        base_n_gap = abs(len(dev_rows) - len(test_rows))
        accepted = None
        for d in dev_keys:
            for t in test_keys:
                side_of_group[d], side_of_group[t] = "test", "dev"
                nd = side_rows(side_of_group, "dev")
                nt = side_rows(side_of_group, "test")
                gap = abs(mean_of(nd) - mean_of(nt))
                n_gap = abs(len(nd) - len(nt))
                if gap < base_gap - EPS and n_gap <= base_n_gap:
                    accepted = (d, t, gap, n_gap)
                    break
                side_of_group[d], side_of_group[t] = "dev", "test"
            if accepted:
                break
        if not accepted:
            break
        swaps += 1

    dev_rows, test_rows = side_rows(side_of_group, "dev"), side_rows(side_of_group, "test")
    dev_mean, test_mean = mean_of(dev_rows), mean_of(test_rows)
    pooled_mean = mean_of(list(by_qid.values()))

    def layer_counts(rows: list[dict]) -> dict:
        out = {0: 0, 1: 0, 2: 0}
        for r in rows:
            out[group_info[r["group"]]["layer"]] += 1
        return {str(k): v for k, v in out.items()}

    rows_sorted = sorted(by_qid.values(), key=lambda r: r["qid"])
    dev_ids = sorted(r["qid"] for r in dev_rows)
    test_ids = sorted(r["qid"] for r in test_rows)
    manifest = {
        "schema": "genartid-split/1",
        "domain": domain,
        "seed": seed,
        "dev_fraction": len(dev_rows) / (len(dev_rows) + len(test_rows)),
        "algorithm": "glm0414-balanced-stratified/1: normalized-text groups -> "
                     "Base-recall strata (0/partial/1 by group mean) -> seeded "
                     "Fisher-Yates within strata (one Random(seed) stream, strata in "
                     "order 0,1,2) -> alternate dev/test starting dev -> deterministic "
                     "first-improvement whole-group swaps (objective |dev_mean-test_mean|, "
                     "must not worsen |dev_n-test_n|)",
        "group_key": "whitespace_normalized_question_text",
        "near_duplicate_policy": "exact normalized-text groups only; semantic near-duplicates NOT grouped (CPU boundary, no embedding model); limitation recorded",
        "historical_exposure": "all questions were used in past experiments; this is a retrospective internal holdout balanced on 0414 Base performance",
        "counts": {
            "questions": len(by_qid), "groups": len(groups),
            "dev_questions": len(dev_ids), "test_questions": len(test_ids),
            "questions_in_multi_id_groups": sum(
                len(m) for m in groups.values() if len(m) > 1),
        },
        "balanced_resplit": {
            "schema": "glm0414-balanced-split/1",
            "balance_source": "Base calibration only (adapter off); never FT results",
            "strata_group_sizes": {str(k): v for k, v in strata_sizes.items()},
            "swaps_applied": swaps,
            "dev_base_macro_r5": dev_mean, "test_base_macro_r5": test_mean,
            "pooled_base_macro_r5": pooled_mean,
            "dev_hit_at_5": sum(1 for r in dev_rows if r["recall"] > 0) / len(dev_rows),
            "test_hit_at_5": sum(1 for r in test_rows if r["recall"] > 0) / len(test_rows),
            "dev_layer_counts": layer_counts(dev_rows),
            "test_layer_counts": layer_counts(test_rows),
            "balance_target_abs_diff": balance_target,
            "target_met": abs(dev_mean - test_mean) <= balance_target + EPS,
            "achieved_abs_diff": abs(dev_mean - test_mean),
        },
        "dev_ids": dev_ids, "test_ids": test_ids,
        "questions": [{
            "qid": r["qid"], "side": side_of_group[r["group"]], "gold": r["gold"],
            "question_sha256": r["question_sha256"],
            "group_key_sha256": r["group"], "group_size": len(groups[r["group"]]),
            "base_recall_at_5": r["recall"],
            "exposure_status": "historically_exposed",
        } for r in rows_sorted],
    }
    from .split import split_manifest_hash
    manifest["split_manifest_hash"] = split_manifest_hash(manifest)
    return manifest
