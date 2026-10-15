"""Scoring: Recall@5 family metrics, gold views, and paired group bootstrap.

Frozen conventions (research_protocol.json):
- Primary metric: macro Recall@5 (arithmetic mean of per-question recall).
- Each question is exactly one statistical unit; the five candidate slots are
  not independent samples.
- ``full_gold_all`` scores every dataset row (empty gold is a data anomaly to
  be reported, and scores zero only under an explicitly recorded rule).
- ``domain_gold_nonempty`` filters gold to the domain law and keeps only rows
  with non-empty domain gold; predictions are NEVER filtered or refilled.
- P@5_fixed divides by 5; P@5_legacy divides by the number of distinct
  predictions and is kept for historical accounting only.
- Bootstrap: paired resampling over question groups, seed 20260911,
  10,000 resamples; a group with several questions contributes all of its
  questions, and the macro mean is recomputed over the drawn questions.
"""
from __future__ import annotations

import random
import statistics
from dataclasses import dataclass

LAWS = {"civil": "《民法典》", "cpl": "《刑事诉讼法》", "criminal_law": "《刑法》"}

BOOTSTRAP_SEED = 20260911
BOOTSTRAP_RESAMPLES = 10_000


def per_question_scores(pred: list[str], gold: list[str]) -> dict[str, float]:
    """Per-question metrics from one top-5 prediction list and one gold list."""
    ps = [x for x in pred[:5]]
    gs = set(gold)
    tp = len(set(ps) & gs)
    recall = tp / len(gs) if gs else 0.0
    precision_fixed = tp / 5
    precision_legacy = tp / max(len(set(ps)), 1)
    f1_legacy = (2 * precision_legacy * recall / (precision_legacy + recall)
                 if precision_legacy + recall > 0 else 0.0)
    first = next((i + 1 for i, x in enumerate(ps) if x in gs), None)
    return {
        "R@5": recall,
        "P@5_fixed": precision_fixed,
        "P@5_legacy": precision_legacy,
        "F1@5_legacy": f1_legacy,
        "MRR@5": 1 / first if first else 0.0,
        "Hit@5": float(bool(set(ps) & gs)),
    }


def question_recall(pred: list[str], gold: list[str]) -> float:
    gs = set(gold)
    if not gs:
        return 0.0
    return len(set(pred[:5]) & gs) / len(gs)


def domain_filter(gold: list[str], domain: str) -> list[str]:
    law = LAWS[domain]
    return [x for x in gold if x.startswith(law + "第")]


def macro_view(rows: list[dict], domain: str, view: str) -> dict[str, float | int]:
    """Aggregate metrics for one gold view.

    ``rows``: dicts with keys ``pred`` (list), ``gold`` (list), ``id``.
    ``view``: "full_gold_all" or "domain_gold_nonempty".
    """
    if view not in ("full_gold_all", "domain_gold_nonempty"):
        raise ValueError(f"unknown view: {view}")
    scores: list[dict[str, float]] = []
    for row in rows:
        pred = list(row["pred"])[:5]
        gold = list(row["gold"])
        if view == "domain_gold_nonempty":
            gold = domain_filter(gold, domain)
            if not gold:
                continue
        scores.append(per_question_scores(pred, gold))
    if not scores:
        return {"n": 0}
    return {"n": len(scores), **{
        key: statistics.mean(s[key] for s in scores) for key in scores[0]
    }}


def summarize_views(rows: list[dict], domain: str) -> dict[str, dict]:
    """Both gold views from exactly the same top-5; never filter predictions."""
    return {view: macro_view(rows, domain, view)
            for view in ("full_gold_all", "domain_gold_nonempty")}


@dataclass
class BootstrapResult:
    n_questions: int
    n_groups: int
    point_delta: float          # FT - Base on the full set
    ci95: list[float]
    gains: int
    ties: int
    losses: int
    seed: int
    resamples: int
    sample_deltas: list[float] = None  # kept for tests/diagnostics; not in as_dict

    def __post_init__(self):
        if self.sample_deltas is None:
            self.sample_deltas = []

    def as_dict(self) -> dict:
        return {
            "n_questions": self.n_questions, "n_groups": self.n_groups,
            "point_delta_ft_minus_base": self.point_delta,
            "ci95_percentile": self.ci95, "gain_tie_loss": [self.gains, self.ties, self.losses],
            "seed": self.seed, "resamples": self.resamples,
        }


def paired_group_bootstrap(rows: list[dict], groups: dict[str, list[str]],
                           base_pred: dict[str, list[str]], ft_pred: dict[str, list[str]],
                           seed: int = BOOTSTRAP_SEED, resamples: int = BOOTSTRAP_RESAMPLES,
                           ) -> BootstrapResult:
    """Paired bootstrap with question groups as resampling units.

    ``rows``: dicts with ``id``, ``gold`` (caller applies the gold view when
    pre-filtering rows).  ``groups``: group_id -> list of question ids.
    ``base_pred`` / ``ft_pred``: question id -> prediction list.  Both arms
    must cover the same ids (gold equality is enforced by the test-gate entry
    point before this function).
    """
    golds = {str(row["id"]): list(row["gold"]) for row in rows}
    ids = sorted(golds)
    if set(ids) != set(base_pred) or set(ids) != set(ft_pred):
        missing = (set(ids) - set(base_pred)) | (set(ids) - set(ft_pred))
        raise ValueError(f"prediction coverage mismatch; missing ids: {sorted(missing)[:5]}")
    group_list = sorted(groups)
    for gid in group_list:
        for qid in groups[gid]:
            if qid not in golds:
                raise ValueError(f"group {gid} references unknown question {qid}")

    def macro(pred: dict[str, list[str]], drawn: list[str]) -> float:
        return statistics.mean(question_recall(pred[q], golds[q]) for q in drawn)

    point = macro(ft_pred, ids) - macro(base_pred, ids)
    per_q_delta = {qid: question_recall(ft_pred[qid], golds[qid])
                   - question_recall(base_pred[qid], golds[qid]) for qid in ids}
    gains = sum(1 for v in per_q_delta.values() if v > 0)
    losses = sum(1 for v in per_q_delta.values() if v < 0)

    rng = random.Random(seed)
    deltas = []
    for _ in range(resamples):
        drawn_groups = [group_list[rng.randrange(len(group_list))] for _ in group_list]
        drawn = [qid for gid in drawn_groups for qid in groups[gid]]
        if not drawn:
            continue
        deltas.append(macro(ft_pred, drawn) - macro(base_pred, drawn))
    deltas.sort()
    if len(deltas) >= 2:
        lo = deltas[int(0.025 * len(deltas))]
        hi = deltas[min(int(0.975 * len(deltas)), len(deltas) - 1)]
    else:
        lo = hi = point
    return BootstrapResult(
        n_questions=len(ids), n_groups=len(group_list), point_delta=point,
        ci95=[lo, hi], gains=gains, ties=len(ids) - gains - losses, losses=losses,
        seed=seed, resamples=len(deltas), sample_deltas=deltas,
    )
