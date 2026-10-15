"""Deterministic dev/test split over question groups.

Frozen rules (research_protocol.json "split"):
- seed 20260910, used exactly once; per domain ~50:50 by GROUPS.
- Group key: whitespace-normalized question text (identical normalized text
  forms one group; different ids with identical text are reported).
- Semantic near-duplicate grouping is NOT performed (no embedding model in
  the CPU boundary); this limitation is recorded in the manifest.
- Groups are sorted by their key hash, then assigned alternately to reach a
  deterministic near-50:50 split (sorted order + seeded Fisher-Yates via
  random.Random(seed) is the exact algorithm; any change is a protocol
  change).  The dev group count is ``int(n * 0.5 + 0.5)``: explicit half-up
  rounding, so dev receives the extra group when the group count is odd
  (Python's built-in round() would apply banker's rounding, which is
  deterministic but surprising; this rule is equally deterministic and
  spelled out).
- Both gold views (full_gold_all / domain_gold_nonempty) are derived later at
  scoring time; the split never drops questions by gold or prediction quality.
- All existing questions are historically exposed (they were used in past
  experiments); every row is marked exposure_status="historically_exposed",
  making this a retrospective internal holdout, never a fresh blind test.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

SPLIT_SEED = 20260910
DEV_FRACTION = 0.5


def normalized_question(text: str) -> str:
    return re.sub(r"\s+", "", str(text))


def group_key_hash(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


@dataclass
class Question:
    qid: str
    raw_question: str
    group_key: str
    gold: list[str]
    side: str = ""
    law_field: str | None = None


def load_questions(path: Path, domain: str, canon) -> list[Question]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path.name}: expected non-empty list")
    questions: list[Question] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{path.name}[{index}]: not an object")
        qid = str(row.get("id", index))
        if qid in seen:
            raise ValueError(f"{path.name}[{index}]: duplicate id {qid}")
        seen.add(qid)
        question = row.get("question")
        gold = row.get("articles")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{path.name}[{index}]: missing question text")
        if not isinstance(gold, list) or not gold:
            raise ValueError(f"{path.name}[{index}]: missing gold")
        questions.append(Question(
            qid=qid, raw_question=question,
            group_key=normalized_question(question),
            gold=[canon(a) for a in gold if canon(a)],
            law_field=row.get("law"),
        ))
    return questions


def build_split(questions: list[Question], domain: str, seed: int = SPLIT_SEED,
                dev_fraction: float = DEV_FRACTION) -> dict:
    """Freeze one dev/test split; returns a manifest structure (no scoring)."""
    groups: dict[str, list[Question]] = {}
    for q in questions:
        groups.setdefault(q.group_key, []).append(q)
    keys = sorted(groups)                       # deterministic base order
    order = list(range(len(keys)))
    random.Random(seed).shuffle(order)         # seeded Fisher-Yates on group order
    n_dev = int(len(keys) * dev_fraction + 0.5)  # half-up; dev gets the odd group
    sides: dict[int, str] = {}
    for position, key_index in enumerate(order):
        sides[key_index] = "dev" if position < n_dev else "test"
    rows, duplicate_group_rows = [], 0
    for key_index, key in enumerate(keys):
        members = groups[key]
        if len(members) > 1:
            duplicate_group_rows += len(members)
        for q in members:
            q.side = sides[key_index]
            rows.append(q)
    dev_ids = sorted(q.qid for q in rows if q.side == "dev")
    test_ids = sorted(q.qid for q in rows if q.side == "test")
    from .scoring import LAWS          # single source of the law prefixes
    law_prefix = LAWS[domain] + "第"   # (B27: no silent cpl fallback)
    manifest = {
        "schema": "genartid-split/1",
        "domain": domain,
        "seed": seed,
        "dev_fraction": dev_fraction,
        "algorithm": "sorted group keys -> seeded Fisher-Yates shuffle -> first int(n*0.5 + 0.5) groups to dev (half-up; dev receives the extra group on odd counts)",
        "group_key": "whitespace_normalized_question_text",
        "near_duplicate_policy": "exact normalized-text groups only; semantic near-duplicates NOT grouped (CPU boundary, no embedding model); limitation recorded",
        "historical_exposure": "all questions were used in past experiments; this is a retrospective internal holdout",
        "counts": {
            "questions": len(rows), "groups": len(keys),
            "dev_questions": len(dev_ids), "test_questions": len(test_ids),
            "questions_in_multi_id_groups": duplicate_group_rows,
            "rows_with_domain_gold": sum(any(g.startswith(law_prefix) for g in q.gold)
                                         for q in rows),
        },
        "dev_ids": dev_ids, "test_ids": test_ids,
        "questions": [{
            "qid": q.qid, "side": q.side, "gold": q.gold,
            "question_sha256": hashlib.sha256(q.raw_question.encode("utf-8")).hexdigest(),
            "group_key_sha256": group_key_hash(q.group_key),
            "group_size": len(groups[q.group_key]),
            "exposure_status": "historically_exposed",
        } for q in sorted(rows, key=lambda x: x.qid)],
    }
    return manifest


def split_manifest_hash(manifest: dict) -> str:
    """Stable fingerprint over the split IDENTITY, recomputable from content.

    Binds: schema/domain/seed/dev_fraction, the full dev/test membership, AND
    every question's (qid, side, gold, question hash, group hash).  Editing
    membership or gold after freezing therefore changes the recomputed hash,
    even if the stored ``split_manifest_hash`` field is left untouched
    (regression-tested).  The stored hash field itself is not part of the
    identity, so this function detects tampering rather than ignoring it.
    """
    identity = {
        "schema": manifest["schema"], "domain": manifest["domain"],
        "seed": manifest["seed"], "dev_fraction": manifest["dev_fraction"],
        "dev_ids": manifest["dev_ids"], "test_ids": manifest["test_ids"],
        "questions": [{
            "qid": q["qid"], "side": q["side"], "gold": q["gold"],
            "question_sha256": q["question_sha256"],
            "group_key_sha256": q["group_key_sha256"],
        } for q in sorted(manifest["questions"], key=lambda x: x["qid"])],
    }
    return hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
