"""Dev-only checkpoint selection and gated test evaluation.

Frozen rules (research_protocol.json "selection"):
- Primary metric: macro Recall@5 on the dev partition, full_gold_all view.
- Highest score wins; exact ties resolve to the EARLIEST global_step; a
  candidate without an integer global_step is REJECTED (no implicit ranking).
- The selector receives dev scores only.  It has no test input, by
  construction: its signature and dataclasses cannot carry test rows.
- The test gate refuses to run unless a frozen selection artifact exists and
  its protocol fingerprint matches the current protocol configuration, and
  refuses any Base/FT coverage or gold mismatch (every question is one
  statistical unit; missing questions are not silently dropped).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from ga_scoring import (BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED,
                      paired_group_bootstrap, question_recall, summarize_views)


@dataclass
class Candidate:
    run_id: str          # unique, e.g. "ckpt-180"
    global_step: int     # required; reject when unknown
    dev_recall_at_5: float

    def key(self):
        return (-self.dev_recall_at_5, self.global_step, self.run_id)


def select_checkpoint(candidates: list[Candidate]) -> Candidate:
    """Pure dev-side selection; ties -> earliest global_step -> run_id order."""
    if not candidates:
        raise ValueError("no candidates provided")
    for c in candidates:
        if not isinstance(c.global_step, int) or isinstance(c.global_step, bool):
            raise ValueError(f"candidate {c.run_id}: global_step must be an integer, "
                             "refusing to rank a candidate with unknown step")
    return sorted(candidates, key=Candidate.key)[0]


def protocol_fingerprint(protocol: dict, profile: dict | None = None) -> str:
    """Stable hash over EVERY evaluation-affecting piece of configuration.

    Covers the selection rules, the decode variants, AND (when the Base/FT
    profile is supplied) the full prompt / chat-template / decode / token
    configuration.  A selection made under one fingerprint must never be
    test-evaluated under another: changing e.g. a sampling temperature or the
    system prompt invalidates the selection (regression-tested).
    ``profile`` is required by run_test_evaluation; selection freezing should
    always pass it too.
    """
    material = {
        "metric": protocol.get("selection", {}).get("metric"),
        "view": protocol.get("selection", {}).get("view"),
        "tie_break": protocol.get("selection", {}).get("tie_break"),
        "topk": 5,
        "parser": protocol.get("parser_version"),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "decode_variants": protocol.get("decode_variants"),
        "protocol_content_sha256": _canonical_sha256(protocol),
    }
    if profile is not None:
        material["profile_content_sha256"] = _canonical_sha256(profile)
        material["profile_prompt"] = profile.get("prompt")
        material["profile_decode"] = profile.get("decode")
        material["profile_chat_template"] = profile.get("chat_template")
    return hashlib.sha256(
        json.dumps(material, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _canonical_sha256(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


@dataclass
class SelectionArtifact:
    domain: str
    selected_run_id: str
    selected_global_step: int
    dev_score: float
    base_run_id: str
    base_dev_score: float
    candidates: list[dict]
    split_manifest_hash: str
    protocol_fingerprint: str
    selected_identity: dict | None = None   # G8: prediction-file identity block
    base_identity: dict | None = None

    def as_dict(self) -> dict:
        data = {
            "schema": "genartid-selection/1", "domain": self.domain,
            "selected_run_id": self.selected_run_id,
            "selected_global_step": self.selected_global_step,
            "dev_macro_recall_at_5": self.dev_score,
            "base_run_id": self.base_run_id,
            "base_dev_macro_recall_at_5": self.base_dev_score,
            "candidates": self.candidates,
            "split_manifest_hash": self.split_manifest_hash,
            "protocol_fingerprint": self.protocol_fingerprint,
        }
        if self.selected_identity is not None:
            data["selected_identity"] = self.selected_identity
        if self.base_identity is not None:
            data["base_identity"] = self.base_identity
        return data


def make_selection(domain: str, candidates: list[Candidate], base: Candidate,
                   split_manifest_hash: str, protocol: dict,
                   profile: dict | None = None,
                   selected_identity: dict | None = None,
                   base_identity: dict | None = None) -> SelectionArtifact:
    best = select_checkpoint(candidates)
    return SelectionArtifact(
        domain=domain, selected_run_id=best.run_id,
        selected_global_step=best.global_step, dev_score=best.dev_recall_at_5,
        base_run_id=base.run_id, base_dev_score=base.dev_recall_at_5,
        candidates=[{"run_id": c.run_id, "global_step": c.global_step,
                     "dev_recall_at_5": c.dev_recall_at_5} for c in candidates],
        split_manifest_hash=split_manifest_hash,
        protocol_fingerprint=protocol_fingerprint(protocol, profile),
        selected_identity=selected_identity, base_identity=base_identity,
    )


def freeze_selection(artifact: SelectionArtifact, path) -> None:
    """Write the selection artifact with a tamper-evident self-hash."""
    data = artifact.as_dict()
    blob = json.dumps(data, ensure_ascii=False, sort_keys=True).encode("utf-8")
    data["selection_sha256"] = hashlib.sha256(blob).hexdigest()
    from pathlib import Path as _P
    out = _P(path)
    out.parent.mkdir(parents=True, exist_ok=True)   # B23: glm0414 layout uses
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2),  # a fresh dir
                   encoding="utf-8", newline="\n")


def load_frozen_selection(path) -> dict:
    """Load and verify a frozen selection artifact; raises on tamper."""
    from pathlib import Path as _P
    data = json.loads(_P(path).read_text(encoding="utf-8"))
    stored = data.pop("selection_sha256", None)
    blob = json.dumps(data, ensure_ascii=False, sort_keys=True).encode("utf-8")
    if stored != hashlib.sha256(blob).hexdigest():
        raise TestGateError("selection artifact hash mismatch (tampered or edited)")
    data["selection_sha256"] = stored
    return data


def verify_gold_match(split_manifest: dict, base_gold: dict[str, list[str]],
                      ft_gold: dict[str, list[str]]) -> None:
    """Base/FT prediction files must carry gold identical to the split manifest."""
    test_ids = set(split_manifest["test_ids"])
    manifest_gold = {q["qid"]: set(q["gold"]) for q in split_manifest["questions"]
                     if q["side"] == "test"}
    for name, gold in (("base", base_gold), ("ft", ft_gold)):
        if set(gold) != test_ids:
            missing = sorted(test_ids - set(gold))[:5]
            extra = sorted(set(gold) - test_ids)[:5]
            raise TestGateError(f"{name} gold coverage mismatch; missing={missing} extra={extra}")
        for qid in sorted(test_ids):
            if set(gold[qid]) != manifest_gold[qid]:
                raise TestGateError(
                    f"{name} gold mismatch for question {qid}: file={sorted(gold[qid])} "
                    f"manifest={sorted(manifest_gold[qid])}; refusing to score")


class TestGateError(RuntimeError):
    """Raised when the test evaluation precondition fails; nothing is scored."""


def run_test_evaluation(selection: dict, protocol: dict,
                        split_manifest: dict, rows: list[dict],
                        base_pred: dict[str, list[str]],
                        ft_pred: dict[str, list[str]],
                        profile: dict | None = None,
                        base_pred_file: dict | None = None,
                        ft_pred_file: dict | None = None,
                        bootstrap_resamples: int = BOOTSTRAP_RESAMPLES) -> dict:
    """Gated test evaluation of the locked FT vs its matched Base.

    ``selection``: a selection artifact in as_dict() form (frozen on disk).
    ``rows``: dicts with qid/gold for the TEST partition only.
    ``profile``: the Base/FT evaluation profile (configs/base_ft_eval_profile.json);
    REQUIRED — the fingerprint binds prompt/decode/token configuration, so a
    selection cannot be evaluated under different decoding.
    Refuses on: fingerprint mismatch (including any decode/prompt/profile
    change), split CONTENT tampering (the manifest hash is RECOMPUTED from the
    file's membership + gold + question hashes, not trusted from a stored
    field), split/selection hash mismatch, coverage gaps, duplicate rows, gold
    mismatch (via verify_gold_match at the entry point), and any selected-run
    identity problem.
    """
    from ga_split import split_manifest_hash as recompute_split_hash
    if profile is None:
        raise TestGateError("the Base/FT evaluation profile is required; the "
                            "protocol fingerprint binds prompt/decode/token config")
    for field in ("domain", "selected_run_id", "selected_global_step",
                  "split_manifest_hash", "protocol_fingerprint"):
        if field not in selection:
            raise TestGateError(f"selection artifact missing field {field}")
    if selection["protocol_fingerprint"] != protocol_fingerprint(protocol, profile):
        raise TestGateError("protocol fingerprint mismatch: selection was made under "
                            "a different protocol/decode/prompt configuration; "
                            "refusing to evaluate")
    stored_hash = split_manifest.get("split_manifest_hash", "")
    recomputed = recompute_split_hash(split_manifest)
    if stored_hash and recomputed != stored_hash:
        raise TestGateError("split manifest content does not match its own stored "
                            "hash (membership/gold tampered after freezing); "
                            "refusing to evaluate")
    if stored_hash != selection["split_manifest_hash"]:
        raise TestGateError("split manifest hash does not match the one the selection was "
                            "made under; refusing to evaluate")
    if split_manifest.get("domain") != selection["domain"]:
        raise TestGateError("split/selection domain mismatch")

    test_ids = set(split_manifest["test_ids"])
    provided = [str(r["qid"]) for r in rows]
    if len(provided) != len(set(provided)):
        dupes = {q for q in provided if provided.count(q) > 1}
        raise TestGateError(f"duplicate test rows: {sorted(dupes)[:5]}")
    if set(provided) != test_ids:
        missing = sorted(test_ids - set(provided))[:5]
        extra = sorted(set(provided) - test_ids)[:5]
        raise TestGateError(f"test coverage mismatch; missing={missing} extra={extra}")
    for pred, name in ((base_pred, "base"), (ft_pred, "ft")):
        if set(pred) != test_ids:
            missing = sorted(test_ids - set(pred))[:5]
            raise TestGateError(f"{name} prediction coverage mismatch; missing {missing}")
    golds = {str(r["qid"]): [str(g) for g in r["gold"]] for r in rows}

    # G8: the files being scored must be the SAME runs the selection froze —
    # same decode, same model/tokenizer, and (FT) the same adapter fingerprint
    sel_id = selection.get("selected_identity")
    base_id = selection.get("base_identity")
    if (sel_id and ft_pred_file is not None) or (base_id and base_pred_file is not None):
        def identity_of(payload: dict) -> dict:
            return {k: payload.get(k) for k in
                    ("decode", "decode_params", "seed", "identity")}
        if sel_id is not None:
            want = identity_of(sel_id)
            got = {"decode": ft_pred_file.get("decode"),
                   "decode_params": ft_pred_file.get("decode_params"),
                   "seed": ft_pred_file.get("seed"),
                   "identity": ft_pred_file.get("identity")}
            if got != want:
                raise TestGateError("FT prediction identity differs from the frozen "
                                    "selection (decode/model/adapter); refusing")
        if base_id is not None:
            want = identity_of(base_id)
            got = {"decode": base_pred_file.get("decode"),
                   "decode_params": base_pred_file.get("decode_params"),
                   "seed": base_pred_file.get("seed"),
                   "identity": base_pred_file.get("identity")}
            if got != want:
                raise TestGateError("Base prediction identity differs from the "
                                    "frozen selection; refusing")

    eval_rows = [{"id": qid, "gold": golds[qid],
                  "pred": ft_pred[qid]} for qid in sorted(test_ids)]
    base_rows = [{"id": qid, "gold": golds[qid],
                  "pred": base_pred[qid]} for qid in sorted(test_ids)]
    ft_summary = summarize_views(eval_rows, selection["domain"])
    base_summary = summarize_views(base_rows, selection["domain"])

    groups: dict[str, list[str]] = {}
    for q in split_manifest["questions"]:
        if q["side"] == "test":
            groups.setdefault(q["group_key_sha256"], []).append(q["qid"])

    bootstrap = paired_group_bootstrap(
        rows=[{"id": qid, "gold": golds[qid]} for qid in sorted(test_ids)],
        groups=groups, base_pred=base_pred, ft_pred=ft_pred,
        resamples=bootstrap_resamples)
    point = ft_summary["full_gold_all"]["R@5"] - base_summary["full_gold_all"]["R@5"]
    return {
        "schema": "genartid-test-eval/1",
        "domain": selection["domain"],
        "selected_run_id": selection["selected_run_id"],
        "selected_global_step": selection["selected_global_step"],
        "n_test_questions": len(test_ids),
        "base_full_gold_all": base_summary["full_gold_all"],
        "ft_full_gold_all": ft_summary["full_gold_all"],
        "base_domain_gold_nonempty": base_summary["domain_gold_nonempty"],
        "ft_domain_gold_nonempty": ft_summary["domain_gold_nonempty"],
        "point_delta_ft_minus_base_full_gold_all": point,
        "bootstrap": bootstrap.as_dict(),
        "interpretation_rule": "CI crossing zero is NOT evidence of no effect; a positive "
                               "point estimate is NOT a confirmed gain. Each question is one "
                               "statistical unit.",
    }
