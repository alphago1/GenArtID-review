"""Retrieval output parsers.

``parse_legacy`` reproduces the notebooks' ``extract_articles`` verbatim,
including its known defects (JSON-then-rescan duplication, duplicates
consuming the top-5 window, first-``{.*}`` JSON selection).  It exists only
for historical parity checks; do not use it for new scoring.

``parse_fixed`` is the revision-2 parser.  Frozen rules:

1. Final-answer region: everything after the last reasoning block
   (``</think>`` or an explicit ``思考过程``/``分析`` header block) when one
   is present; otherwise the whole raw output.
2. JSON extraction: all balanced top-level JSON values (``{...}`` objects
   and ``[...]`` arrays) in the final-answer region are candidates; the LAST
   valid JSON value is the answer (a model may show examples earlier and the
   final answer last).  An earlier valid value is never merged with a later
   one.  A valid JSON value that is not a dict (e.g. a bare array) is a
   wrong-shape answer: predictions are EMPTY with status ``json_not_dict``
   and the regex fallback is NOT taken.
3. Valid JSON with an ``articles`` list: candidates are the string entries
   matching the fixed identifier grammar, in array order, capped at the
   first 5 slots.  Non-string entries and non-matching strings are recorded
   as invalid slots and do NOT enter the list; they also do not shrink the
   window of later entries (slot = position in the array, not in the
   filtered list).  The raw text is never re-scanned for extra mentions.
4. Valid JSON without a usable ``articles`` list (missing, non-list, or
   zero valid entries): predictions are EMPTY; no fallback refill.  The
   status records the exact reason.
5. Malformed JSON (no balanced valid object): deterministic fallback scans
   the final-answer region with the fixed grammar, order-preserving, first 5
   slots.  This mirrors the legacy fallback but never mixes JSON and scan
   results.
6. Dedup happens AFTER the 5-slot cap, preserving order.  Duplicates and
   invalid entries occupy their slots; the top-5 window is never expanded
   and nothing after slot 5 is ever pulled in.
7. ``第第``-style doubling fails the grammar and is dropped (no silent
   repair).  Cross-law IDs are kept verbatim (never merged by number).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from ga_identifiers import ART_RE, ART_RE_FIXED, canon, canon_fixed

TOPK = 5
_RE_LEGACY = re.compile(ART_RE)
_RE_FIXED = re.compile(ART_RE_FIXED)
_THINK_END = re.compile(r"</think\s*>")


def final_answer_region(raw: str) -> tuple[str, str | None]:
    """Split reasoning from the final answer.  Returns (region, kind).

    kind is "after_think_block" when a reasoning block is present, else None.
    Only explicit markers are honoured; no guessing at verbose preambles.
    """
    marker = _THINK_END.search(raw)
    if marker:
        return raw[marker.end():], "after_think_block"
    return raw, None


def _balanced_json_values(text: str) -> list[str]:
    """All balanced top-level ``{...}`` / ``[...]`` substrings (string-literal
    aware), in order of appearance.  Brackets nested inside a value do not
    start a new candidate."""
    values: list[str] = []
    stack: list[str] = []
    start = -1
    in_string = False
    escape = False
    for index, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            if not stack:
                start = index
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
                if not stack and start >= 0:
                    values.append(text[start:index + 1])
                    start = -1
    return values


def parse_legacy(raw: str, topk: int = TOPK) -> list[str]:
    """Verbatim notebook semantics; see module docstring for the defect list."""
    mentions: list[str] = []
    m = re.search(r"\{.*\}", raw, re.S)
    if m:
        try:
            obj = json.loads(m.group())
            if isinstance(obj, dict) and isinstance(obj.get("articles"), list):
                for item in obj["articles"]:
                    if isinstance(item, str):
                        mentions.extend(_RE_LEGACY.findall(item))
        except (ValueError, TypeError):
            pass
    if not mentions:
        mentions = _RE_LEGACY.findall(raw)[:topk]
    else:
        for m2 in _RE_LEGACY.findall(raw):
            if len(mentions) >= topk:
                break
            mentions.append(m2)
    return [canon(x) for x in mentions[:topk] if canon(x)]


@dataclass
class ParseFixed:
    predictions: list[str]
    status: str                   # ok | partial | invalid | json_no_articles | json_invalid_entries | json_not_dict
    region_kind: str | None
    json_used: bool
    json_error: str | None = None
    invalid_slots: list[str] = field(default_factory=list)
    duplicate_slots: list[int] = field(default_factory=list)
    raw_candidates: list[str] = field(default_factory=list)

    def __iter__(self):
        """Allow ``pred, status = parse_fixed(...)`` style unpacking."""
        return iter((self.predictions, self.status))


def parse_fixed(raw: str, topk: int = TOPK) -> ParseFixed:
    """Revision-2 parser; rules frozen in the module docstring."""
    if not isinstance(raw, str):
        raise TypeError(f"raw output must be str, got {type(raw).__name__}")
    region, kind = final_answer_region(raw)
    invalid: list[str] = []
    predictions: list[str] = []
    status = "invalid"
    json_used = False
    json_error: str | None = None

    for candidate in reversed(_balanced_json_values(region)):
        try:
            value = json.loads(candidate)
        except (ValueError, TypeError) as error:
            json_error = f"unparseable value: {error}"
            continue                   # last value invalid -> try the previous one
        json_used = True               # the LAST valid JSON value is the answer
        if not isinstance(value, dict):
            status = "json_not_dict"   # wrong shape: empty, and no scan fallback
            break
        articles = value.get("articles")
        if not isinstance(articles, list):
            status = "json_no_articles"
            break
        slots = 0
        for entry in articles:
            if slots >= topk:
                break
            if not isinstance(entry, str):
                invalid.append(repr(entry))
                slots += 1
                continue
            match = _RE_FIXED.search(entry)
            if match:
                predictions.append(canon_fixed(match.group()))
            else:
                invalid.append(entry if len(entry) <= 60 else entry[:57] + "...")
            slots += 1
        if not predictions:
            status = "json_invalid_entries" if invalid else "json_no_articles"
        else:
            status = "ok" if len(predictions) >= topk else "partial"
        break

    if not json_used:
        matches = _RE_FIXED.findall(region)
        predictions = [canon_fixed(x) for x in matches[:topk]]
        status = "ok" if len(predictions) >= topk else ("partial" if predictions else "invalid")

    seen: set[str] = set()
    deduped: list[str] = []
    duplicates: list[int] = []
    for index, item in enumerate(predictions):
        if item in seen:
            duplicates.append(index)
            continue
        seen.add(item)
        deduped.append(item)
    return ParseFixed(
        predictions=deduped, status=status, region_kind=kind, json_used=json_used,
        json_error=json_error, invalid_slots=invalid, duplicate_slots=duplicates,
        raw_candidates=predictions,
    )
