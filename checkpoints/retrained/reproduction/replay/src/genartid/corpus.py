"""Strict corpus loading with per-source newline repair and normalization.

Frozen rules:
- Article names must parse under the legacy identifier grammar; on failure the
  loader RAISES.  Guessing an article number from the row index (as
  reference/source/data_scripts/prepare_stage1_data.py:40-43 does) is
  forbidden and would silently corrupt ID bindings.
- ``raw_content`` is the verbatim file text.
- ``repaired_content`` applies exactly one explicit, per-source repair rule:
  for cpl the two-character sequence backslash-n is replaced with a real
  newline (the cpl snapshot carries this literal in all 307 contents); for
  civil the repair is a no-op (the civil snapshot contains no newlines and
  no literal backslash-n).  No generic unicode unescaping is performed
  anywhere (it would corrupt Chinese characters).
- ``normalized_content`` additionally removes all whitespace, mirroring the
  reference audit's ``whitespace_normalized``; used only for matching and
  ambiguity checks, never as a training target.
- Duplicate canonical IDs and empty contents are rejected.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .identifiers import canon_fixed, parse_identifier_fixed, part_ordinal

DOMAIN_FILES = {
    "civil": ("civil_code_articles.json", "《民法典》", "《中华人民共和国民法典》"),
    "cpl": ("cpl_articles.json", "《刑事诉讼法》", "《中华人民共和国刑事诉讼法》"),
    # 刑法 carries amendment-inserted 之N articles (452 main + 53 parts = 505)
    "criminal_law": ("criminal_law_articles.json", "《刑法》", "《中华人民共和国刑法》"),
}


@dataclass
class Article:
    index: int                    # 0-based position in the source file
    canon_id: str                 # 《Law》第N条 (Arabic N; 之N suffix preserved verbatim)
    raw_name: str                 # verbatim name field
    number: int                   # parsed article number
    raw_content: str              # verbatim content field
    repaired_content: str         # after the per-source repair rule
    normalized_content: str       # whitespace-stripped, for matching only
    source_hash: str              # sha256 of raw_name + "\x00" + raw_content
    part: str = ""                # 之N suffix ("之一") for amendment-inserted articles

    @property
    def full_name(self) -> str:
        law = self.canon_id.split("第")[0]
        return self.canon_id.replace(law, "《中华人民共和国" + law[1:], 1)

    def surface_name(self) -> str:
        """Historical surface form: 《中华人民共和国民法典》第一条 (Chinese numerals)."""
        from .identifiers import to_cn
        law = self.canon_id.split("第")[0]
        return f"《中华人民共和国{law[1:]}第{to_cn(self.number)}条{self.part}"

    @property
    def order_key(self) -> tuple[int, int]:
        """(main number, 之N ordinal) — unique ascending order within a corpus."""
        return (self.number, part_ordinal(self.part))


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", "", text)


def load_corpus(path: Path, domain: str) -> list[Article]:
    """Load one domain corpus; raises on any structural anomaly."""
    if domain not in DOMAIN_FILES:
        raise ValueError(f"unknown domain: {domain}")
    law_canon = DOMAIN_FILES[domain][1]
    # No filename-equality guard: the per-row law-name check below is the
    # semantic guard.  A filename check would make the loader refuse ad-hoc
    # fixture files before the structural checks could report the real reason.
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path.name}: expected a non-empty JSON list")
    articles: list[Article] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"{path.name}[{index}]: not an object")
        unknown = set(item) - {"name", "content", "id"}
        if unknown:
            raise ValueError(f"{path.name}[{index}]: unknown fields {sorted(unknown)}")
        name, content = item.get("name"), item.get("content")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{path.name}[{index}]: missing/empty name")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"{path.name}[{index}]: missing/empty content")
        parsed = parse_identifier_fixed(name)
        if parsed is None or parsed[0] != law_canon:
            raise ValueError(
                f"{path.name}[{index}]: name {name!r} does not parse as a {law_canon} id; "
                "index-guessing is forbidden")
        canon_id = canon_fixed(name)
        if canon_id in seen:
            raise ValueError(f"{path.name}[{index}]: duplicate canonical id {canon_id}")
        seen.add(canon_id)
        repaired = content.replace("\\n", "\n") if domain in ("cpl", "criminal_law") else content
        source_hash = hashlib.sha256(
            (name + "\x00" + content).encode("utf-8")).hexdigest()
        articles.append(Article(
            index=index, canon_id=canon_id, raw_name=name, number=parsed[1],
            raw_content=content, repaired_content=repaired,
            normalized_content=_normalize_ws(repaired.replace("\\n", "\n")),
            source_hash=source_hash,
            part=parsed[2] or "",
        ))
    keys = [a.order_key for a in articles]
    if keys != sorted(keys) or len(set(keys)) != len(keys):
        raise ValueError(f"{path.name}: article (number, part) keys are not unique and "
                         "ascending (gaps are allowed, duplicates and jumbled order "
                         "are not)")
    return articles


def corpus_report(articles: list[Article], domain: str) -> dict:
    """Countable facts about a loaded corpus, for the data manifest."""
    import statistics
    lengths = [len(a.repaired_content) for a in articles]
    return {
        "domain": domain,
        "article_count": len(articles),
        "canon_id_first": articles[0].canon_id,
        "canon_id_last": articles[-1].canon_id,
        "content_chars": {"min": min(lengths), "median": statistics.median(lengths),
                           "max": max(lengths), "sum": sum(lengths)},
        "literal_backslash_n_repaired": domain == "cpl",
        "tokenizer": "pending_gpu_gate",
    }
