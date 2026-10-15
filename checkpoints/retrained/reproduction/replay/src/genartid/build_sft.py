"""Corpus-only SFT data construction.  TRAINING DATA BOUNDARY:

Every builder function in this module accepts a loaded corpus (list[Article])
and a config dict ONLY.  There is deliberately no parameter, import or path
that could admit eval questions, cases, or any non-corpus text.  Mixing real
or synthetic cases into training data is out of scope by protocol.

Direction definitions are frozen (research_protocol.json):
- F = id -> full article text.
- R = full article text -> id.
Ratios are defined over SAMPLE COUNTS.  Equal-total-slot budget control: each
domain gets ``sample_slots_per_article`` (4) slots per article, so the five
ratios (1,0),(3,1),(1,1),(1,3),(0,1) all split 4n slots into integers.  This is
a data budget, NOT an epoch count.  Within a direction, slots cycle
deterministically over articles in ascending number order, so coverage and
per-article repetition counts are exactly reportable.
"""
from __future__ import annotations

import hashlib
import json
import re
import statistics
from dataclasses import dataclass

from .corpus import Article
from .identifiers import to_cn

RATIOS_F_R = [(1, 0), (3, 1), (1, 1), (1, 3), (0, 1)]
SLOTS_PER_ARTICLE = 4
DATA_SEED = 20260910
MASK_PLACEHOLDER = "[[MASK]]"
MASK_FRACTIONS = (0.15, 0.30)
MASK_TARGETS = ("full_reconstruction", "span_only")

LAW_FULL_NAME = {"civil": "《中华人民共和国民法典》", "cpl": "《中华人民共和国刑事诉讼法》",
                 "criminal_law": "《中华人民共和国刑法》"}

F_PROMPT_TEMPLATE = "请准确复述{law}第{num_cn}条{part}的完整内容。"
R_PROMPT_TEMPLATE = (
    "以下是{law}的一条条文原文，请给出这条条文的编号，"
    "只输出形如{law}第X条的编号，不要输出原文或任何解释。\n\n{content}"
)
MASK_PROMPT_TEMPLATE = (
    "下面是{law}第{num_cn}条的一部分内容，其中一段已被{placeholder}替换。"
    "请恢复{law}第{num_cn}条的完整原文，不要添加解释。\n\n{masked}"
)
MASK_SPAN_PROMPT_TEMPLATE = (
    "下面是{law}第{num_cn}条的一部分内容，其中一段已被{placeholder}替换。"
    "请只输出被{placeholder}替换的那段原文，不要输出其他内容。\n\n{masked}"
)


def _stable_offset(seed: int, key: str, modulus: int) -> int:
    digest = hashlib.sha256(json.dumps([seed, key]).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % max(modulus, 1)


def _percentiles(values: list[int]) -> dict:
    if not values:
        return {"min": 0, "p50": 0, "p90": 0, "max": 0}
    s = sorted(values)
    return {"min": s[0], "p50": s[len(s) // 2], "p90": s[int(0.9 * (len(s) - 1))], "max": s[-1]}


@dataclass
class SftSample:
    sample_id: str
    domain: str
    mode: str
    ratio: str
    variant: str
    slot_index: int
    source_article_ids: list[str]
    source_hash: str
    prompt: str
    completion: str
    seed: int

    def as_dict(self) -> dict:
        return {
            "sample_id": self.sample_id, "domain": self.domain, "mode": self.mode,
            "ratio": self.ratio, "variant": self.variant, "slot_index": self.slot_index,
            "source_article_ids": self.source_article_ids, "source_hash": self.source_hash,
            "prompt": self.prompt, "completion": self.completion, "seed": self.seed,
            "prompt_chars": len(self.prompt), "completion_chars": len(self.completion),
        }


def f_sample(domain: str, article: Article, ratio: str, slot: int) -> SftSample:
    law = LAW_FULL_NAME[domain]
    prompt = F_PROMPT_TEMPLATE.format(law=law, num_cn=to_cn(article.number),
                                  part=article.part)
    return SftSample(
        sample_id=f"{domain}|{ratio}|F|{slot:05d}", domain=domain, mode="F",
        ratio=ratio, variant="baseline", slot_index=slot,
        source_article_ids=[article.canon_id], source_hash=article.source_hash,
        prompt=prompt, completion=article.repaired_content, seed=DATA_SEED)


def r_sample(domain: str, article: Article, ratio: str, slot: int) -> SftSample:
    law = LAW_FULL_NAME[domain]
    prompt = R_PROMPT_TEMPLATE.format(law=law, content=article.repaired_content)
    completion = f"{law}第{to_cn(article.number)}条{article.part}"
    return SftSample(
        sample_id=f"{domain}|{ratio}|R|{slot:05d}", domain=domain, mode="R",
        ratio=ratio, variant="baseline", slot_index=slot,
        source_article_ids=[article.canon_id], source_hash=article.source_hash,
        prompt=prompt, completion=completion, seed=DATA_SEED)


def build_ratio_dataset(domain: str, articles: list[Article], ratio: tuple[int, int],
                        slots_per_article: int = SLOTS_PER_ARTICLE,
                        seed: int = DATA_SEED) -> dict:
    """Build one (F, R) ratio dataset with the equal-total-slot budget."""
    a, b = ratio
    if (a, b) not in RATIOS_F_R:
        raise ValueError(f"ratio {ratio} is not in the frozen grid {RATIOS_F_R}")
    if not articles:
        raise ValueError("empty corpus")
    total = slots_per_article * len(articles)
    denom = a + b
    if total * a % denom or total * b % denom:
        raise ValueError(f"ratio {ratio} does not split {total} slots into integers")
    f_slots, r_slots = total * a // denom, total * b // denom
    ordered = sorted(articles, key=lambda x: x.number)
    samples: list[SftSample] = []
    ratio_label = f"{a}f_{b}r"
    for i in range(f_slots):
        samples.append(f_sample(domain, ordered[i % len(ordered)], ratio_label, i))
    for i in range(r_slots):
        samples.append(r_sample(domain, ordered[i % len(ordered)], ratio_label, i))

    def coverage(mode: str) -> dict:
        used = [s for s in samples if s.mode == mode]
        per_article: dict[str, int] = {}
        for s in used:
            per_article[s.source_article_ids[0]] = per_article.get(s.source_article_ids[0], 0) + 1
        reps = list(per_article.values()) or [0]
        return {"samples": len(used),
                "unique_articles_covered": len(per_article),
                "repeats_per_article": {"min": min(reps), "max": max(reps)}}

    stats = {
        "domain": domain, "ratio_f_r": list(ratio), "ratio_label": ratio_label,
        "slots_per_article": slots_per_article, "total_slots": total,
        "f_slots": f_slots, "r_slots": r_slots, "seed": seed,
        "articles_count": len(articles),
        "f_coverage": coverage("F"), "r_coverage": coverage("R"),
        "prompt_chars": _percentiles([len(s.prompt) for s in samples]),
        "completion_chars": _percentiles([len(s.completion) for s in samples]),
        "token_stats": "pending_gpu_gate (no verified local tokenizer in this workspace)",
        "supervised_token_note": "sample-count ratio is NOT a supervised-token or gradient-share measurement",
    }
    return {"schema": "genartid-corpus-only-sft/1", "stats": stats,
            "samples": [s.as_dict() for s in samples]}


def build_masked_dataset(domain: str, articles: list[Article], fraction: float,
                         target: str, seed: int = DATA_SEED) -> dict:
    """Masked-span reconstruction prototype; experimental, never mixed into ratios.

    One contiguous span of the article content is replaced by a visible
    placeholder.  ``target`` selects the supervision: "full_reconstruction"
    (complete original text) or "span_only" (just the masked span).  These are
    two different experiments and must not be pooled.  The placeholder is an
    ordinary visible token; a real masked-token run needs tokenizer/label
    checks at the GPU gate.
    """
    if fraction not in MASK_FRACTIONS:
        raise ValueError(f"fraction {fraction} not in {MASK_FRACTIONS}")
    if target not in MASK_TARGETS:
        raise ValueError(f"target {target} not in {MASK_TARGETS}")
    law = LAW_FULL_NAME[domain]
    ordered = sorted(articles, key=lambda x: x.number)
    samples: list[SftSample] = []
    for article in ordered:
        content = article.repaired_content
        span_len = max(1, round(len(content) * fraction))
        offset = _stable_offset(seed, f"mask|{domain}|{article.canon_id}", len(content) - span_len + 1)
        span = content[offset:offset + span_len]
        masked = content[:offset] + MASK_PLACEHOLDER + content[offset + span_len:]
        num_cn = to_cn(article.number)
        variant = f"mask{fraction}_{target}"
        if target == "full_reconstruction":
            prompt = MASK_PROMPT_TEMPLATE.format(
                law=law, num_cn=num_cn, part=article.part,
                    placeholder=MASK_PLACEHOLDER, masked=masked)
            completion = content
        else:
            prompt = MASK_SPAN_PROMPT_TEMPLATE.format(
                law=law, num_cn=num_cn, part=article.part,
                    placeholder=MASK_PLACEHOLDER, masked=masked)
            completion = span
        samples.append(SftSample(
            sample_id=f"{domain}|{variant}|{article.canon_id}", domain=domain,
            mode="masked_span_reconstruction", ratio="none", variant=variant,
            slot_index=article.index, source_article_ids=[article.canon_id],
            source_hash=article.source_hash, prompt=prompt, completion=completion,
            seed=seed))
    return {"schema": "genartid-corpus-only-sft-mask/1",
            "stats": {"domain": domain, "fraction": fraction, "target": target,
                      "samples": len(samples), "seed": seed,
                      "placeholder": MASK_PLACEHOLDER,
                      "note": "experimental CPU prototype; not a validated training recipe"},
            "samples": [s.as_dict() for s in samples]}


def verify_mask_restorable(dataset: dict, articles: list[Article]) -> dict:
    """Round-trip check for a masked dataset against the source corpus.

    span_only:  pre + completion + post must equal the original content.
    full_reconstruction: completion must equal the original content exactly.
    The masked body is located by the exact template prefix (content may
    itself contain newlines, so positional splitting is not safe).
    """
    by_id = {a.canon_id: a for a in articles}
    domain = dataset["stats"]["domain"]
    law = LAW_FULL_NAME[domain]
    target = dataset["stats"]["target"]
    checked, failures = 0, []
    for s in dataset["samples"]:
        article = by_id[s["source_article_ids"][0]]
        template = (MASK_PROMPT_TEMPLATE if target == "full_reconstruction"
                    else MASK_SPAN_PROMPT_TEMPLATE)
        prefix = template.format(law=law, num_cn=to_cn(article.number), part=article.part,
                                 placeholder=MASK_PLACEHOLDER, masked="")
        if not s["prompt"].startswith(prefix):
            failures.append({"sample_id": s["sample_id"], "reason": "prompt prefix mismatch"})
            continue
        masked_body = s["prompt"][len(prefix):]
        if MASK_PLACEHOLDER not in masked_body:
            failures.append({"sample_id": s["sample_id"], "reason": "placeholder missing"})
            continue
        pre, post = masked_body.split(MASK_PLACEHOLDER, 1)
        if target == "span_only":
            restored = pre + s["completion"] + post
        else:
            restored = s["completion"]
        checked += 1
        if restored != article.repaired_content:
            failures.append({"sample_id": s["sample_id"], "reason": "restoration mismatch"})
    return {"checked": checked, "target": target, "failed_count": len(failures),
            "failures": failures[:10]}


def clause_survey(domain: str, articles: list[Article]) -> dict:
    """Survey of sentence/paragraph -> id candidates.  NOT training data.

    Splits repaired content at sentence boundaries, keeps exact-substring
    fragments, and records uniqueness within the domain corpus plus
    provenance (article id + character span).  Short fragments (<=15 chars)
    are marked as legacy-short (a historical factor), not promoted to the
    clean R task.  Ambiguous multi-source fragments are reported, never
    silently assigned.
    """
    norm = {a.canon_id: a.normalized_content for a in articles}
    rows: list[dict] = []
    ambiguous = 0
    for article in sorted(articles, key=lambda x: x.number):
        text = article.repaired_content
        pieces = [p.strip() for p in re.split(r"(?<=[。；！？\n])", text) if p.strip()]
        # Merge consecutive short pieces into paragraphs when separated by newline.
        offset = 0
        for piece in pieces:
            start = text.find(piece, offset)
            offset = start + len(piece)
            needle = re.sub(r"\s+", "", piece)
            if not needle:
                continue
            sources = [aid for aid, nc in norm.items() if needle in nc]
            entry = {
                "fragment": piece, "chars": len(needle),
                "source_article_id": article.canon_id,
                "span": [start, start + len(piece)],
                "n_sources": len(sources), "unique_source": len(sources) == 1,
                "legacy_short": len(needle) <= 15,
            }
            if len(sources) > 1:
                ambiguous += 1
                entry["also_in"] = [a for a in sources if a != article.canon_id][:5]
            rows.append(entry)
    unique_long = [r for r in rows if r["unique_source"] and not r["legacy_short"]]
    return {
        "schema": "genartid-clause-survey/1", "domain": domain,
        "note": "survey only; no training data is emitted from this function",
        "totals": {
            "fragments": len(rows),
            "unique_and_long": len(unique_long),
            "unique_but_short_legacy": sum(r["unique_source"] and r["legacy_short"] for r in rows),
            "ambiguous_multi_source": ambiguous,
            "articles_covered_by_unique_long": len(set(
                r["source_article_id"] for r in unique_long)),
        },
        "fragment_chars": _percentiles([r["chars"] for r in rows]),
        "fragments": rows,
    }
