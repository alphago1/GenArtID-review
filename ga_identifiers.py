"""Canonical Chinese article identifiers.

c2i / to_cn / canon are verbatim semantic ports of the notebook helpers
(``c2i``, ``canon`` in cplTrainAndEval.ipynb cell 3 / civilTrainAndEval.ipynb
cell 6, and reference/source/id_memory_intervention/common.py).  Behaviour of
these helpers is frozen: any change is a protocol change, not a bug fix.

The identifier grammar is extended (only in the *fixed* parser path, see
parsing.py) with an optional 之N article-part suffix used by other Chinese
statutes (e.g. 《中华人民共和国刑法》第二百六十六条之一).  The legacy
grammar deliberately does not know this suffix.
"""
from __future__ import annotations

import re

CN_M = {"零": 0, "〇": 0, "○": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
        "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
CN_U = {"十": 10, "百": 100, "千": 1000, "万": 10000}

# Legacy grammar (verbatim from notebooks): no 之N suffix support.
ART_RE = r"《.+?》第[0-9０-９一二三四五六七八九十百千万零〇○两]+条"
ID_RE = re.compile(r"^(《.+?》)第([0-9０-９一二三四五六七八九十百千万零〇○两]+)条$")

# Fixed grammar (protocol revision 2): optional 之N part suffix.
SUFFIX_RE = r"(?:之[一二三四五六七八九十百零〇○]+)?"
ART_RE_FIXED = rf"《.+?》第[0-9０-９一二三四五六七八九十百千万零〇○两]+条{SUFFIX_RE}"
ID_RE_FIXED = re.compile(
    rf"^(《.+?》)第([0-9０-９一二三四五六七八九十百千万零〇○两]+)条(之[一二三四五六七八九十百零〇○]+)?$")


def c2i(text: str) -> int | None:
    """Chinese/Arabic numeral string -> int; None when unparseable.

    Verbatim port of the notebook function; do not "improve".
    """
    if not text:
        return None
    text = str(text).strip().translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    if text.isdigit():
        return int(text)
    total = section = number = 0
    for ch in text:
        if ch in CN_M:
            number = CN_M[ch]
        elif ch in CN_U:
            unit = CN_U[ch]
            if unit == 10000:
                section = (section + (number or 0)) * unit
                total += section
                section = number = 0
            else:
                if number == 0:
                    number = 1
                section += number * unit
                number = 0
        else:
            return None
    return total + section + number


_CN_DIGITS = "零一二三四五六七八九"
_CN_UNITS = ["", "十", "百", "千"]


def to_cn(n: int) -> str:
    """int -> standard Chinese numeral for 1 <= n <= 9999 (corpus surface style)."""
    if not 1 <= n <= 9999:
        raise ValueError(f"to_cn supports 1..9999, got {n}")
    digits = [(n // 1000 % 10, "千"), (n // 100 % 10, "百"), (n // 10 % 10, "十"), (n % 10, "")]
    parts: list[str] = []
    emitted = False
    pending_zero = False
    for d, unit in digits:
        if d == 0:
            if emitted:
                pending_zero = True
            continue
        if pending_zero:
            parts.append("零")
            pending_zero = False
        if unit == "十" and d == 1 and not emitted:
            parts.append("十")
        else:
            parts.append(_CN_DIGITS[d] + unit)
        emitted = True
    return "".join(parts)


def canon(art: str) -> str:
    """Canonical article identifier: 《Law》第N条 with Arabic N, no spaces.

    Verbatim port of the notebook function; do not "improve".
    """
    if not isinstance(art, str):
        return ""
    x = art.strip()
    x = re.sub(r"\s+", "", x)
    x = x.replace("（", "(").replace("）", ")")
    x = x.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    m = ID_RE.match(x)
    if not m:
        return x
    law = re.sub(r"^《中华人民共和国", "《", m.group(1))
    number = c2i(m.group(2))
    if number is None:
        return f"{law}第{m.group(2)}条"
    return f"{law}第{number}条"


def canon_fixed(art: str) -> str:
    """Canonical identifier under the fixed grammar (with 之N suffix).

    The main article number is normalized to Arabic (same as ``canon``); an
    optional 之N part suffix is preserved verbatim (e.g. 《刑法》第266条之一),
    because the suffix is an ordinal marker, not a plain number.  Falls back
    to ``canon`` semantics for strings outside the fixed grammar.
    """
    if not isinstance(art, str):
        return ""
    x = art.strip()
    x = re.sub(r"\s+", "", x)
    x = x.replace("（", "(").replace("）", ")")
    x = x.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    m = ID_RE_FIXED.match(x)
    if not m:
        return x
    law = re.sub(r"^《中华人民共和国", "《", m.group(1))
    number = c2i(m.group(2))
    if number is None:
        return f"{law}第{m.group(2)}条"
    if m.group(3) is None:
        return f"{law}第{number}条"
    return f"{law}第{number}条{m.group(3)}"


def parse_identifier(text: str) -> tuple[str, int] | None:
    """Parse a canonical identifier into (law, number); None if not parseable.

    Uses the legacy grammar (no 之N), matching the reference common.py.
    """
    m = ID_RE.match(re.sub(r"\s+", "", text.strip()))
    if not m:
        return None
    law = re.sub(r"^《中华人民共和国", "《", m.group(1))
    number = c2i(m.group(2))
    if number is None:
        return None
    return law, number


def parse_identifier_fixed(text: str) -> tuple[str, int, str | None] | None:
    """Parse under the fixed grammar: (law, number, 之N suffix or None).

    ``criminal_law`` articles carry amendment-inserted 之N parts (e.g.
    《中华人民共和国刑法》第二百七十六条之一) which the legacy parser
    rejects.  ``part`` is the verbatim suffix ("之一") so callers can
    order by (number, part) and render full identifiers.
    """
    m = ID_RE_FIXED.match(re.sub(r"\s+", "", text.strip()))
    if not m:
        return None
    law = re.sub(r"^《中华人民共和国", "《", m.group(1))
    number = c2i(m.group(2))
    if number is None:
        return None
    return law, number, m.group(3)


def part_ordinal(part: str | None) -> int:
    """之N suffix -> ordinal for ordering (0 when absent); None-safe."""
    if not part:
        return 0
    value = c2i(part[1:]) if len(part) > 1 else None
    return value if value is not None else 0


def civil_number(text: str) -> int | None:
    parsed = parse_identifier(text)
    if parsed and parsed[0] == "《民法典》":
        return parsed[1]
    return None
