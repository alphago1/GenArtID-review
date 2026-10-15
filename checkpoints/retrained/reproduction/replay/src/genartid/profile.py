"""Unified Base/FT evaluation profile loading and CPU validation.

CPU-validated here: profile structure, greedy/sampling separation, explicit
sampling parameters, and the pure equality logic that will compare a Base run
config with an FT run config at the GPU gate (identical in every must-match
field; only the adapter path may differ).

NOT claimed here: that any model output has been produced or verified under
this profile.  That is a GPU-gate item.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = ROOT / "configs" / "base_ft_eval_profile.json"

MUST_MATCH_FIELDS = [
    "system_prompt", "user_template", "chat_template", "decode", "global_seed",
    "max_new_tokens", "eos_token_id", "pad_token_id", "parser_version",
]


def load_profile(path: Path | None = None) -> dict:
    return json.loads((path or PROFILE_PATH).read_text(encoding="utf-8"))


def validate_profile(profile: dict) -> list[str]:
    """Structural checks; returns a list of problems (empty when valid)."""
    problems: list[str] = []
    if profile.get("schema") != "genartid-baseft-profile/1":
        problems.append("unexpected schema version")
    decode = profile.get("decode", {})
    greedy = decode.get("greedy")
    if not isinstance(greedy, dict):
        problems.append("missing greedy profile")
    else:
        if greedy.get("do_sample") is not False:
            problems.append("greedy must set do_sample=false")
        for key in ("temperature", "top_p", "top_k"):
            if greedy.get(key) is not None:
                problems.append(f"greedy must not set {key}")
    sample = decode.get("sample_explicit")
    if not isinstance(sample, dict):
        problems.append("missing sample_explicit profile")
    else:
        if sample.get("do_sample") is not True:
            problems.append("sample_explicit must set do_sample=true")
        for key in ("temperature", "top_p", "top_k"):
            value = sample.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                problems.append(f"sample_explicit must set explicit numeric {key}")
        if sample.get("temperature") is None or sample["temperature"] <= 0:
            problems.append("sample temperature must be > 0")
        if not (0 < float(sample.get("top_p", 0)) <= 1):
            problems.append("sample top_p must be in (0,1]")
        if not isinstance(sample.get("top_k"), int) or sample["top_k"] < 0:
            problems.append("sample top_k must be an int >= 0")
        if not sample.get("seeds"):
            problems.append("sampling must pre-register at least one seed")
        for variant in (greedy, sample):
            if variant.get("repetition_penalty") is None:
                problems.append("repetition_penalty must be explicit in both variants")
            if not isinstance(variant.get("max_new_tokens"), int) or variant["max_new_tokens"] < 1:
                problems.append("max_new_tokens must be a positive int in both variants")
    if "system_prompt_verbatim_notebook_top5" not in profile.get("prompt", {}):
        problems.append("system prompt missing")
    if not profile.get("base_ft_must_match"):
        problems.append("must-match field list missing")
    if not profile.get("gpu_gates_pending"):
        problems.append("gpu gate list missing")
    precision = profile.get("precision", {})
    if "is_bf16_supported" not in json.dumps(precision, ensure_ascii=False):
        problems.append("precision section must require an explicit bf16 support check")
    return problems


def render_run_config(profile: dict, variant: str, global_seed: int,
                      adapter: str | None) -> dict:
    """The exact per-run comparison unit at the GPU gate."""
    if variant not in profile["decode"]:
        raise ValueError(f"unknown decode variant: {variant}")
    decode = profile["decode"][variant]
    return {
        "system_prompt": profile["prompt"]["system_prompt_verbatim_notebook_top5"],
        "user_template": profile["prompt"]["user_template"],
        "chat_template": profile["chat_template"],
        "decode": {k: v for k, v in decode.items() if k != "seeds"},
        "global_seed": global_seed,
        "max_new_tokens": decode["max_new_tokens"],
        "eos_token_id": profile["tokens"]["eos"],
        "pad_token_id": profile["tokens"]["pad"],
        "parser_version": profile["parser"]["scoring_parser"],
        "adapter": adapter,
    }


def check_base_ft_match(base_config: dict, ft_config: dict) -> list[str]:
    """Pure equality check over the must-match fields; adapter may differ."""
    problems = []
    for field in MUST_MATCH_FIELDS:
        if field not in base_config or field not in ft_config:
            problems.append(f"missing field {field} in one of the configs")
        elif base_config[field] != ft_config[field]:
            problems.append(f"field {field} differs between Base and FT")
    allowed = {"adapter"}
    for key in set(base_config) | set(ft_config):
        if key not in MUST_MATCH_FIELDS and key not in allowed:
            problems.append(f"unexpected field {key}; add it to the must-match list or remove it")
    return problems
