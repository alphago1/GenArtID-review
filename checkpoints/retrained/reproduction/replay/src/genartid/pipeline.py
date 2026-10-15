"""Training/generation pipeline shared logic (CPU-importable, torch-free).

This module holds what the pipeline entry scripts share so it can be
unit-tested on CPU without torch/transformers:

- ``TRAIN_SYSTEM_PROMPT`` and the explicit TRL-format decision: we PRE-RENDER
  the chat template ourselves and hand TRL plain ``prompt``/``completion``
  strings (exactly what the historical notebooks did).  TRL treats plain
  string columns and conversational datasets differently; we deliberately
  choose the pre-rendered string format and verify token boundaries ourselves
  (prompt ids are a strict prefix of the full sequence; supervised completion
  tokens > 0).
- ``render_training_prompt`` needs only a tokenizer protocol object exposing
  ``apply_chat_template``; tests inject a stub.
- ``token_audit`` works on pre-rendered rows and an ``encode_fn`` callable;
  tests inject a deterministic fake, the runner passes the real tokenizer.
- ``plan_training`` turns a ratio dataset plus the frozen budget into an
  executable plan (steps, checkpoints, save cadence) and refuses budget
  inconsistencies against the ablation matrix.

Nothing here imports torch; GPU-only code lives inside guarded entry-script
functions.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

# Historical notebook prompt (kept for reference/parity only); the ACTIVE
# prompt lives in configs/research_protocol.json::train_system_prompt (G7
# change: neutralized to cover both directions; see protocol note).
TRAIN_SYSTEM_PROMPT_HISTORICAL = (
    "/no_think\n你是一个专业的中国法律背诵助手。请严格根据用户要求输出法条内容，不要添加解释。"
)
TRAIN_SYSTEM_PROMPT = TRAIN_SYSTEM_PROMPT_HISTORICAL


def load_train_system_prompt(protocol: dict | None = None) -> str:
    """Active training system prompt: protocol config wins, else the module
    default.  Rendered runs must record WHICH prompt string was used."""
    if protocol and isinstance(protocol.get("train_system_prompt"), str) \
            and protocol["train_system_prompt"].strip():
        return protocol["train_system_prompt"]
    return TRAIN_SYSTEM_PROMPT

# Frozen training budget (mirrors configs/ablation_matrix.json; entry scripts
# cross-check both sources and refuse on divergence).
BUDGET = {
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "effective_batch": 16,
    # G9 single source of truth: optimizer steps are an EXPLICIT budget, not
    # an epochs formula.  385 = ceil(1228/16) batches/epoch x 5 epochs;
    # 1575 = ceil(5040/16) x 5.  The earlier int(slots*5/16) formula computed
    # 383 for cpl and disagreed with the matrix (GPT finding) — retired.
    "max_steps": {"cpl": 385, "civil": 1575, "criminal_law": 635},
    "num_train_epochs": 5.0,          # provenance note only; max_steps governs
    "lora_r": 32, "lora_alpha": 64, "lora_dropout": 0.05,
    "learning_rate": 1e-4, "lr_scheduler": "cosine", "warmup_ratio": 0.03,
    "save_steps": {"cpl": 10, "civil": 40, "criminal_law": 20},
    "training_seeds": [20260910, 20260911, 20260912],
}
DOMAIN_SLOTS = {"civil": 5040, "cpl": 1228, "criminal_law": 2020}   # 4 slots x articles (1260/307/505)


class PipelineError(RuntimeError):
    """Raised on any pipeline precondition failure; nothing is executed."""


def strip_template_auto_prefix(tokenizer, rendered: str) -> str:
    """Remove template-rendered special-prefix tokens that the tokenizer ALSO
    auto-adds with add_special_tokens=True (B20, GLM-4).

    GLM-4's chat template renders '[gMASK]<sop>' AND its tokenizer prepends the
    same pair by default, so TRL's default tokenization of the pre-rendered
    string duplicates them.  Stripping the template copy makes the tokenizer's
    auto-prefix the single source; tokenizers that add nothing (Qwen) are
    no-ops.
    """
    try:
        probe = "x"
        no_sp = tokenizer.encode(probe, add_special_tokens=False)
        with_sp = tokenizer.encode(probe)
        extra = with_sp[:len(with_sp) - len(no_sp)]
        if not extra:
            return rendered
        prefix_text = tokenizer.decode(extra)
    except Exception:
        return rendered
    if prefix_text and rendered.startswith(prefix_text):
        return rendered[len(prefix_text):]
    return rendered


def render_training_prompt(tokenizer, user_text: str, system_prompt: str | None = None) -> str:
    """Pre-render the chat template for one training sample.

    ``tokenizer`` needs only ``apply_chat_template``.  ``system_prompt``
    defaults to the module constant; callers that loaded the protocol config
    should pass ``load_train_system_prompt(protocol)`` so the rendered text
    matches the frozen configuration exactly.
    """
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt or TRAIN_SYSTEM_PROMPT},
         {"role": "user", "content": user_text}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return strip_template_auto_prefix(tokenizer, rendered)


def template_kwargs(profile: dict) -> dict:
    """Chat-template kwargs declared by a profile's chat_template section.

    Every key except the human-facing ``note`` is passed through verbatim to
    ``apply_chat_template``.  The Qwen profile carries
    ``enable_thinking: false``; the GLM-4-9B-0414 profile carries only
    ``add_generation_prompt`` because its official template has no thinking
    branch — passing model-invalid template flags is a profile error, not a
    runtime default.
    """
    section = profile.get("chat_template")
    if not isinstance(section, dict):
        raise PipelineError("profile has no chat_template section")
    kwargs = {k: v for k, v in section.items() if k != "note"}
    if "add_generation_prompt" not in kwargs:
        raise PipelineError("chat_template section must declare add_generation_prompt")
    return kwargs


def render_eval_prompt(tokenizer, profile: dict, question: str) -> str:
    """Pre-render ONE evaluation prompt under the profile's template flags.

    Single rendering source for every Base/FT generation entry.  Applies
    ``strip_template_auto_prefix`` so tokenizers that auto-add the template's
    special prefix (GLM-4's [gMASK]<sop>) keep exactly one copy; tokenizers
    that add nothing (Qwen) are unaffected.
    """
    rendered = tokenizer.apply_chat_template(
        [{"role": "system",
          "content": profile["prompt"]["system_prompt_verbatim_notebook_top5"]},
         {"role": "user",
          "content": profile["prompt"]["user_template"].format(question=question)}],
        tokenize=False, **template_kwargs(profile))
    return strip_template_auto_prefix(tokenizer, rendered)


def generation_eos(profile: dict, tokenizer) -> int | list[int]:
    """Inference stop ids for a profile: the explicit multi-EOS list when the
    profile registers one (GLM-4-9B-0414: [151329, 151336, 151338] from
    generation_config.json), else the tokenizer's own single eos id (legacy
    Qwen profiles).  A single-id GenerationConfig is a protocol violation for
    profiles that register a stop set."""
    ids = profile.get("tokens", {}).get("inference_eos_token_ids")
    if isinstance(ids, list) and ids and all(isinstance(i, int) for i in ids):
        return list(ids)
    return tokenizer.eos_token_id


@dataclass
class TokenAudit:
    samples: int
    prompt_tokens_total: int
    supervised_tokens_total: int
    max_full_tokens: int
    truncated: int                      # full sequence longer than cap
    zero_supervision: int               # supervised tokens == 0 at cap
    prefix_mismatches: int              # prompt ids not a prefix of full ids
    per_mode: dict

    def as_dict(self) -> dict:
        return {
            "samples": self.samples,
            "prompt_tokens_total": self.prompt_tokens_total,
            "supervised_tokens_total": self.supervised_tokens_total,
            "max_full_tokens": self.max_full_tokens,
            "truncated": self.truncated,
            "zero_supervision": self.zero_supervision,
            "prefix_mismatches": self.prefix_mismatches,
            "per_mode": self.per_mode,
        }


def token_audit(rows: list[dict], encode_fn, eos_token: str, cap: int) -> TokenAudit:
    """Token/label-boundary audit over PRE-RENDERED rows.

    Each row: {"prompt": rendered_prompt, "completion": raw_completion,
    "mode": "F"|"R"|mask_variant}.  ``encode_fn(text) -> list[int]`` must add
    NO special tokens: the rendered prompt already contains template tokens
    and the completion is appended raw, mirroring the notebooks'
    ``completion = out + eos_token`` construction.
    """
    per_mode: dict[str, dict] = {}
    totals = dict(prompt=0, supervised=0, max_full=0, truncated=0,
                  zero_supervision=0, prefix_mismatch=0, n=0)
    for row in rows:
        prompt_ids = list(encode_fn(row["prompt"]))
        full_ids = list(encode_fn(row["prompt"] + row["completion"] + eos_token))
        prefix_ok = full_ids[:len(prompt_ids)] == prompt_ids
        supervised = max(0, min(cap, len(full_ids)) - len(prompt_ids))
        stats = per_mode.setdefault(row["mode"], dict(
            prompt=0, supervised=0, max_full=0, truncated=0,
            zero_supervision=0, prefix_mismatch=0, n=0))
        for bucket in (totals, stats):
            bucket["n"] += 1
            bucket["prompt"] += len(prompt_ids)
            bucket["supervised"] += supervised
            bucket["max_full"] = max(bucket["max_full"], len(full_ids))
            bucket["truncated"] += len(full_ids) > cap
            bucket["zero_supervision"] += supervised == 0
            bucket["prefix_mismatch"] += not prefix_ok
    return TokenAudit(
        samples=totals["n"], prompt_tokens_total=totals["prompt"],
        supervised_tokens_total=totals["supervised"],
        max_full_tokens=totals["max_full"], truncated=totals["truncated"],
        zero_supervision=totals["zero_supervision"],
        prefix_mismatches=totals["prefix_mismatch"],
        per_mode=per_mode,
    )


def plan_training(domain: str, dataset: dict, seed: int,
                  matrix: dict | None = None) -> dict:
    """Executable training plan from a ratio dataset + frozen budget.

    Refuses on: domain mismatch, seed not in the pre-registered list, slot
    count inconsistent with the budget rule, and divergence between this
    module's BUDGET and the ablation matrix (when supplied).
    """
    if domain not in DOMAIN_SLOTS:
        raise PipelineError(f"unknown domain {domain}")
    if seed not in BUDGET["training_seeds"]:
        raise PipelineError(f"seed {seed} is not in the pre-registered list "
                            f"{BUDGET['training_seeds']}")
    if matrix is not None:
        tf = matrix["shared_controls"]["training_fixed"]
        for key in ("per_device_train_batch_size", "gradient_accumulation_steps",
                    "max_steps", "learning_rate", "lora_r", "lora_alpha"):
            if tf.get(key) != BUDGET.get(key):
                raise PipelineError(
                    f"budget divergence between pipeline.BUDGET and ablation "
                    f"matrix on {key}: {BUDGET.get(key)} vs {tf.get(key)}; "
                    "refusing to train under an unfrozen budget")
    samples = len(dataset["samples"])
    expected_slots = DOMAIN_SLOTS[domain]
    if samples != expected_slots:
        raise PipelineError(
            f"{domain}: dataset has {samples} samples, budget expects "
            f"{expected_slots} slots (4 x articles); refusing")
    eff = BUDGET["effective_batch"]
    steps = BUDGET["max_steps"][domain]      # explicit budget (G9)
    save_steps = BUDGET["save_steps"][domain]
    # candidate checkpoints: cadence saves + the final step when it is not on
    # the cadence grid; unique global_step only
    cadence = set(range(save_steps, steps + 1, save_steps))
    cadence.add(steps)
    checkpoints = len(cadence)
    batches_per_epoch = -(-samples // eff)   # ceil
    return {
        "domain": domain,
        "ratio_label": dataset["stats"]["ratio_label"],
        "samples": samples,
        "seed": seed,
        "effective_batch": eff,
        "optimizer_steps": steps,
        "steps_note": f"explicit max_steps budget; = ceil({samples}/{eff}) batches/epoch x 5 epochs = {batches_per_epoch * 5}",
        "save_steps": save_steps,
        "expected_checkpoints": checkpoints,
        "candidate_steps": sorted(cadence),
        "dataset_sha256": hashlib.sha256(json.dumps(
            dataset, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest(),
        "trl_format": "pre-rendered prompt/completion strings (template applied by "
                      "render_training_prompt; TRL string branch, completion_only_loss "
                      "expected True for prompt-completion data — verified at runtime "
                      "from the collator)",
        "train_system_prompt": TRAIN_SYSTEM_PROMPT,
    }


def select_smoke_rows(dataset: dict, n: int) -> list[dict]:
    """Deterministic mixed-direction subset for smoke/pilot training checks.

    The dataset is stored F-block-then-R-block; a naive ``samples[:n]`` slice
    is single-direction (regression-tested).  This takes half from each
    direction and always includes the shortest and longest completion of
    each, so short, long and mixed padding cases all reach the batch.
    """
    if n < 4:
        raise PipelineError("select_smoke_rows needs n >= 4")
    samples = dataset["samples"]
    f = [s for s in samples if s["mode"] == "F"]
    r = [s for s in samples if s["mode"] == "R"]
    if not f or not r:
        raise PipelineError("smoke rows need both directions present")

    def pick(rows: list[dict], k: int) -> list[dict]:
        by_len = sorted(rows, key=lambda s: len(s["completion"]))
        chosen = [by_len[0], by_len[-1]]            # shortest + longest
        rest = [s for s in rows if s not in chosen]
        half = max(0, k - len(chosen))
        chosen += rest[:half]
        return chosen

    per = (n + 1) // 2
    return pick(f, per) + pick(r, n - per)


def stable_query_seed(seed: int, sample_id) -> int:
    """Per-question sampling seed (same construction as the reference runner)."""
    encoded = json.dumps([seed, str(sample_id)], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:4], "big")


def load_profile_and_protocol(profile_path, protocol_path) -> tuple[dict, dict]:
    from pathlib import Path
    profile = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    return profile, protocol


def trainer_recipe(domain: str, out_dir: str, seed: int, max_steps: int | None = None,
                   max_length: int = 1024, train_steps_log: bool = False,
                   target_modules: list | str | None = None) -> dict:
    """Single source for the LoRA + SFTConfig construction (G7 item 6).

    Both the training entry (f_train_sft.py) and the GPU smoke
    (i_gpu_smoke.py) MUST build their trainer from this recipe so there is
    exactly one place where LoRA targets/rank/alpha or optimizer settings
    live.  Returns plain kwargs; the caller constructs LoraConfig/SFTConfig
    (GPU-env imports stay out of this torch-free module).  Any key here is a
    frozen budget item — changing it changes the experiment protocol.

    ``target_modules`` defaults to the frozen q/v/o set; the ONLY sanctioned
    override is the pre-registered all-linear ablation (PEFT accepts the
    string "all-linear"), recorded in the run plan.
    """
    if domain not in DOMAIN_SLOTS:
        raise PipelineError(f"unknown domain {domain}")
    steps = max_steps if max_steps is not None else BUDGET["max_steps"][domain]
    # transformers 5.x removed warmup_ratio from TrainingArguments (B15);
    # the frozen 0.03 ratio converts to explicit steps exactly the way the
    # historical scheduler did: ceil(ratio * num_training_steps).
    warmup_steps = math.ceil(BUDGET["warmup_ratio"] * steps)
    return {
        "lora": {"r": BUDGET["lora_r"], "lora_alpha": BUDGET["lora_alpha"],
                 "lora_dropout": BUDGET["lora_dropout"], "bias": "none",
                 "task_type": "CAUSAL_LM",
                 "target_modules": target_modules or ["q_proj", "v_proj", "o_proj"]},
        "warmup_provenance": {
            "warmup_ratio_frozen": BUDGET["warmup_ratio"],
            "warmup_steps_resolved": warmup_steps,
            "max_steps_resolved": steps,
            "note": "ceil(0.03 x max_steps); transformers 5.17 accepts only warmup_steps",
        },
        "sft": {
            "output_dir": out_dir,
            "max_steps": steps,
            "per_device_train_batch_size": BUDGET["per_device_train_batch_size"],
            "gradient_accumulation_steps": BUDGET["gradient_accumulation_steps"],
            "learning_rate": BUDGET["learning_rate"],
            "lr_scheduler_type": BUDGET["lr_scheduler"],
            "warmup_steps": warmup_steps,
            "logging_steps": 1 if train_steps_log else 10,
            "save_strategy": "no" if max_steps is not None and max_steps <= 50 else "steps",
            "save_steps": BUDGET["save_steps"][domain],
            "bf16": True, "fp16": False,
            "max_length": max_length, "packing": False,
            "report_to": "none", "seed": seed, "data_seed": seed,
            "completion_only_loss": True,
        },
    }
