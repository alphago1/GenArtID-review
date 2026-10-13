"""Local (GPU-stage) execution for the multimodel memory-recall probe.

Everything here imports torch/transformers/peft LAZILY inside functions —
importing this module on a CPU-only box is safe and testable.  Every entry
point raises ``LocalResourceError`` with a PRECISE missing-item list when the
configured resources or hardware are absent; there are no unconditional
"blocked" exits — the gate is the resource/authorization state, not a stub.

Frozen protocol facts (configs/figure3/*, configs/glm0414/base_ft_eval_profile.json):
- GLM Base/FT: 4-bit NF4 + double quant, bf16 compute (hard fail without),
  adapter-only (never merged), official chat template without enable_thinking,
  multi-EOS [151329, 151336, 151338], pad 151329;
- memory encoder: shibing624/text2vec-base-chinese, K = 128 - special tokens,
  non-overlapping chunks keeping the full tail, attention-mask mean pooling,
  token-count weighted chunk average, whole-text L2 normalization;
- memory classes: Eff. >= 0.85 target-first, Dup. otherwise-other >= 0.85,
  Err. the rest; empty normalized output is Err.; comparison pool = the whole
  domain corpus; ties by first canonical id.
"""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import ga_memory_recall_probe as probe
from ga_corpus import Article

EOS_TOKEN_IDS = [151329, 151336, 151338]
PAD_TOKEN_ID = 151329
ENCODER_WINDOW_TOKENS = 128


class LocalResourceError(SystemExit):
    """Precise missing-resource report; never an unconditional stub exit."""

    def __init__(self, missing: list[str], stage: str):
        self.missing = missing
        super().__init__(
            f"[{stage}] local stage blocked by missing resources/authorization:\n  - "
            + "\n  - ".join(missing)
            + "\nProvide the paths/authorization above; other independent stages stay runnable.")


# ---------------------------------------------------------------------------
# resource resolution


def _dir_summary(path: Path) -> dict:
    out = {}
    for f in sorted(Path(path).iterdir()):
        if f.is_file():
            out[f.name] = (probe.sha256_file(f) if f.suffix in (".json", ".jinja",
                             ".txt", ".md") and f.stat().st_size < 2_000_000
                           else f"{f.stat().st_size}B")
    return out


@dataclass
class LocalResources:
    glm_base_path: str
    six_adapters: dict[str, str]          # f"{domain}:{seed}" -> adapter dir
    encoder_path: str
    checks: list[str] = field(default_factory=list)

    def missing_report(self) -> list[str]:
        return list(self.checks)


def resolve_local_resources(local_cfg: dict) -> LocalResources:
    """Validate the configured local paths; collect a precise missing list.

    ``local_cfg`` comes from configs/figure3/local_resources.json.  Nothing is
    loaded here — only existence/structure checks on disk.
    """
    missing: list[str] = []
    base = local_cfg.get("glm_base_path")
    if not base:
        missing.append("glm_base_path not configured (GLM-4-9B-0414 local dir)")
    elif not Path(base).is_dir():
        missing.append(f"glm_base_path does not exist: {base}")
    else:
        for required in ("config.json", "tokenizer.json", "tokenizer_config.json",
                         "chat_template.jinja"):
            if not (Path(base) / required).is_file():
                missing.append(f"glm_base_path lacks {required}: {base}")
        shards = list(Path(base).glob("*.safetensors"))
        if not shards:
            missing.append(f"glm_base_path has no *.safetensors weight shards: {base}")
    adapters = local_cfg.get("six_adapters") or {}
    wanted_keys = {f"{d}:{s}" for d in ("civil", "criminal_law")
                   for s in ("20260910", "20260911", "20260912")}
    for key in sorted(wanted_keys):
        path = adapters.get(key)
        if not path:
            missing.append(f"six_adapters[{key}] not configured")
        elif not Path(path).is_dir():
            missing.append(f"six_adapters[{key}] does not exist: {path}")
        elif not (Path(path) / "adapter_model.safetensors").is_file() \
                or not (Path(path) / "adapter_config.json").is_file():
            missing.append(f"six_adapters[{key}] lacks adapter files: {path}")
    encoder = local_cfg.get("encoder_path")
    if not encoder:
        missing.append("encoder_path not configured (text2vec-base-chinese local dir)")
    elif not Path(encoder).is_dir():
        missing.append(f"encoder_path does not exist: {encoder}")
    else:
        for required in ("config.json",):
            if not (Path(encoder) / required).is_file():
                missing.append(f"encoder_path lacks {required}: {encoder}")
    return LocalResources(glm_base_path=base or "", six_adapters=adapters,
                          encoder_path=encoder or "", checks=missing)


def require_resources(local_cfg: dict, stage: str, need_encoder: bool = False,
                      need_cuda: bool = False) -> LocalResources:
    res = resolve_local_resources(local_cfg)
    if need_encoder and not local_cfg.get("encoder_path"):
        pass  # already in missing list
    if need_cuda:
        ok, detail = cuda_bf16_status()
        if not ok:
            res.checks.append(f"CUDA/bf16 unavailable: {detail}")
    if res.checks:
        raise LocalResourceError(res.checks, stage)
    return res


def cuda_bf16_status() -> tuple[bool, str]:
    try:
        import torch
    except ImportError as error:
        return False, f"torch not importable ({error})"
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() is False"
    if not torch.cuda.is_bf16_supported():
        return False, "bf16 not supported on this GPU"
    return True, torch.cuda.get_device_name(0)


# ---------------------------------------------------------------------------
# tokenizer-only audit (CPU, no weights)


def reference_token_audit(base_path: str, articles_by_domain: dict[str, list[Article]],
                          recitation_cap_tokens: int) -> dict:
    """Token-length audit over all 1765 reference texts with the REAL GLM
    tokenizer.  Loads tokenizer files only — never model weights."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
    specials = tokenizer.num_special_tokens_to_add(pair=False)
    k = ENCODER_WINDOW_TOKENS - specials
    audit = {"schema": "figure3-local-token-audit/1", "tokenizer_dir": base_path,
             "encoder_chunk_k": k, "specials_per_encode": specials, "domains": {}}
    for domain, articles in articles_by_domain.items():
        lengths = []
        for article in articles:
            ids = tokenizer.encode(article.repaired_content, add_special_tokens=False)
            lengths.append(len(ids))
        lengths_sorted = sorted(lengths)
        n = len(lengths)
        lmax = lengths_sorted[-1]
        audit["domains"][domain] = {
            "n": n,
            "token_min": lengths_sorted[0],
            "token_median": lengths_sorted[n // 2],
            "token_max": lmax,
            "token_sum": sum(lengths),
            "lmax_plus_64": lmax + 64,
            "lmax_plus_64_gt_recitation_cap": lmax + 64 > recitation_cap_tokens,
            "encoder_chunks_for_longest": len(probe.chunk_windows(lmax, k)),
        }
    return audit


# ---------------------------------------------------------------------------
# GLM generation (lazy torch; GPU stage)


def batch_plan(n_items: int, batch_size: int) -> list[tuple[int, int]]:
    """Fixed contiguous [start, end) batches; partial tail batch is kept and
    never padded with duplicate items."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    return [(start, min(start + batch_size, n_items))
            for start in range(0, n_items, batch_size)]


def chunk_token_counts(windows: list[tuple[int, int]]) -> list[int]:
    """Per-window real-token counts, derived eagerly per window.

    Regression: an inline generator ``hi - lo for _, hi in windows`` written
    inside the batching loop captured the LOOP variable ``lo`` (late binding)
    and yielded a 0 first count for every multi-chunk text.  Keep the
    derivation self-contained and unit-test it.
    """
    return [hi - lo for lo, hi in windows]


def _load_glm(base_path: str, adapter_path: str | None):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        base_path, quantization_config=bnb, device_map={"": 0},
        torch_dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.eval()
    return model, tokenizer


def set_adapter(model, adapter_path: str | None):
    """Switch the active LoRA adapter on a persistent base model.

    ``None`` disables all adapters (Base mode).  The first adapter wraps the
    plain model once; later adapters reuse the ``active`` slot via
    load_adapter + set_adapter.  Adapter-only: never merged.
    """
    from peft import PeftModel
    if adapter_path is None:
        if isinstance(model, PeftModel):
            return model  # caller uses disable_adapter() context for Base rows
        return model
    if not isinstance(model, PeftModel):
        return PeftModel.from_pretrained(model, adapter_path, adapter_name="active",
                                         is_trainable=False)
    model.load_adapter(adapter_path, adapter_name="active", is_trainable=False)
    model.set_adapter("active")
    model.eval()
    return model


def adapter_activation_check(model, tokenizer, sample_prompt: str) -> dict:
    """GPU gate from the profile: adapter on/off next-token logits must both
    be finite and differ."""
    import torch
    from peft import PeftModel
    encoded = tokenizer(sample_prompt, return_tensors="pt").to("cuda:0")

    def first_logits():
        with torch.inference_mode():
            out = model(**encoded)
        return out.logits[0, -1, :]

    if isinstance(model, PeftModel):
        with model.disable_adapter():
            base_logits = first_logits().float()
        adapter_logits = first_logits().float()
    else:
        base_logits = adapter_logits = first_logits().float()
    delta = (adapter_logits - base_logits).abs().max().item()
    finite = bool(torch.isfinite(base_logits).all() and torch.isfinite(adapter_logits).all())
    return {"finite": finite, "max_abs_logit_delta": delta,
            "adapter_active": bool(delta > 0)}


def generation_token_status(token_ids: list[int], max_new_tokens: int) -> dict:
    """Classify the actual first EOS, excluding later batch padding tokens."""
    for index, token_id in enumerate(token_ids):
        if token_id in EOS_TOKEN_IDS:
            return {"finish_reason": "stop", "length_stop": False,
                    "generated_tokens": index + 1, "stop_token_id": token_id}
    if len(token_ids) < max_new_tokens:
        raise ValueError("generation ended without a registered EOS before the cap")
    return {"finish_reason": "length", "length_stop": True,
            "generated_tokens": len(token_ids), "stop_token_id": None}


def validated_resume_rows(path: Path, run_identity: str, expected_ids: set[str]) -> list[dict]:
    """Reject foreign identities, duplicate successes and corrupt cached rows."""
    if not path.exists():
        return []
    rows, seen = [], set()
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as error:
            raise ValueError(f"{path.name}:{index}: invalid cached JSON") from error
        if row.get("run_identity") != run_identity:
            raise ValueError(f"{path.name}:{index}: cached run identity mismatch")
        item_id = row.get("item_id")
        if item_id not in expected_ids or item_id in seen:
            raise ValueError(f"{path.name}:{index}: unexpected/duplicate cached item")
        if not isinstance(row.get("raw_text"), str) or row.get("system_failure"):
            raise ValueError(f"{path.name}:{index}: invalid successful cached output")
        seen.add(item_id)
        rows.append(row)
    return rows


def generate_recitation_run(base_path: str, adapter_path: str | None,
                            articles: list[Article], out_path: Path,
                            run_identity: str, run_meta: dict,
                            profile: dict, max_new_tokens: int = 2048,
                            batch_size: int = 4, inference_seed: int = 20260910,
                            progress_every: int = 200,
                            preloaded: tuple | None = None,
                            only_items: set[str] | None = None) -> dict:
    """One local recitation run (Base or one seed adapter) over one domain.

    Resumes from ``out_path`` by job id; already-complete rows with the same
    run identity are kept, missing ids are regenerated in the original order.
    Journal rows are flushed after every batch; a partial batch recomputes
    its original members, committing only missing rows.  ``preloaded`` carries a persistent
    (model, tokenizer) so caller-side adapter swaps avoid reloads.
    ``only_items`` restricts generation to a subset (smoke) while resume
    coverage is still computed against that subset only.
    """
    import torch
    from transformers import GenerationConfig

    jobs = [{"item_id": a.canon_id, "prompt_user":
             probe.RECITATION_USER_TEMPLATE.format(full_name=a.raw_name)}
            for a in articles if only_items is None or a.canon_id in only_items]
    rows = validated_resume_rows(out_path, run_identity, {j["item_id"] for j in jobs})
    done = {r["item_id"] for r in rows}
    missing_jobs = [j for j in jobs if j["item_id"] not in done]

    template_kwargs = {k: v for k, v in profile.get("chat_template", {}).items()
                       if k != "note"}
    if "add_generation_prompt" not in template_kwargs:
        template_kwargs["add_generation_prompt"] = True
    law_display = run_meta["law_display"]
    system = probe.RECITATION_SYSTEM_TEMPLATE.format(law_display=law_display)
    generation = GenerationConfig(
        do_sample=False, num_beams=1, repetition_penalty=1.0,
        max_new_tokens=max_new_tokens, eos_token_id=list(EOS_TOKEN_IDS),
        pad_token_id=PAD_TOKEN_ID, use_cache=True)
    torch.manual_seed(inference_seed)

    model = None
    tokenizer = None
    own_load = preloaded is None
    try:
        if missing_jobs:
            if preloaded is not None:
                model, tokenizer = preloaded
            else:
                model, tokenizer = _load_glm(base_path, adapter_path)
                tokenizer.padding_side = "left"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        generated = 0
        for start, end in batch_plan(len(jobs), batch_size):
            batch = jobs[start:end]
            if all(j["item_id"] in done for j in batch):
                continue
            prompts = [tokenizer.apply_chat_template(
                [{"role": "system", "content": system},
                 {"role": "user", "content": j["prompt_user"]}],
                tokenize=False, **template_kwargs) for j in batch]
            enc = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda:0")
            base_context = (model.disable_adapter()
                            if adapter_path is None and hasattr(model, "disable_adapter")
                            else nullcontext())
            with base_context, torch.inference_mode():
                gen = model.generate(**enc, generation_config=generation)
            new_tokens = gen[:, enc["input_ids"].shape[1]:]
            batch_rows = []
            for j, token_row in zip(batch, new_tokens):
                if j["item_id"] in done:
                    continue
                token_ids = token_row.tolist()
                token_status = generation_token_status(token_ids, max_new_tokens)
                raw = tokenizer.decode(token_ids[:token_status["generated_tokens"]],
                                       skip_special_tokens=True)
                batch_rows.append({
                    "job_id": f"mem::{run_meta['run_key']}::{run_meta['domain']}::{j['item_id']}",
                    "model_key": run_meta["run_key"], "registry_key": run_meta["registry_key"],
                    "domain": run_meta["domain"], "task": "recitation",
                    "training_seed": run_meta.get("training_seed"),
                    "checkpoint": run_meta.get("checkpoint"),
                    "item_id": j["item_id"], "run_identity": run_identity,
                    "protocol_hash": run_meta["protocol_hash"],
                    "raw_text": raw, "reasoning_content": None,
                    "usage": None, **token_status,
                    "adapter": adapter_path, "batch_size": batch_size,
                    "inference_seed": inference_seed, "backend": "local_nf4",
                })
            rows.extend(batch_rows)
            generated += len(batch_rows)
            with out_path.open("a", encoding="utf-8", newline="\n") as handle:
                for row in batch_rows:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            if generated % progress_every < batch_size:
                print(f"  [{run_meta['run_key']}/{run_meta['domain']}] "
                      f"{len(done) + generated}/{len(jobs)}", flush=True)
    finally:
        if own_load:
            model = None
            tokenizer = None
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
    return {"run_key": run_meta["run_key"], "domain": run_meta["domain"],
            "rows": len(rows), "generated_this_invocation": len(missing_jobs),
            "out_path": str(out_path)}


# ---------------------------------------------------------------------------
# text2vec encoder + memory scoring (CPU-capable; lazy torch, optional numpy)


class Text2vecEncoder:
    """Frozen full-text chunked embedding per the plan §8.1 protocol.

    K = 128 - num_special_tokens_to_add(pair=False); one encode of the full
    text without special tokens; non-overlapping K-token chunks keeping the
    whole tail; per-chunk special tokens added; attention-mask mean pooling;
    token-count weighted chunk average; whole-text L2 normalization.
    """

    def __init__(self, encoder_path: str, identity: dict, batch_size: int = 32,
                 num_threads: int | None = None):
        import os
        import torch
        from transformers import AutoModel, AutoTokenizer
        # Many-core boxes thrash tiny per-chunk matmuls with the default thread
        # count (observed: >20 min stuck at 100% on a 128-core CPU; 7.7 s at 4
        # threads).  CPU scoring wants a small pool.
        torch.set_num_threads(num_threads or min(4, os.cpu_count() or 4))
        self.torch = torch
        self.path = encoder_path
        self.identity = identity
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(encoder_path, local_files_only=True)
        self.model.eval()
        self.batch_size = batch_size
        self.specials = self.tokenizer.num_special_tokens_to_add(pair=False)
        self.k = ENCODER_WINDOW_TOKENS - self.specials
        if self.k <= 0:
            raise ValueError("encoder consumes the whole 128-token window with "
                             "specials; protocol requires K >= 1")

    def _pool(self, encoded) -> list[list[float]]:
        with self.torch.inference_mode():
            hidden = self.model(**encoded).last_hidden_state        # B×T×H
            mask = encoded["attention_mask"].unsqueeze(-1).float()  # B×T×1
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1.0)
            return (summed / counts).tolist()

    def embed_long_text(self, text: str) -> tuple[list[float], dict]:
        """Full-text protocol embedding; returns (vector, audit)."""
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not normalized:
            raise ValueError("empty text cannot be embedded (anomaly)")
        token_ids = self.tokenizer.encode(normalized, add_special_tokens=False)
        total = len(token_ids)
        windows = probe.chunk_windows(total, self.k)
        chunk_vecs, chunk_counts = [], []
        for start in range(0, len(windows), self.batch_size):
            batch_windows = windows[start:start + self.batch_size]
            batch_encoded = []
            for lo, hi in batch_windows:
                ids = [self.tokenizer.cls_token_id] + token_ids[lo:hi] \
                    + [self.tokenizer.sep_token_id]
                batch_encoded.append(ids)
            width = max(len(ids) for ids in batch_encoded)
            input_ids, attention = [], []
            for ids in batch_encoded:
                pad = width - len(ids)
                input_ids.append(ids + [self.tokenizer.pad_token_id] * pad)
                attention.append([1] * len(ids) + [0] * pad)
            encoded = {"input_ids": self.torch.tensor(input_ids),
                       "attention_mask": self.torch.tensor(attention)}
            for vector in self._pool(encoded):
                chunk_vecs.append(vector)
            chunk_counts.extend(chunk_token_counts(batch_windows))
        pooled = probe.weighted_mean_pool(chunk_vecs, chunk_counts)
        audit = {"text_sha256": probe.sha256_text(normalized), "total_tokens": total,
                 "chunks": len(windows), "k": self.k,
                 "chunk_token_counts": chunk_counts}
        return probe.l2_normalize(pooled), audit

    def embed_many(self, texts: list[str]) -> tuple[list[list[float]], list[dict]]:
        vectors, audits = [], []
        for text in texts:
            vector, audit = self.embed_long_text(text)
            vectors.append(vector)
            audits.append(audit)
        return vectors, audits


def similarity_matrix(outputs, references, use_numpy: bool = True):
    """Cosine matrix (len(outputs) × len(references)); numpy when available,
    pure-Python otherwise (fixture scale).  Both sides already L2-normalized."""
    if use_numpy:
        try:
            import numpy
        except ImportError:
            use_numpy = False
    if use_numpy:
        y = numpy.array(outputs)
        r = numpy.array(references)
        return y @ r.T
    return [[probe.cosine(y, r) for r in references] for y in outputs]


def score_memory_run(run_key: str, domain: str, articles: list[Article],
                     raw_outputs: dict[str, str], encoder: Text2vecEncoder,
                     threshold: float = probe.MEMORY_THRESHOLD) -> dict:
    """Score one run: encode all references once (caller may pass a cache),
    classify every output Eff./Dup./Err., and return per-row records.

    The comparison pool is the WHOLE domain corpus; the target comparison has
    priority; other-article ties resolve by the first canonical id; empty
    normalized outputs are Err. without any embedding call.
    """
    reference_texts = [a.repaired_content for a in articles]
    ref_vectors = encoder.embed_many(reference_texts)[0]
    target_index = {a.canon_id: i for i, a in enumerate(articles)}
    rows, problems = [], []
    to_embed, to_embed_ids = [], []
    for canon_id in sorted(raw_outputs):
        text = raw_outputs[canon_id]
        if not text.replace("\r\n", "\n").replace("\r", "\n").strip():
            rows.append({"item_id": canon_id, "class": "Err.", "empty_output": True,
                         "s_target": None, "s_other": None, "best_other_id": None})
        else:
            to_embed.append(text)
            to_embed_ids.append(canon_id)
    out_vectors = encoder.embed_many(to_embed)[0] if to_embed else []
    sims = similarity_matrix(out_vectors, ref_vectors)
    if len(sims) and len(sims[0]) != len(articles):
        raise ValueError("similarity matrix width != comparison pool size")
    for row_index, canon_id in enumerate(to_embed_ids):
        target = target_index[canon_id]
        row = list(sims[row_index])
        s_target = float(row[target])
        row[target] = -math.inf                 # exclude the target for s_other
        best_score = max(row)
        best_id = articles[row.index(best_score)].canon_id
        cls = probe.classify_memory(False, s_target, best_score, threshold)
        rows.append({"item_id": canon_id, "class": cls, "empty_output": False,
                     "s_target": s_target, "s_other": best_score,
                     "best_other_id": best_id})
    counts = probe.memory_rate([r["class"] for r in rows])
    return {"run_key": run_key, "domain": domain, "n": counts["n"],
            "eff": counts["eff"], "dup": counts["dup"], "err": counts["err"],
            "memory_rate": counts["memory_rate"], "rows": rows,
            "problems": problems}


# ---------------------------------------------------------------------------
# aggregation (pure; runs on fixtures and on real scores alike)


def build_points(score_index: dict, expected_points: int = 8) -> dict:
    """Assemble the 8-point table from per-run summaries.

    ``score_index``: {"runs": [ {run_key, registry_key, domain, role, seed,
    memory_rate, r_at_5, n_memory, n_test, memory_source_sha256,
    retrieval_source_sha256, protocol_hash, m_seed_sd?...} ]}.  SSL points
    average the three seeds' PAIRED (M, R); Base/API points are single runs;
    x and y always bind to the same runs.
    """
    runs = score_index["runs"]
    points = []
    marks = {"qwen3_8b": "square", "ling_mini_2": "triangle",
             "glm0414_base": "x", "glm0414_ssl": "*"}
    plain_keys = ("qwen3_8b", "ling_mini_2", "glm0414_base")
    for registry_key in plain_keys:
        for domain in probe.DOMAINS:
            matches = [r for r in runs if r["registry_key"] == registry_key
                       and r["domain"] == domain]
            if len(matches) != 1:
                raise probe.CoverageError(
                    f"{registry_key}/{domain}: expected exactly 1 run, got {len(matches)}")
            run = matches[0]
            if run.get("memory_rate") is None or run.get("r_at_5") is None:
                raise probe.CoverageError(
                    f"{registry_key}/{domain}: memory or retrieval score missing — "
                    "points are never built from partial runs")
            points.append(_point_from_run(run, mark=marks.get(registry_key, "*"),
                                          is_ssl=False))
    for domain in probe.DOMAINS:
        seeds = sorted((r for r in runs if r["registry_key"] == "glm0414_ssl"
                        and r["domain"] == domain), key=lambda r: r["seed"])
        if len(seeds) != 3:
            raise probe.CoverageError(
                f"glm0414_ssl/{domain}: expected 3 seed runs, got {len(seeds)}")
        for seed_run in seeds:
            if seed_run.get("memory_rate") is None or seed_run.get("r_at_5") is None:
                raise probe.CoverageError(
                    f"glm0414_ssl/{domain}/seed{seed_run['seed']}: paired score missing")
        hashes = {s.get("protocol_hash") for s in seeds}
        if len(hashes) != 1:
            raise probe.CoverageError(f"glm0414_ssl/{domain}: seeds span protocol "
                                      f"hashes {hashes}; mixing versions is forbidden")
        agg = probe.aggregate_seed_points(
            [(s["memory_rate"], s["r_at_5"]) for s in seeds])
        points.append({
            "point_id": f"glm0414_ssl::{domain}", "model": seeds[0]["model"],
            "provider": seeds[0].get("provider", "local NF4"), "domain": domain,
            "model_run_key": "glm0414_ssl",
            "m_mean": agg["m_mean"], "m_seed_sd": agg["m_seed_sd"],
            "r_mean": agg["r_mean"], "r_seed_sd": agg["r_seed_sd"],
            "m_protocol_hash": seeds[0]["protocol_hash"],
            "r_protocol_hash": seeds[0]["protocol_hash"],
            "n_runs": 3, "n_articles": seeds[0].get("m_n"),
            "n_test": seeds[0].get("r_n"),
            "m_source_sha256": [s["memory_source_sha256"] for s in seeds],
            "r_source_sha256": [s["retrieval_source_sha256"] for s in seeds],
            "protocol_hash": seeds[0]["protocol_hash"],
            "is_ssl_aggregate": True, "mark": marks["glm0414_ssl"],
            "seed_detail": [{"seed": s["seed"], "checkpoint": s.get("checkpoint"),
                             "m": s["memory_rate"], "r": s["r_at_5"]} for s in seeds],
        })
    if len(points) != expected_points:
        raise probe.CoverageError(f"{len(points)} points != expected {expected_points}")
    return {"schema": "figure3-points/1", "points": points}


def _point_from_run(run: dict, mark: str, is_ssl: bool) -> dict:
    return {
        "point_id": f"{run['run_key']}::{run['domain']}", "model": run["model"],
        "provider": run.get("provider", ""), "domain": run["domain"],
        "model_run_key": run["run_key"],
        "m_mean": run["memory_rate"], "m_seed_sd": None,
        "r_mean": run["r_at_5"], "r_seed_sd": None,
        "m_protocol_hash": run.get("protocol_hash"),
        "r_protocol_hash": run.get("protocol_hash"),
        "n_runs": 1, "n_articles": run.get("m_n"), "n_test": run.get("r_n"),
        "m_source_sha256": run["memory_source_sha256"],
        "r_source_sha256": run["retrieval_source_sha256"],
        "protocol_hash": run.get("protocol_hash"),
        "is_ssl_aggregate": False, "mark": mark,
    }
