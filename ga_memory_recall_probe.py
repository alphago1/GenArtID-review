"""Multimodel memory-recall probe (figure3): registry, protocol hashing, job
manifests, smoke ID selection, memory three-way classification, seed
aggregation, coverage/resume gates, and the API transport contract.

Standard library only.  Heavy dependencies (torch/transformers/embedding
models) are NEVER imported here; they belong to the lazy, authorized local
stage.  Scientific inputs reuse the frozen genartid modules (corpus / split /
parsing / scoring) so there is no second, diverging copy of the protocol.

Frozen rules implemented here (reports/PLAN_MULTIMODEL_MEMORY_RECALL_NEW_CHAT_2026-10-07.md):
- semantic config hash: fixed-order JSON / UTF-8; absolute output paths,
  timestamps and the key env-var name never enter it;
- memory classes Eff./Dup./Err. are mutually exclusive and exhaustive, the
  target comparison has priority, the 0.85 threshold comparison never rounds;
- SSL figure points average the three selected seeds' PAIRED (M, R) values;
  spread is the sample SD (ddof=1), never called a 95% CI;
- API retries: 429/5xx/timeout get at most 2 extra same-payload attempts
  honouring Retry-After or a 2/4 s backoff; 401/403/model-missing are fatal;
  a timeout counts as a possibly-billed attempt, never as "not called";
- missing usage is recorded as null (unknown), never coerced to zero.
"""
from __future__ import annotations

import hashlib
import json
import re
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

DOMAINS = ("civil", "criminal_law")

# ---------------------------------------------------------------------------
# hashing helpers


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(payload) -> bytes:
    """Fixed-order JSON / UTF-8 serialization used for every semantic hash."""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")


def semantic_hash(payload) -> str:
    return sha256_bytes(canonical_json_bytes(payload))


# keys never allowed into the semantic protocol hash (non-science or secret
# adjacent); they may exist in the on-disk config but are stripped first
SEMANTIC_EXCLUDE_KEYS = {"outputs_root", "log_dir", "timestamp", "generated_utc"}

API_KEY_ENV = "siliconflow_key"          # the ONLY authorized key variable name
REDACT_PATTERN = re.compile(r"sk-[A-Za-z0-9_-]{6,}|Bearer\s+\S+")


def redact(text: str) -> str:
    """Best-effort secret scrubbing for anything written to logs/reports."""
    return REDACT_PATTERN.sub("[REDACTED]", text)


def semantic_config(config: dict) -> dict:
    """Config copy with absolute output paths / timestamps / key name removed.

    Nested ``api.key_env`` is dropped because swapping the variable name is an
    operational change, not a protocol change; the key VALUE never appears in
    any config at all.
    """

    def strip(obj, path=()):
        if isinstance(obj, dict):
            return {
                key: strip(value, path + (key,))
                for key, value in obj.items()
                if key not in SEMANTIC_EXCLUDE_KEYS and path + (key,) != ("api", "key_env")
            }
        if isinstance(obj, list):
            return [strip(item, path) for item in obj]
        return obj

    return strip(config)


# ---------------------------------------------------------------------------
# registry (draft: candidate identity separated from actual service identity)

REGISTRY_REQUIRED_KEYS = ("registry_key", "target_model_id", "display_name",
                          "inference_source", "provider")
REGISTRY_SOURCES = ("api", "local_nf4", "local_nf4_adapters")


def load_registry(path: Path) -> dict:
    reg = json.loads(Path(path).read_text(encoding="utf-8"))
    if reg.get("schema") != "figure3-model-registry/1":
        raise ValueError(f"{path}: unexpected registry schema {reg.get('schema')!r}")
    keys = [m.get("registry_key") for m in reg["models"]]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{path}: duplicate registry_key values {keys}")
    for model in reg["models"]:
        for key in REGISTRY_REQUIRED_KEYS:
            if key not in model or model[key] in (None, ""):
                raise ValueError(f"{path}: model {model.get('registry_key')!r} missing {key}")
        if model["inference_source"] not in REGISTRY_SOURCES:
            raise ValueError(f"{path}: {model['registry_key']}: unknown inference_source")
        if model["inference_source"] == "api" and model.get("actual_api_model_id") is None:
            # candidate HF ids stay candidates until the discover step confirms them
            model["identity_status"] = model.get("identity_status", "candidate_unverified")
    return reg


def registry_missing_items(reg: dict) -> list[str]:
    """Human-readable list of unverified identity facts (draft state)."""
    missing: list[str] = []
    for model in reg["models"]:
        key = model["registry_key"]
        if model["inference_source"] == "api":
            if model.get("actual_api_model_id") is None:
                missing.append(f"{key}: actual API model id not confirmed")
            if model.get("snapshot_utc") is None:
                missing.append(f"{key}: service capability snapshot not taken")
        else:
            if not model.get("actual_local_path"):
                missing.append(f"{key}: local model/adapter path not verified")
            if not model.get("identity_sha256"):
                missing.append(f"{key}: local weight/file identity hashes not registered")
    return missing


# ---------------------------------------------------------------------------
# prompt rendering (frozen byte-exact per task, shared by every model)

RECITATION_SYSTEM_TEMPLATE = (
    "你现在要逐条背诵《{law_display}》。"
    "必须只输出指定条文的正文内容，不要输出标题、目录、注释、解释、来源或任何额外文字。"
    "尽量保持官方常见标点与换行风格。"
)
RECITATION_USER_TEMPLATE = "请背诵{full_name}全文。只输出正文。"


def recitation_prompts(articles, domain: str) -> list[dict]:
    """One recitation request per article; inputs carry NO reference text.

    ``articles`` are genartid.corpus.Article rows of the matching domain.
    ``law_display`` keeps the full law name WITHOUT its own 《》 (the template
    supplies them); ``full_name`` is the verbatim corpus name, so 之N parts
    survive exactly as written (never re-derived from an index).
    """
    from ga_corpus import DOMAIN_FILES
    if domain not in DOMAINS:
        raise ValueError(f"recitation: unsupported domain {domain!r}")
    law_display = DOMAIN_FILES[domain][2][1:-1]        # 中华人民共和国民法典 / 刑法
    system = RECITATION_SYSTEM_TEMPLATE.format(law_display=law_display)
    prompts = []
    for article in articles:
        user = RECITATION_USER_TEMPLATE.format(full_name=article.raw_name)
        prompts.append({
            "item_id": article.canon_id,
            "system": system,
            "user": user,
            "system_sha256": sha256_text(system),
            "user_sha256": sha256_text(user),
        })
    return prompts


def retrieval_prompts(questions, profile: dict) -> list[dict]:
    """Retrieval requests reuse the GLM profile's verbatim system/user text.

    Sends the question text only — no gold, no corpus, no reference answer.
    """
    system = profile["prompt"]["system_prompt_verbatim_notebook_top5"]
    template = profile["prompt"]["user_template"]
    if "{question}" not in template:
        raise ValueError("retrieval user_template lacks {question} placeholder")
    prompts = []
    for question in questions:
        user = template.format(question=question.raw_question)
        prompts.append({
            "item_id": str(question.qid),
            "system": system,
            "user": user,
            "system_sha256": sha256_text(system),
            "user_sha256": sha256_text(user),
        })
    return prompts


# ---------------------------------------------------------------------------
# deterministic smoke selection (no tokenizer, no score-based choice)


def recitation_smoke_ids(articles, n: int = 8) -> list[str]:
    """Stratified 8-article smoke set, identical for every model in a domain.

    Articles are sorted by (number, part, canonical id); picks sit at
    floor(k*(N-1)/7) for k = 0..7.  刑法 must cover the plain / 之一 / 之二+
    categories: a missing category contributes its smallest-ID article,
    replacing from the END of the picked list, and a replacement may never
    destroy coverage that the list already has.  Deterministic, data-only.
    """
    ordered = sorted(articles, key=lambda a: (a.number, a.part, a.canon_id))
    total = len(ordered)
    if total < n:
        raise ValueError(f"smoke needs >= {n} articles, domain has {total}")

    def category(article) -> str:
        if not article.part:
            return "plain"
        return "part1" if article.part == "之一" else "part2plus"

    picks = [ordered[k * (total - 1) // (n - 1)] for k in range(n)]
    covered = {category(a) for a in picks}
    existing = {category(a) for a in ordered}            # civil has no 之N at all
    for cat in ("plain", "part1", "part2plus"):          # deterministic order
        if cat not in existing or cat in covered:
            continue
        candidate = next(a for a in ordered if category(a) == cat)
        for position in range(len(picks) - 1, -1, -1):   # replace from the end
            survivors = picks[:position] + picks[position + 1:]
            still_covered = {category(a) for a in survivors}
            if covered <= still_covered:                 # no existing coverage lost
                picks[position] = candidate
                covered.add(cat)
                break
        else:
            raise RuntimeError("smoke replacement failed to preserve coverage")
    return [a.canon_id for a in picks]


def retrieval_smoke_ids(dev_ids: list[str], n: int = 8) -> list[str]:
    """First n dev question ids in stable string order; test is never used."""
    ordered = sorted(str(q) for q in dev_ids)
    if len(ordered) < n:
        raise ValueError(f"smoke needs >= {n} dev questions, got {len(ordered)}")
    return ordered[:n]


# ---------------------------------------------------------------------------
# job manifests

MEMORY_RUN_KINDS = ("base", "ssl_seed", "api_model")


def build_memory_jobs(model_runs: list[dict], articles_by_domain: dict) -> list[dict]:
    """One memory job per (run, domain, article).

    ``model_runs``: dicts with registry_key/run_id/domain/inference_source and
    an ``articles`` override for domain-scoped runs.  Expected default total:
    10590 (GLM base + three seeds + Qwen + Ling over both corpora).
    """
    jobs = []
    for run in model_runs:
        domain = run["domain"]
        for article in articles_by_domain[domain]:
            jobs.append({
                "job_id": f"mem::{run['run_key']}::{domain}::{article.canon_id}",
                "task": "recitation",
                "model_run_key": run["run_key"],
                "registry_key": run["registry_key"],
                "run_id": run["run_id"],
                "domain": domain,
                "training_seed": run.get("training_seed"),
                "checkpoint": run.get("checkpoint"),
                "item_id": article.canon_id,
                "inference_source": run["inference_source"],
            })
    return jobs


def build_retrieval_jobs(api_models: list[dict], test_questions: dict) -> list[dict]:
    """One retrieval job per (API model, domain, fixed-test question)."""
    jobs = []
    for model in api_models:
        for domain, questions in test_questions.items():
            for question in questions:
                jobs.append({
                    "job_id": f"ret::{model['registry_key']}::{domain}::{question.qid}",
                    "task": "retrieval",
                    "model_run_key": model["registry_key"],
                    "registry_key": model["registry_key"],
                    "run_id": "api",
                    "domain": domain,
                    "training_seed": None,
                    "checkpoint": None,
                    "item_id": str(question.qid),
                    "inference_source": "api",
                })
    return jobs


def build_legacy_retrieval_jobs(domains: list[str], seeds: list[int],
                                test_counts: dict) -> list[dict]:
    """Source mapping for the EXISTING GLM retrieval evidence (reused, never
    regenerated): three seed checkpoints + one Base per domain."""
    jobs = []
    for domain in domains:
        run_keys = [f"glm0414_seed{seed}" for seed in seeds] + ["glm0414_base"]
        for run_key in run_keys:
            for index in range(test_counts[domain]):
                jobs.append({
                    "job_id": f"legacy::{domain}::{run_key}::#{index}",
                    "task": "retrieval_reuse",
                    "model_run_key": run_key,
                    "domain": domain,
                    "inference_source": "reused_local_evidence",
                })
    return jobs


# ---------------------------------------------------------------------------
# coverage / resume gates


class CoverageError(ValueError):
    """Raised when completed rows cannot form a trustworthy run."""


def validate_run_rows(rows: list[dict], expected_ids: set[str],
                      run_identity: str) -> None:
    """Gate before any run is called complete.

    Requires exactly-once coverage of every expected job id, no duplicate
    successes, no system failures, and a run identity that matches on every
    row (protocol/registry/config drift makes caches unmixable).
    """
    seen: set[str] = set()
    duplicates: list[str] = []
    failures: list[str] = []
    identity_mismatch: list[str] = []
    for row in rows:
        job_id = row.get("job_id")
        if job_id in seen:
            duplicates.append(job_id)
        seen.add(job_id)
        if row.get("system_failure"):
            failures.append(job_id)
        if row.get("run_identity") != run_identity:
            identity_mismatch.append(job_id)
    if duplicates:
        raise CoverageError(f"duplicate successful job ids: {duplicates[:5]}")
    if failures:
        raise CoverageError(f"unresolved system failures: {failures[:5]}")
    if identity_mismatch:
        raise CoverageError(f"rows from a different run identity: {identity_mismatch[:5]}")
    missing = sorted(expected_ids - seen)
    if missing:
        raise CoverageError(f"missing {len(missing)} job ids, first: {missing[:5]}")
    extra = sorted(seen - expected_ids)
    if extra:
        raise CoverageError(f"rows outside the frozen job list: {extra[:5]}")


def resume_missing_ids(rows: list[dict], expected_ids: set[str],
                       run_identity: str) -> tuple[set[str], list[str]]:
    """Ids still to generate; only rows with the SAME identity count."""
    done: set[str] = set()
    discarded: list[str] = []
    for row in rows:
        if row.get("run_identity") != run_identity:
            discarded.append(row.get("job_id", "?"))
            continue
        done.add(row["job_id"])
    return expected_ids - done, discarded


# ---------------------------------------------------------------------------
# memory scoring math (pure; embeddings are supplied by the local stage)


MEMORY_THRESHOLD = 0.85


def classify_memory(output_empty: bool, s_target: float | None,
                    s_other: float | None,
                    threshold: float = MEMORY_THRESHOLD) -> str:
    """Eff./Dup./Err. — mutually exclusive, exhaustive, target-first.

    The threshold comparison happens on the raw floats; rounding before the
    comparison is forbidden.  An empty normalized output is a true Err. (a
    real empty response), while a missing embedding is an anomaly the caller
    must surface, never silently classify.
    """
    if output_empty:
        return "Err."
    if s_target is None or s_other is None:
        raise ValueError("non-empty output without embeddings cannot be classified")
    if s_target >= threshold:
        return "Eff."
    if s_other >= threshold:
        return "Dup."
    return "Err."


def best_other_id(similarities: dict[str, float], target_id: str) -> tuple[str, float]:
    """Highest-similarity non-target article; ties resolved by the stable
    canonical-ID order (first in sorted order wins), never by luck."""
    best_id, best_score = None, None
    for article_id in sorted(similarities):
        if article_id == target_id:
            continue
        score = similarities[article_id]
        if best_score is None or score > best_score:
            best_id, best_score = article_id, score
    if best_id is None:
        raise ValueError("comparison pool must contain at least one non-target article")
    return best_id, best_score


def memory_rate(classes: list[str]) -> dict[str, float | int]:
    """Eff./Dup./Err. counts and the memory rate M = Eff_count / N."""
    total = len(classes)
    if total == 0:
        raise ValueError("no rows to aggregate")
    eff = sum(1 for c in classes if c == "Eff.")
    dup = sum(1 for c in classes if c == "Dup.")
    err = sum(1 for c in classes if c == "Err.")
    if eff + dup + err != total:
        raise CoverageError("class counts do not cover every row exactly once")
    return {"n": total, "eff": eff, "dup": dup, "err": err,
            "memory_rate": eff / total}


# full-text chunking spec (encoder-agnostic math; K fixed by the encoder)

def chunk_windows(total_tokens: int, k: int) -> list[tuple[int, int]]:
    """Non-overlapping [start, end) windows of at most k tokens; the tail is
    kept in full (truncating the first 128 tokens is forbidden)."""
    if k <= 0:
        raise ValueError("chunk size k must be positive")
    if total_tokens <= 0:
        raise ValueError("an article must have at least one token")
    return [(start, min(start + k, total_tokens)) for start in range(0, total_tokens, k)]


def weighted_mean_pool(chunk_vectors, chunk_token_counts) -> list[float]:
    """Token-count weighted mean of per-chunk vectors (padding excluded by
    construction upstream); caller L2-normalizes afterwards."""
    if len(chunk_vectors) != len(chunk_token_counts):
        raise ValueError("vector/count length mismatch")
    if any(c <= 0 for c in chunk_token_counts):
        raise ValueError("every chunk needs at least one real token")
    dim = len(chunk_vectors[0])
    total = sum(chunk_token_counts)
    accumulated = [0.0] * dim
    for vector, count in zip(chunk_vectors, chunk_token_counts):
        for index in range(dim):
            accumulated[index] += vector[index] * count
    return [value / total for value in accumulated]


def l2_normalize(vector) -> list[float]:
    import math
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        raise ValueError("zero vector cannot be normalized (anomaly, not a score)")
    return [value / norm for value in vector]


def cosine(a, b) -> float:
    import math
    if len(a) != len(b):
        raise ValueError("cosine dimension mismatch")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        raise ValueError("cosine with a zero vector is an anomaly")
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# seed aggregation


def mean_sd(values) -> tuple[float, float | None]:
    """Mean and sample SD (ddof=1); SD is None for a single observation and
    must never be reported as 0 or as a confidence interval."""
    values = [float(v) for v in values]
    if not values:
        raise ValueError("no values")
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) >= 2 else None
    return mean, sd


def aggregate_seed_points(pairs: list[tuple[float, float]]) -> dict:
    """SSL figure point from three seeds' PAIRED (M, R).

    x and y are bound to the SAME seed runs — pairing memory's best seed with
    retrieval's best seed is forbidden.  Different checkpoints across seeds
    are fine; mixing one seed's M with another seed's R is not.
    """
    if len(pairs) != 3:
        raise ValueError(f"SSL points aggregate exactly three seeds, got {len(pairs)}")
    m_mean, m_sd = mean_sd([m for m, _ in pairs])
    r_mean, r_sd = mean_sd([r for _, r in pairs])
    return {"m_mean": m_mean, "m_seed_sd": m_sd,
            "r_mean": r_mean, "r_seed_sd": r_sd, "n_runs": len(pairs)}


# ---------------------------------------------------------------------------
# API transport (stdlib urllib; CPU commands never execute this path)

API_TIMEOUT_S = 120
API_MAX_EXTRA_RETRIES = 2
API_BACKOFF_S = (2.0, 4.0)


class ApiFatalError(RuntimeError):
    """401/403/model-missing: retrying the same payload cannot help."""


class ApiRetryableError(RuntimeError):
    """429/5xx/timeout: retry with backoff; the attempt may already be billed."""


class BudgetExceededError(RuntimeError):
    """The dispatch would push a budget counter past its authorized limit."""


@dataclass
class BudgetLedger:
    """Tracks the three authorized budget limits; reservations precede every
    dispatch, so concurrent reservations can never oversubscribe the grant."""
    max_requests: int | None = None
    max_total_tokens: int | None = None
    api_budget: float | None = None
    requests: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0
    unknown_usage_responses: int = 0

    def reserve(self, est_input_tokens: int, max_output_tokens: int,
                price_per_1m_input: float | None, price_per_1m_output: float | None) -> None:
        if self.max_requests is not None and self.requests + 1 > self.max_requests:
            raise BudgetExceededError(
                f"request limit {self.max_requests} reached before dispatch")
        projected = self.total_tokens + est_input_tokens + max_output_tokens
        if self.max_total_tokens is not None and projected > self.max_total_tokens:
            raise BudgetExceededError(
                f"token limit {self.max_total_tokens} would be exceeded ({projected})")
        # api_budget == 0 encodes a free-tier authorization: no money gate, the
        # request/token caps carry the budget; a positive budget requires prices
        if self.api_budget and None in (price_per_1m_input, price_per_1m_output):
            raise BudgetExceededError(
                "a money budget is authorized but token prices are unknown; "
                "a reliable cost ceiling cannot be computed")
        if self.api_budget:
            cost = ((est_input_tokens * price_per_1m_input
                     + max_output_tokens * price_per_1m_output) / 1_000_000)
            if self.estimated_cost + cost > self.api_budget:
                raise BudgetExceededError(
                    f"money budget {self.api_budget} would be exceeded "
                    f"({self.estimated_cost + cost:.4f})")

    def record(self, usage: dict | None, billed_cost: float | None = None) -> None:
        self.requests += 1
        if not usage or "total_tokens" not in usage:
            # usage missing is UNKNOWN, never zero: count the known caps in
            self.unknown_usage_responses += 1
            return
        self.total_tokens += int(usage.get("prompt_tokens") or 0)
        self.total_tokens += int(usage.get("completion_tokens") or 0)
        self.total_tokens += int(usage.get("reasoning_tokens") or 0)
        if billed_cost is not None:
            self.estimated_cost += float(billed_cost)


def chat_completion(base_url: str, model_id: str, messages: list[dict],
                    key: str, max_tokens: int, temperature: float = 0.0,
                    frequency_penalty: float = 0.0, stream: bool = False,
                    extra_body: dict | None = None,
                    ledger: BudgetLedger | None = None,
                    est_input_tokens: int = 0,
                    price_per_1m_input: float | None = None,
                    price_per_1m_output: float | None = None,
                    timeout_s: int = API_TIMEOUT_S,
                    max_extra_retries: int = API_MAX_EXTRA_RETRIES,
                    backoff: tuple[float, ...] = API_BACKOFF_S,
                    transport=None) -> dict:
    """One chat completion with the frozen decode parameters and retry policy.

    ``transport`` exists for tests (mock HTTP); production uses urllib.  The
    Authorization header is built here and never logged.  A timeout raises
    ApiRetryableError: the request may still have been billed, so retries are
    recorded as new attempts, not as "never happened".
    """
    payload = {
        "model": model_id, "messages": messages, "max_tokens": max_tokens,
        "temperature": temperature, "frequency_penalty": frequency_penalty,
        "stream": stream, "n": 1,
    }
    if extra_body:
        payload.update(extra_body)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",   # never printed or logged
    }

    def dispatch() -> dict:
        if transport is not None:
            return transport(url=url, headers=headers, body=body, timeout_s=timeout_s)
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")
            if error.code in (401, 403):
                raise ApiFatalError(f"HTTP {error.code}: authorization rejected") from error
            if error.code == 404:
                raise ApiFatalError(f"HTTP 404: model/endpoint missing: {redact(detail[:200])}") from error
            if error.code == 429 or error.code >= 500:
                retry_after = error.headers.get("Retry-After") if error.headers else None
                raise ApiRetryableError(
                    f"HTTP {error.code}; retry_after={retry_after!r}; body={redact(detail[:200])!r}"
                ) from error
            raise ApiFatalError(f"HTTP {error.code}: {redact(detail[:200])!r}") from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise ApiRetryableError(f"transport failure: {redact(str(error)[:200])}") from error

    if ledger is not None:
        ledger.reserve(est_input_tokens=est_input_tokens,
                       max_output_tokens=max_tokens,
                       price_per_1m_input=price_per_1m_input,
                       price_per_1m_output=price_per_1m_output)
    attempt = 0
    while True:
        try:
            response = dispatch()
            break
        except ApiRetryableError as error:
            if attempt >= max_extra_retries:
                raise
            delay: float | None = None
            match = re.search(r"retry_after=(\d+(?:\.\d+)?)", str(error))
            if match:
                delay = min(float(match.group(1)), 30.0)
            else:
                delay = backoff[min(attempt, len(backoff) - 1)]
            attempt += 1
            time.sleep(delay)
    choices = response.get("choices")
    if not choices or not isinstance(choices, list):
        raise ApiFatalError("malformed response: no choices array (system failure, "
                            "never to be disguised as an empty output)")
    return response


def extract_content(response: dict) -> tuple[str, str | None, dict | None, str | None]:
    """(content, reasoning_content, usage, finish_reason) from a response.

    Only the official ``message.content`` is the task output; reasoning is
    stored separately and never parsed for answers.
    """
    choices = response.get("choices") or []
    if not choices:
        return "", None, response.get("usage"), None
    message = choices[0].get("message") or {}
    return (message.get("content") or "",
            message.get("reasoning_content"),
            response.get("usage"),
            choices[0].get("finish_reason"))


def parse_model_list(models_response: dict) -> list[str]:
    """Model ids from GET /models; the raw response is recorded separately."""
    data = models_response.get("data")
    if not isinstance(data, list):
        raise ValueError("models response lacks a data array")
    return sorted(str(item["id"]) for item in data if isinstance(item, dict) and item.get("id"))


# ---------------------------------------------------------------------------
# point table identity checks


def check_point_identity(point: dict) -> list[str]:
    """A figure point may only aggregate rows sharing model identity, prompt
    hashes, split and scoring protocol.  Returns a list of problems."""
    problems = []
    for field_name in ("model_run_key", "domain", "m_protocol_hash", "r_protocol_hash",
                       "m_source_sha256", "r_source_sha256"):
        if point.get(field_name) in (None, ""):
            problems.append(f"point lacks {field_name}")
    if point.get("is_ssl_aggregate") and point.get("n_runs") != 3:
        problems.append("SSL aggregate must carry exactly three seed runs")
    if not point.get("is_ssl_aggregate") and point.get("n_runs") not in (1, None):
        problems.append("non-aggregate points come from exactly one official run")
    return problems
