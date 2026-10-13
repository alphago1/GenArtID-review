"""Base/FT prediction generation entry. dev partition runs freely; the test
partition is HARD-GATED behind a frozen selection artifact.

From the workspace root:

    # CPU dry-run: render prompts, verify seeds/coverage; NO predictions written.
    python scripts/g_generate_predictions.py --domain cpl --partition dev \
        --run-id base --decode greedy --limit 8 --dry-run

    # GPU: Base greedy predictions over dev (or FT with --adapter).
    python scripts/g_generate_predictions.py --domain cpl --partition dev \
        --run-id base --decode greedy --model /path/to/base \
        --tokenizer /path/to/tok --out predictions/cpl/base

    # test partition: additionally requires --selection pointing at a frozen
    # selection artifact whose fingerprint covers BOTH config files and whose
    # split hash matches the recomputed split content; refuses otherwise.

Output (real run): predictions.json = {run_id, global_step|null, partition,
per_question: {qid: {pred, gold, raw, seed, parse_status}}}, ready for
scripts/h_dev_select.py (dev) or scripts/run_test_eval.py (test).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ga_parsing import parse_fixed  # noqa: E402
from ga_pipeline import (generation_eos, load_profile_and_protocol,  # noqa: E402
                               render_eval_prompt, stable_query_seed)
from ga_select import TestGateError, load_frozen_selection, protocol_fingerprint  # noqa: E402
from ga_split import split_manifest_hash  # noqa: E402

DATA = {"civil": "eval_civil_code_questions.json", "cpl": "eval_cpl_questions.json",
         "criminal_law": "eval_criminal_law_questions.json"}
PROFILE_DEFAULT = ROOT / "base_ft_eval_profile.json"
PROTOCOL_DEFAULT = ROOT / "research_protocol.json"




def _dir_identity(path: str | None) -> dict | None:
    """Fingerprint a model/adapter/tokenizer directory by its SMALL text
    files (config/tokenizer/template json + jinja).  Weight tensors are never
    hashed (cost); record sizes of large files instead."""
    if not path:
        return None
    root = Path(path)
    if not root.is_dir():
        return {"error": f"not a directory: {path}"}
    out = {}
    for f in sorted(root.iterdir()):
        if not f.is_file():
            continue
        if f.suffix in (".json", ".jinja") and f.stat().st_size < 2_000_000:
            out[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
        else:
            out[f.name] = f"{f.stat().st_size}B"
    return out

def load_split(domain: str, split_path: str | None = None) -> dict:
    path = Path(split_path) if split_path else (
        ROOT / "artifacts" / "split" / f"{domain}_split.json")
    split = json.loads(path.read_text(encoding="utf-8"))
    if split.get("domain") != domain:
        raise SystemExit(f"{path.name}: split domain {split.get('domain')!r} != {domain!r}")
    stored = split.get("split_manifest_hash", "")
    if stored and split_manifest_hash(split) != stored:
        raise SystemExit(f"{path.name}: split content tampered (recomputed hash "
                         "differs from stored field); regenerate the split first")
    return split


def gate_test_partition(args, split: dict, profile: dict, protocol: dict) -> dict:
    """test requires a frozen selection bound to the CURRENT configs+split."""
    if not args.selection:
        raise SystemExit("test partition requires --selection (frozen selection "
                         "artifact); S-stage decisions must stay dev-only")
    selection = load_frozen_selection(args.selection)
    if selection["protocol_fingerprint"] != protocol_fingerprint(protocol, profile):
        raise SystemExit("selection protocol fingerprint mismatch: configs changed "
                         "after the selection was frozen; refusing")
    if selection["split_manifest_hash"] != split["split_manifest_hash"]:
        raise SystemExit("selection was made under a different split; refusing")
    if args.run_id not in (selection["base_run_id"], selection["selected_run_id"]):
        raise SystemExit(f"run-id {args.run_id!r} is neither the matched base "
                         f"{selection['base_run_id']!r} nor the selected run "
                         f"{selection['selected_run_id']!r}; test evaluates only those two")
    return selection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=["civil", "cpl", "criminal_law"], required=True)
    parser.add_argument("--partition", choices=["dev", "test"], required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--global-step", type=int, default=None,
                        help="checkpoint step; REQUIRED for FT dev candidates "
                             "(the selector refuses unknown steps)")
    parser.add_argument("--decode", choices=["greedy", "sample_explicit"], default="greedy")
    parser.add_argument("--seed", type=int, default=20260910,
                        help="global seed; sampling uses per-question stable seeds")
    parser.add_argument("--adapter", help="FT adapter directory; omit for Base")
    parser.add_argument("--selection", help="frozen selection artifact (test only)")
    parser.add_argument("--model", help="local base model directory (real runs)")
    parser.add_argument("--tokenizer", help="local tokenizer directory")
    parser.add_argument("--profile", default=str(PROFILE_DEFAULT),
                        help="Base/FT evaluation profile (explicit for the 0414 flow)")
    parser.add_argument("--protocol", default=str(PROTOCOL_DEFAULT))
    parser.add_argument("--split", default=None,
                        help="split manifest (default artifacts/split/<domain>_split.json)")
    parser.add_argument("--limit", type=int, help="first N questions of the partition (smoke)")
    parser.add_argument("--out", help="output run directory (required for real runs)")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile, protocol = load_profile_and_protocol(args.profile, args.protocol)

    if args.decode == "sample_explicit" and args.seed not in \
            protocol["decode_variants"]["sample_explicit"]["seeds"]:
        raise SystemExit(f"seed {args.seed} not in pre-registered sampling seeds "
                         f"{protocol['decode_variants']['sample_explicit']['seeds']}")
    if args.adapter and args.partition == "dev" and args.global_step is None:
        raise SystemExit("FT dev candidates require --global-step (selector refuses "
                         "candidates with unknown step)")
    if args.adapter and args.model is None and not args.dry_run:
        raise SystemExit("real FT run requires --model (base weights for the adapter)")

    split = load_split(args.domain, args.split)
    if args.partition == "test":
        gate_test_partition(args, split, profile, protocol)
    qids = split[f"{args.partition}_ids"]
    if args.limit:
        qids = qids[:args.limit]

    questions = {str(q["id"]): q for q in json.loads(
        (ROOT / "inputs" / "data" / DATA[args.domain]).read_text(encoding="utf-8"))}
    missing = [q for q in qids if q not in questions]
    if missing:
        raise SystemExit(f"split references unknown question ids: {missing[:5]}")

    from ga_identifiers import canon_fixed
    split_qs = {q["qid"]: q for q in split["questions"]}
    rows = [{"qid": q, "question": questions[q]["question"],
             "gold": [canon_fixed(a) for a in questions[q]["articles"] if canon_fixed(a)],
             "question_sha256": split_qs[q]["question_sha256"],
             "seed": stable_query_seed(args.seed, q)}
            for q in qids]

    if args.dry_run:
        print(f"[dry-run] domain={args.domain} partition={args.partition} "
              f"run_id={args.run_id} n={len(rows)} decode={args.decode}")
        recomputed = stable_query_seed(args.seed, rows[0]["qid"])
        print(f"[dry-run] per-question seed stable: {rows[0]['seed']} == {recomputed} "
              f"-> {rows[0]['seed'] == recomputed}")
        print("[dry-run] prompt rendering requires --tokenizer; no predictions written")
        return

    if not (args.model or (args.adapter and args.model)):
        raise SystemExit("real run requires --model")
    if not args.out:
        raise SystemExit("real run requires --out (new output run directory)")
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable (cardless mode?); generation refused — "
                         "use --dry-run for CPU validation")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("bf16 unsupported; refusing (frozen precision rule)")
    run_generation(args, profile, protocol, split, rows)


def run_generation(args, profile: dict, protocol: dict, split: dict,
                   rows: list[dict]) -> None:
    import torch
    from peft import PeftModel
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig, GenerationConfig, set_seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, device_map={"": 0},
        torch_dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval()

    decode = profile["decode"][args.decode]
    eos = generation_eos(profile, tokenizer)
    pad_id = profile.get("tokens", {}).get("pad_token_id")
    if not isinstance(pad_id, int):
        pad_id = tokenizer.pad_token_id
    generation = GenerationConfig(
        do_sample=decode["do_sample"], num_beams=decode["num_beams"],
        temperature=decode["temperature"], top_p=decode["top_p"], top_k=decode["top_k"],
        repetition_penalty=decode["repetition_penalty"],
        max_new_tokens=decode["max_new_tokens"],
        eos_token_id=eos, pad_token_id=pad_id,
        bos_token_id=tokenizer.bos_token_id, use_cache=True)

    out_dir = Path(args.out)
    if out_dir.exists():
        raise SystemExit(f"output directory exists: {out_dir}; use a new run name")

    per_question = {}
    for row in rows:
        prompt = render_eval_prompt(tokenizer, profile, row["question"])
        encoded = tokenizer(prompt, return_tensors="pt").to("cuda:0")
        set_seed(row["seed"])
        with torch.inference_mode():
            generated = model.generate(**encoded, generation_config=generation)
        tokens = generated[0, encoded["input_ids"].shape[1]:].tolist()
        raw = tokenizer.decode(tokens, skip_special_tokens=True).strip()
        parsed = parse_fixed(raw)
        per_question[row["qid"]] = {
            "pred": parsed.predictions, "gold": row["gold"], "raw": raw,
            "seed": row["seed"], "parse_status": parsed.status,
            "question_sha256": row["question_sha256"],
        }
        print(f"[{len(per_question)}/{len(rows)}] qid={row['qid']} "
              f"status={parsed.status}", flush=True)

    decode_params = {k: v for k, v in profile["decode"][args.decode].items()
                     if k != "seeds"}
    payload = {
        "run_id": args.run_id,
        "global_step": args.global_step,
        "domain": args.domain, "partition": args.partition,
        "decode": args.decode,
        "decode_params": decode_params,          # G8: the ACTUAL decode used
        "seed": args.seed,
        "adapter": args.adapter,
        "split_manifest_hash": split["split_manifest_hash"],
        "split_file": str(args.split) if args.split else "default",
        "generation_eos_token_id": eos if not isinstance(eos, list) else list(eos),
        "profile_sha256": hashlib.sha256(json.dumps(
            profile, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest(),
        "protocol_sha256": hashlib.sha256(json.dumps(
            protocol, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest(),
        "identity": {                            # G8: model/adapter/tokenizer fingerprints
            "model": _dir_identity(args.model),
            "adapter": _dir_identity(args.adapter),
            "tokenizer": _dir_identity(args.tokenizer or args.model),
            "torch": __import__("torch").__version__,
            "transformers": __import__("transformers").__version__,
        },
        "per_question": per_question,
    }
    # B25: create the directory only after generation succeeded, and write
    # atomically — a crash mid-run must not leave an empty/partial run dir
    # that the exists-guard above then refuses to retry into
    out_dir.mkdir(parents=True)
    tmp_path = out_dir / "predictions.json.tmp"
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8", newline="\n")
    import os
    os.replace(tmp_path, out_dir / "predictions.json")
    print(f"predictions -> {out_dir / 'predictions.json'}")


if __name__ == "__main__":
    main()
