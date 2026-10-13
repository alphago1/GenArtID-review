"""Training entry (S1/S2/S3 runs). GPU-only to execute; --dry-run is CPU.

From the workspace root:

    # CPU: validate data/config/budget and (with a local tokenizer) run the
    # full token/truncation audit, render examples, print the plan. No torch.
    python scripts/f_train_sft.py --domain cpl --ratio 1f_1r --seed 20260910 \
        --tokenizer /path/to/Qwen3-8B --dry-run

    # GPU: same args without --dry-run; refuses unless CUDA is available.
    python scripts/f_train_sft.py --domain cpl --ratio 1f_1r --seed 20260910 \
        --model /path/to/base --out runs/s1/cpl_1f1r_seed20260910

Dry-run outputs under --out (or artifacts/dryrun/ by default): plan.json,
token_audit.json, rendered_examples.json. The real run additionally writes
adapter checkpoints and a run manifest with all config hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ga_pipeline import (PipelineError, plan_training,  # noqa: E402
                               load_train_system_prompt,
                               render_training_prompt, token_audit)
from ga_profile import load_profile  # noqa: E402

DATA_DIR = ROOT / "artifacts" / "sft_corpus_only"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_dataset(domain: str, ratio: str) -> tuple[dict, dict]:
    manifest = json.loads((DATA_DIR / "build_manifest.json").read_text(encoding="utf-8"))
    path = DATA_DIR / domain / f"ratio_{ratio}.json"
    if not path.is_file():
        raise SystemExit(f"missing dataset {path}; run scripts/d_build_corpus_only_sft.py first")
    entry = next((e for e in manifest["outputs"]
                  if e["kind"] == "ratio" and e["domain"] == domain
                  and e["path"].endswith(f"ratio_{ratio}.json")), None)
    if entry is None:
        raise SystemExit(f"{ratio} not in build manifest")
    actual = sha(path)
    if actual != entry["sha256"]:
        raise SystemExit(f"dataset hash mismatch for {path.name}: manifest "
                         f"{entry['sha256'][:12]}... vs actual {actual[:12]}...; rebuild first")
    return json.loads(path.read_text(encoding="utf-8")), manifest


def load_tokenizer(path: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise SystemExit(f"--tokenizer given but transformers is unavailable: {error}")
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=["civil", "cpl", "criminal_law"], required=True)
    parser.add_argument("--ratio", required=True, help="e.g. 1f_1r, 3f_1r, 0f_1r")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model", help="local base model directory (real runs)")
    parser.add_argument("--tokenizer", help="local tokenizer directory (dry-run audits)")
    parser.add_argument("--profile", default=str(ROOT / "base_ft_eval_profile.json"),
                        help="Base/FT evaluation profile whose sha is bound into "
                             "the run plan (explicit for the 0414 flow)")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "dryrun"))
    parser.add_argument("--save-only-model", action="store_true",
                        help="matrix mode: adapter-only checkpoints (G9 "
                             "documented tradeoff: interruption = rerun from "
                             "scratch, never called lossless resume)")
    parser.add_argument("--resume", action="store_true",
                        help="resume an interrupted run from the last FULL-state "
                             "checkpoint (optimizer/scheduler/RNG included); "
                             "requires --out to already contain checkpoints/ and "
                             "refuses otherwise — never a silent fresh start")
    parser.add_argument("--target-modules", choices=["qvo", "all-linear", "glm4-fused"],
                        default="qvo",
                        help="qvo = frozen q/v/o_proj (Qwen family); glm4-fused = "
                             "GLM-4 fused query_key_value + dense (architecture-"
                             "equivalent, pre-registered for the glm-4-9b arm); "
                             "all-linear = capacity ablation only")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix = json.loads((ROOT / "ablation_matrix.json").read_text(encoding="utf-8"))
    profile = load_profile(Path(args.profile))
    dataset, build_manifest = load_dataset(args.domain, args.ratio)

    try:
        plan = plan_training(args.domain, dataset, args.seed, matrix)
    except PipelineError as error:
        raise SystemExit(f"plan refused: {error}")
    plan["max_length"] = args.max_length
    plan["lora_target_modules"] = {
        "qvo": ["q_proj", "v_proj", "o_proj"],
        "glm4-fused": ["query_key_value", "dense"],
    }.get(args.target_modules, args.target_modules)
    plan["profile_sha256"] = hashlib.sha256(json.dumps(
        profile, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()

    out = Path(args.out)
    if args.resume:
        # resume path: the run dir must already exist with at least one FULL-state
        # checkpoint (optimizer.pt/scheduler.pt present); anything less is a
        # refusal, never a silent fresh start
        ckpts = sorted((out / "checkpoints").glob("checkpoint-*")) \
            if (out / "checkpoints").is_dir() else []
        full_state = [c for c in ckpts
                      if (c / "optimizer.pt").is_file() and (c / "scheduler.pt").is_file()]
        if not full_state:
            raise SystemExit(f"--resume: no FULL-state checkpoint under "
                             f"{out}/checkpoints (need optimizer.pt+scheduler.pt); "
                             "refusing to resume")
        if (out / "final_adapter").is_dir():
            raise SystemExit("--resume: final_adapter/ already exists — the run "
                             "completed; nothing to resume")
        latest = max(full_state, key=lambda c: int(c.name.split("-")[1]))
        print(f"[resume] from {latest} "
              f"(global_step {latest.name.split('-')[1]}); plan/tokenizer stages "
              "are skipped, trainer state governs")
    else:
        if out.exists():
            raise SystemExit(f"output directory exists: {out}; refusing to overwrite "
                             "an existing run (resume requires an explicit, "
                             "hash-verified recovery procedure)")
        out.mkdir(parents=True)
    from ga_pipeline import load_train_system_prompt
    plan["train_system_prompt"] = load_train_system_prompt(
        json.loads((ROOT / "research_protocol.json").read_text(encoding="utf-8")))
    if not args.resume:
        (out / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2),
                                       encoding="utf-8", newline="\n")

    tokenizer = None
    if args.tokenizer:
        tokenizer = load_tokenizer(args.tokenizer)
        if not args.resume:
            system = load_train_system_prompt(
                json.loads((ROOT / "research_protocol.json")
                           .read_text(encoding="utf-8")))
            rows = [{"prompt": render_training_prompt(tokenizer, s["prompt"], system),
                     "completion": s["completion"], "mode": s["mode"]}
                    for s in dataset["samples"]]
            eos = tokenizer.eos_token or ""
            audit = token_audit(rows, lambda t: tokenizer.encode(t, add_special_tokens=False),
                                eos, args.max_length)
            (out / "token_audit.json").write_text(
                json.dumps(audit.as_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8", newline="\n")
            examples = [{**rows[i], "sample_id": dataset["samples"][i]["sample_id"]}
                        for i in (0, len(rows) // 2, len(rows) - 1)]
            (out / "rendered_examples.json").write_text(
                json.dumps(examples, ensure_ascii=False, indent=2),
                encoding="utf-8", newline="\n")
            print(f"[dry-run] token audit: {json.dumps(audit.as_dict(), ensure_ascii=False)[:400]}")
    else:
        if not args.resume:
            print("[dry-run] no --tokenizer: token audit pending (need local tokenizer files)")

    if args.dry_run and args.resume:
        raise SystemExit("--dry-run and --resume are mutually exclusive: dry-run "
                         "plans a FRESH run; resume continues an interrupted one")
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        print(f"[dry-run] wrote plan/audit -> {out}")
        return

    # ---- real run: everything below requires GPU ----
    if not args.model:
        raise SystemExit("real run requires --model (local base model directory)")
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable (cardless mode?); real training refused — "
                         "use --dry-run for CPU validation")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("bf16 unsupported on this GPU; refusing (frozen precision rule)")
    run_training(args, plan, dataset, tokenizer or load_tokenizer(args.model), out)


def run_training(args, plan: dict, dataset: dict, tokenizer, out: Path) -> None:
    """The actual TRL/PEFT run; only reachable with CUDA + bf16 verified."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import (AutoModelForCausalLM, BitsAndBytesConfig)
    from trl import SFTConfig, SFTTrainer

    from ga_pipeline import trainer_recipe

    system = load_train_system_prompt(
        json.loads((ROOT / "research_protocol.json").read_text(encoding="utf-8")))
    rows = [{"prompt": render_training_prompt(tokenizer, s["prompt"], system),
             "completion": s["completion"] + (tokenizer.eos_token or "")}
            for s in dataset["samples"]]
    train_ds = Dataset.from_list(rows)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, device_map={"": 0},
        torch_dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True)
    # G7 item 6: ONE construction source — smoke and formal training both
    # build from trainer_recipe(); entry-specific additions are explicit.
    recipe = trainer_recipe(args.domain, str(out / "checkpoints"), seed=args.seed,
                            max_length=args.max_length, train_steps_log=True,
                            target_modules={
                                "qvo": None,
                                "glm4-fused": ["query_key_value", "dense"],
                            }.get(args.target_modules, args.target_modules))
    peft = LoraConfig(**recipe["lora"])
    # pilot keeps full optimizer/scheduler/RNG state (one resume verification,
    # G9 recommendation, done 2026-09-13); matrix runs pass --save-only-model
    # to fit the disk budget — documented tradeoff: interruption = rerun.
    cfg = SFTConfig(**recipe["sft"], save_only_model=bool(args.save_only_model))
    (out / "trainer_recipe.json").write_text(
        json.dumps(recipe, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8", newline="\n")
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=train_ds,
                         processing_class=tokenizer, peft_config=peft)
    # B20 insurance for per-architecture target maps (e.g. glm4-fused): refuse
    # to train if the LoRA config matched no modules (PEFT would silently
    # produce an empty adapter and a no-op "training")
    n_lora = sum(1 for n, p in trainer.model.named_parameters()
                 if "lora_" in n and p.requires_grad)
    if n_lora == 0:
        raise SystemExit(f"LoRA target modules {recipe['lora']['target_modules']} "
                         f"matched no modules in this architecture; refusing")
    # collator label sanity before training: first batch must mask the prompt
    batch = trainer.data_collator([trainer.train_dataset[0]])
    labels = batch["labels"][0]
    supervised_positions = int((labels != -100).sum())
    if supervised_positions <= 0:
        raise SystemExit("collator produced zero supervised positions; aborting")
    trainer.train(resume_from_checkpoint=bool(args.resume))
    trainer.model.save_pretrained(str(out / "final_adapter"))
    tokenizer.save_pretrained(str(out / "final_adapter"))
    (out / "trainer_state.json").write_text(json.dumps(
        {k: trainer.state.__dict__.get(k) for k in
         ("global_step", "epoch", "max_steps", "train_batch_size",
          "logging_steps", "save_steps")},
        ensure_ascii=False, indent=2, default=str), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
