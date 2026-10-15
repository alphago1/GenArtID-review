"""Evaluate a saved adapter with the frozen Recall protocol; explicit --run."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
ADAPTERS = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "src"))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slot", required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--questions", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, help="Override for a newly trained adapter")
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    meta = read(ADAPTERS / args.slot / "metadata.json")
    data = next(e for e in read(ROOT / "data_manifest.json")["evaluation"] if e["domain"] == meta["domain"])
    if hashlib.sha256(args.questions.read_bytes()).hexdigest() != data["sha256"]:
        raise ValueError("Question file hash differs from frozen input")
    adapter = args.adapter or ADAPTERS / args.slot
    weight_sha = hashlib.sha256((adapter / "adapter_model.safetensors").read_bytes()).hexdigest()
    if args.adapter is None and weight_sha != meta["weight_sha256"]:
        raise ValueError("Released adapter hash mismatch")
    split = read(ROOT / "splits" / (meta["domain"] + ".json"))
    questions = {str(q["id"]): q for q in read(args.questions)}
    expected = read(ADAPTERS / args.slot / "per_question.json")
    from genartid.identifiers import canon_fixed
    from genartid.parsing import parse_fixed
    for row in expected:
        gold = list(dict.fromkeys(c for a in questions[row["item_id"]]["articles"] if (c := canon_fixed(a))))
        if set(gold) != set(row["gold_ids"]):
            raise ValueError("Gold differs from released record")
    print(json.dumps({"input_check": "PASS", "questions": len(split["test_ids"]),
                      "adapter_sha256": weight_sha, "executed": args.run}), flush=True)
    if not args.run:
        return
    if args.out.exists():
        raise ValueError("Use a fresh output directory")
    import torch
    from transformers import GenerationConfig, set_seed
    from genartid import local_memory_exec as local
    from genartid.pipeline import render_eval_prompt, stable_query_seed
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("CUDA with bf16 support is required")
    model, tokenizer = local._load_glm(str(args.model), str(adapter))
    tokenizer.padding_side = "left"
    generation = GenerationConfig(do_sample=False, num_beams=1, repetition_penalty=1.0,
                                  max_new_tokens=256, eos_token_id=[151329, 151336, 151338],
                                  pad_token_id=151329, use_cache=True)
    profile = read(ROOT / "configs/glm0414/base_ft_eval_profile.json")
    args.out.mkdir(parents=True)
    rows = []
    for qid in split["test_ids"]:
        q = questions[qid]
        encoded = tokenizer(render_eval_prompt(tokenizer, profile, q["question"]), return_tensors="pt").to("cuda:0")
        set_seed(stable_query_seed(meta["training_seed"], qid))
        with torch.inference_mode():
            generated = model.generate(**encoded, generation_config=generation)
        tokens = generated[0, encoded["input_ids"].shape[1]:].tolist()
        parsed = parse_fixed(tokenizer.decode(tokens, skip_special_tokens=True).strip())
        gold = list(dict.fromkeys(c for a in q["articles"] if (c := canon_fixed(a))))
        pred = parsed.predictions[:5]
        hits = len(set(pred) & set(gold))
        rows.append({"item_id": qid, "gold_ids": gold, "predicted_ids": pred,
                     "parse_status": parsed.status, "correct_count": hits, "gold_count": len(gold),
                     "R@5_exact": str(Fraction(hits, len(gold)) if gold else Fraction()),
                     "P@5_fixed": hits / 5})
        (args.out / "per_question.partial.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("Recall", len(rows), "/", len(split["test_ids"]), flush=True)
    (args.out / "per_question.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    exact = sum((Fraction(r["R@5_exact"]) for r in rows), Fraction()) / len(rows)
    (args.out / "metrics.json").write_text(json.dumps({
        "adapter_sha256": weight_sha, "n": len(rows), "recall_at_5_exact": str(exact),
        "recall_at_5": float(exact), "historical_exposure": "retrospective",
        "parser": "fixed_v2", "do_sample": False, "batch_size": 1,
        "max_new_tokens": 256}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
