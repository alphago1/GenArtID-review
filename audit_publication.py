"""Audit released historical outputs on CPU. No generation, training or downloads."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
from ga_parsing import parse_fixed


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def lines(path):
    return [json.loads(x) for x in (ROOT / path).read_text(encoding="utf-8").splitlines() if x.strip()]


def aggregate(rows):
    values = {k: [] for k in ["P", "R", "F1", "H1", "H3", "H5", "MRR"]}
    positive_ranks = []
    for row in rows:
        pred, gold = row["predicted_ids"][:5], set(row["gold_ids"])
        hits = len(set(pred) & gold)
        p, r = hits / 5, hits / len(gold)
        rank = next((i for i, x in enumerate(pred, 1) if x in gold), None)
        for key, value in {"P": p, "R": r, "F1": 2 * p * r / (p + r) if p + r else 0,
                           "H1": int(rank is not None and rank <= 1), "H3": int(rank is not None and rank <= 3),
                           "H5": int(rank is not None), "MRR": 1 / rank if rank else 0}.items():
            values[key].append(value)
        if rank:
            positive_ranks.append(rank)
    return {**{k: statistics.mean(v) for k, v in values.items()}, "Rank": statistics.mean(positive_ranks)}


def main():
    subprocess.run([sys.executable, "-B", str(ROOT / "audit_results.py")], check=True)
    table1 = read("results/table1_main/index.json")
    main_questions = 0
    for entry in table1["generative_runs"]:
        rows = lines(entry["path"])
        if len({r["record_id"] for r in rows}) != len(rows):
            raise ValueError("Duplicate Table 1 record locator")
        ranks = []
        for row in rows:
            if row["evaluated"]:
                gold = set(row["gold_ids"])
                ranks.append(next((i for i, x in enumerate(row["predicted_ids"][:5], 1) if x in gold), None))
        metrics = [f"{sum(r is not None and r <= k for r in ranks) / len(ranks):.4f}" for k in [1, 3, 5]]
        metrics.append(f"{sum(1/r if r else 0 for r in ranks) / len(ranks):.4f}")
        if metrics != entry["printed_ranking"]:
            raise ValueError("Table 1 ranking differs: " + entry["model"])
        main_questions += len(rows)
    raw_index = read("results/fig3_table7/index.json")
    raw_memory = raw_retrieval = 0
    for run in raw_index["runs"]:
        key, domain = run["run_key"], run["domain"]
        memory, retrieval = lines(run["raw_recitations"]), lines(run["raw_retrieval"])
        scored_memory = read("results/per_article/" + key + "_" + domain + ".json")
        scored_retrieval = read("results/per_question/" + key + "_" + domain + ".json")
        if [r["item_id"] for r in memory] != [r["item_id"] for r in scored_memory]:
            raise ValueError("Raw recitation coverage/order differs")
        if [r["item_id"] for r in retrieval] != [r["item_id"] for r in scored_retrieval]:
            raise ValueError("Raw retrieval coverage/order differs")
        for raw, scored in zip(retrieval, scored_retrieval):
            if parse_fixed(raw["raw_response"]).predictions[:5] != scored["predicted_ids"]:
                raise ValueError("Raw response parser differs")
        if sum(not r["raw_response"].strip() for r in memory) != sum(r["empty_output"] for r in scored_memory):
            raise ValueError("Empty recitation accounting differs")
        raw_memory += len(memory)
        raw_retrieval += len(retrieval)
    ki = read("results/knowledge_injection.json")
    for domain, item in ki.items():
        base = aggregate(read("results/per_question/glm0414_base_" + domain + ".json"))
        results = []
        for run in item["runs"]:
            ft = aggregate(read("results/per_question/glm0414_seed" + str(run["seed"]) + "_" + domain + ".json"))
            for scores, expected in [(base, run["base"]), (ft, run["ft"])]:
                for key, value in expected.items():
                    if abs(scores[key] - value) > 1e-12:
                        raise ValueError("Table 2 metric differs: " + domain + " " + key)
            results.append(ft)
        for key in item["mean"]:
            if abs(statistics.mean(r[key] for r in results) - item["mean"][key]) > 1e-12 or abs(statistics.stdev(r[key] for r in results) - item["sd"][key]) > 1e-12:
                raise ValueError("Table 2 mean/SD differs")
    table3 = read("results/table3_ablation/ARCHIVE_MANIFEST.json")
    for entry in table3["files"]:
        if hashlib.sha256((ROOT / "results/table3_ablation" / entry["name"]).read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Table 3 differs from archived bytes")
    print(json.dumps({"status": "PASS", "table1_verified_generative_runs": len(table1["generative_runs"]),
                      "table1_question_model_records": main_questions, "table2_seed_runs": 6,
                      "table3_verbatim_files": len(table3["files"]), "raw_recitations": raw_memory,
                      "raw_retrieval_answers": raw_retrieval, "new_adapter_training_executed": False}))


if __name__ == "__main__":
    main()
