"""
Recompute Table 5: Candidate ranking quality (Avg Correct Rank, Top-1/3/5 Hit)
Reads from the actual result files (not old cached data).

Data sources:
  - Civil Code: base_local_per_sample.json, checkpoint-280.json
  - Criminal Procedure Law: base.json, checkpoint-180.json

Metrics:
  - Avg Correct Rank: mean rank (1-5) of the first correct gold article, among samples where
    at least one correct article is found. Lower is better.
  - Top-1 Hit: % of ALL samples where a correct article is at rank 1.
  - Top-3 Hit: % of ALL samples where a correct article is at rank <= 3.
  - Top-5 Hit: % of ALL samples where a correct article is at rank <= 5.
"""
import json
import os
import re

# --- Paths ---
CIVIL_BASE = r"D:\download\LexRAG-main-new\民法典\civilresults\results\base_local_per_sample.json"
CIVIL_CKPT = r"D:\download\LexRAG-main-new\民法典\civilresults\results\checkpoint-280.json"
CPL_BASE   = r"D:\download\LexRAG-main-new\刑事诉讼法\刑事诉讼法的结果\base.json"
CPL_CKPT   = r"D:\download\LexRAG-main-new\刑事诉讼法\刑事诉讼法的结果\checkpoint-180.json"

OUT_DIR = r"D:\download\LexRAG-main-new\paper_data\table5_ranking_recomputed"


def normalize_article(a):
    """Normalize article ID for comparison: strip spaces, unify brackets."""
    a = a.strip()
    a = re.sub(r'\s+', '', a)
    a = a.replace('（', '(').replace('）', ')')
    return a


def find_correct_rank(pred_list, gold_list):
    """Return 1-based rank of first correct article, or None if none found."""
    gold_set = {normalize_article(g) for g in gold_list}
    for i, p in enumerate(pred_list):
        if normalize_article(p) in gold_set:
            return i + 1
    return None


def compute_metrics(samples, label):
    """Compute Avg Correct Rank, Top-1 Hit, Top-3 Hit, Top-5 Hit."""
    total = len(samples)
    ranks = []           # only for samples with at least one correct
    top1 = top3 = top5 = 0

    for s in samples:
        pred = s.get('pred', s.get('predictions', []))
        gold = s.get('gold', s.get('gt', []))

        if not pred or not gold:
            continue

        rank = find_correct_rank(pred, gold)
        if rank is not None:
            ranks.append(rank)
            top5 += 1
            if rank <= 3:
                top3 += 1
            if rank == 1:
                top1 += 1

    avg_rank = sum(ranks) / len(ranks) if ranks else float('nan')

    return {
        'label': label,
        'samples': total,
        'samples_with_correct': len(ranks),
        'avg_correct_rank': round(avg_rank, 2),
        'top1_hit': round(top1 / total * 100, 1),
        'top3_hit': round(top3 / total * 100, 1),
        'top5_hit': round(top5 / total * 100, 1),
        'top1_count': top1,
        'top3_count': top3,
        'top5_count': top5,
    }


# --- Load data ---
def load_civil_base():
    with open(CIVIL_BASE, 'r', encoding='utf-8') as f:
        return json.load(f)['per_sample']

def load_civil_ckpt():
    with open(CIVIL_CKPT, 'r', encoding='utf-8') as f:
        return json.load(f)['ps']

def load_cpl_base():
    with open(CPL_BASE, 'r', encoding='utf-8') as f:
        return json.load(f)['results']

def load_cpl_ckpt():
    with open(CPL_CKPT, 'r', encoding='utf-8') as f:
        return json.load(f)['per_sample']


# --- Compute ---
results = []

for samples, label in [
    (load_civil_base(), 'Civil Code | Qwen3-8B Base (4-bit)'),
    (load_civil_ckpt(), 'Civil Code | Qwen3-8B LoRA (checkpoint-280)'),
    (load_cpl_base(),   'Criminal Procedure Law | Qwen3-8B Base'),
    (load_cpl_ckpt(),   'Criminal Procedure Law | Qwen3-8B LoRA (checkpoint-180)'),
]:
    m = compute_metrics(samples, label)
    results.append(m)
    print(f"{label}")
    print(f"  Samples: {m['samples']}")
    print(f"  With correct: {m['samples_with_correct']}")
    print(f"  Avg Correct Rank: {m['avg_correct_rank']}")
    print(f"  Top-1 Hit: {m['top1_hit']}% ({m['top1_count']}/{m['samples']})")
    print(f"  Top-3 Hit: {m['top3_hit']}% ({m['top3_count']}/{m['samples']})")
    print(f"  Top-5 Hit: {m['top5_hit']}% ({m['top5_count']}/{m['samples']})")
    print()

# --- Save results ---
output = {
    'description': 'Table 5: Candidate ranking quality after knowledge injection (recomputed from actual result files)',
    'data_sources': {
        'civil_base': CIVIL_BASE,
        'civil_checkpoint': CIVIL_CKPT,
        'cpl_base': CPL_BASE,
        'cpl_checkpoint': CPL_CKPT,
    },
    'results': results,
}

out_path = os.path.join(OUT_DIR, 'ranking_metrics.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f'Saved to {out_path}')

# --- Print LaTeX / Markdown table ---
print()
print('=' * 70)
print('MARKDOWN TABLE')
print('=' * 70)
print()
print('| Legal Text | Model | Avg Correct Rank | Top-1 Hit | Top-3 Hit | Top-5 Hit |')
print('|---|---:|---:|---:|---:|---:|')
for r in results:
    parts = r['label'].split(' | ')
    lt = parts[0]
    model = parts[1]
    print(f'| {lt} | {model} | {r["avg_correct_rank"]} | {r["top1_hit"]}% | {r["top3_hit"]}% | {r["top5_hit"]}% |')

# --- Also compute per-sample details for verification ---
detail_path = os.path.join(OUT_DIR, 'per_sample_details.json')
details = {}
for samples, key in [
    (load_civil_base(), 'civil_base'),
    (load_civil_ckpt(), 'civil_checkpoint_280'),
    (load_cpl_base(),   'cpl_base'),
    (load_cpl_ckpt(),   'cpl_checkpoint_180'),
]:
    detail_list = []
    for s in samples:
        sid = s.get('id', '')
        q = s.get('q', s.get('question', ''))
        pred = s.get('pred', [])
        gold = s.get('gold', [])
        rank = find_correct_rank(pred, gold)
        detail_list.append({
            'id': sid,
            'question': q[:80],
            'gold': gold,
            'pred': pred,
            'correct_rank': rank,
        })
    details[key] = detail_list

with open(detail_path, 'w', encoding='utf-8') as f:
    json.dump(details, f, ensure_ascii=False, indent=2)
print(f'\nPer-sample details saved to {detail_path}')
