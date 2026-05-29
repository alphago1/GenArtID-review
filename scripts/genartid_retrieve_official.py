"""Table 2 右半数据重跑：各法律子集 × 两个模型的 LLM-RAG top-5 检索实验。

模型:
  - deepseek-ai/DeepSeek-V3.2 (SiliconFlow)
  - Qwen/Qwen3-8B (SiliconFlow)

数据集（从 dataset.json 第一轮提取，article 开头匹配）:
  - civil_code_330.json (349 样本)
  - criminal_law_all.json (88 样本)
  - work_injury_all.json (57 样本)

Usage:
  set SILICON=sk-xxx
  python run.py --dataset all --model all
  python run.py --dataset civil_code --model qwen3-8b
"""
import argparse
import json
import os
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

# ── Config ──────────────────────────────────────────────
API_KEY = os.getenv("siliconflow_key", "") or os.getenv("SILICON", "") or os.getenv("SILICONFLOW_API_KEY", "")
BASE_URL = "https://api.siliconflow.cn/v1"
DS_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DS_BASE_URL = "https://api.deepseek.com"
MODELS = {
    "deepseek-v3.2": "deepseek-ai/DeepSeek-V3.2",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "deepseek-v4-flash": "deepseek-v4-flash",
    "deepseek-v4-pro": "deepseek-v4-pro",
}
# Base URLs and API keys per model
MODEL_CONFIG = {
    "deepseek-v3.2": {"base_url": "https://api.siliconflow.cn/v1", "api_key_env": "siliconflow_key"},
    "qwen3-8b": {"base_url": "https://api.siliconflow.cn/v1", "api_key_env": "siliconflow_key"},
    "deepseek-v4-flash": {"base_url": "https://api.deepseek.com", "api_key_env": "DEEPSEEK_API_KEY"},
    "deepseek-v4-pro": {"base_url": "https://api.deepseek.com", "api_key_env": "DEEPSEEK_API_KEY"},
}

TOP5_SYSTEM_PROMPT = """你是一个专业的中国法律检索专家。

你的任务不是作答，而是为后续数据库检索生成候选法条。
请根据用户案情，输出最相关的5条法条编号，按相关性从高到低排序。

硬性要求：
1. 必须恰好输出5条；
2. 每条都必须是"《法律名称》第XXX条"；
3. 不要输出解释、分析、理由、序号或其他任何文字；
4. 即使不确定，也必须给出5条最可能的候选法条；
5. 优先输出最核心、最直接适用的法条；
6. 只输出 JSON，不要输出其他内容。

输出格式：
{
  "articles": [
    "《法律名称》第XXX条",
    "《法律名称》第XXX条",
    "《法律名称》第XXX条",
    "《法律名称》第XXX条",
    "《法律名称》第XXX条"
  ]
}"""

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS = {
    "civil_code": os.path.join(SCRIPT_DIR, "data", "civil_code_330.json"),
    "criminal_law": os.path.join(SCRIPT_DIR, "data", "criminal_law_all.json"),
    "work_injury": os.path.join(SCRIPT_DIR, "data", "work_injury_all.json"),
    "all_first_turn": os.path.join(SCRIPT_DIR, "..", "..", "data", "first_turn_by_law.json"),
}
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

print_lock = threading.Lock()

# ── Article Normalization ────────────────────────────────

def chinese_to_arabic(cn_str: str) -> int:
    cn_num_map = {'零': 0, '一': 1, '二': 2, '三': 3, '四': 4,
                   '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
                   '十': 10, '百': 100, '千': 1000}
    if not cn_str:
        return 0
    if cn_str.isdigit():
        return int(cn_str)
    result, temp = 0, 0
    for ch in cn_str:
        if ch not in cn_num_map:
            continue
        val = cn_num_map[ch]
        if val >= 10:
            if temp == 0: temp = 1
            result += temp * val; temp = 0
        else:
            temp = val
    return result + temp


def normalize_article(ref: str) -> str:
    """Normalize to canonical form: 《法律名》第N条"""
    if not ref:
        return ""
    ref = ref.strip()
    m = re.match(r'《(.+?)》(.+)', ref)
    if not m:
        return ref
    law_name = m.group(1).strip().replace("中华人民共和国", "")
    article_part = m.group(2).strip()

    num_m = re.search(r'第?([零一二三四五六七八九十百千\d]+)条', article_part)
    if not num_m:
        return f"《{law_name}》{article_part}"
    return f"《{law_name}》第{chinese_to_arabic(num_m.group(1))}条"


def extract_articles(text: str) -> list:
    """Parse model response → top-5 normalized article refs."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)

    articles = []
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            arts = obj.get("articles", [])
            if isinstance(arts, list):
                articles = [normalize_article(a) for a in arts if a]
    except Exception:
        pass

    if not articles:
        found = re.findall(r'《[^》]+》第[零一二三四五六七八九十百千\d]+条', text)
        articles = [normalize_article(a) for a in found[:5]]

    seen, unique = set(), []
    for a in articles:
        if a and a not in seen:
            seen.add(a); unique.append(a)
    return unique[:5]


def compute_metrics(gt_articles: list, pred_articles: list):
    gt_set = set(normalize_article(a) for a in gt_articles)
    pred_set = set(pred_articles[:5])
    if not gt_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    tp = len(gt_set & pred_set)
    p = tp / len(pred_set) if pred_set else 0.0
    r = tp / len(gt_set)
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1}


# ── Data Loading ─────────────────────────────────────────

def load_data(dataset_name: str) -> list:
    with open(DATASETS[dataset_name], 'r', encoding='utf-8') as f:
        data = json.load(f)
    items = []
    for entry in data:
        if 'conversation' in entry:
            t0 = entry['conversation'][0]
            q = t0.get('user', '')
            gt = t0.get('article', [])
        else:
            q = entry.get('question', '')
            gt = entry.get('articles', entry.get('gt_articles', []))
        if q:
            items.append({
                'id': entry.get('id', ''),
                'question': q.strip(),
                'gt_articles': gt if isinstance(gt, list) else [gt],
            })
    return items


# ── API Call ─────────────────────────────────────────────

def call_llm(client: OpenAI, model_name: str, question: str, model_key: str = "", max_retries: int = 3) -> str:
    messages = [
        {"role": "system", "content": TOP5_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    kwargs = {"model": model_name, "messages": messages, "temperature": 0.0}
    # no_think for DeepSeek models
    if "deepseek" in model_key.lower():
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                time.sleep(3 * (attempt + 1))
            else:
                time.sleep(1)
    return ""


# ── Runner ───────────────────────────────────────────────

def run_dataset(dataset_name: str, model_key: str, max_workers: int = 8):
    model_display = MODELS[model_key]
    items = load_data(dataset_name)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"{dataset_name}_{model_key}.jsonl")

    # Resume
    done_ids = set()
    if os.path.exists(out_path):
        with open(out_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)['id'])
                except Exception:
                    pass

    pending = [it for it in items if it['id'] not in done_ids]
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}  |  Total={len(items)}  Pending={len(pending)}")
    print(f"Model: {model_display}")
    print(f"{'='*60}")

    if pending:
        cfg = MODEL_CONFIG[model_key]
        api_key = os.getenv(cfg["api_key_env"], "")
        if not api_key:
            if cfg["api_key_env"] == "siliconflow_key":
                api_key = os.getenv("SILICON", "") or os.getenv("SILICONFLOW_API_KEY", "")
        client = OpenAI(api_key=api_key, base_url=cfg["base_url"])
        def proc(it):
            resp = call_llm(client, model_display, it['question'], model_key=model_key)
            preds = extract_articles(resp)
            m = compute_metrics(it['gt_articles'], preds)
            r = {"id": it['id'], "question": it['question'][:100],
                 "gt_articles": it['gt_articles'], "pred_articles": preds,
                 "raw_response": resp[:500],
                 "precision": round(m['precision'], 6),
                 "recall": round(m['recall'], 6),
                 "f1": round(m['f1'], 6)}
            with print_lock:
                print(f"  [{it['id']}] P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}")
            return r

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(proc, it): it for it in pending}
            for fut in as_completed(futs):
                r = fut.result()
                with open(out_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Aggregate
    all_r = [json.loads(l) for l in open(out_path, 'r', encoding='utf-8') if l.strip()]
    n = len(all_r)
    if n == 0:
        print("No results!"); return None
    avg = lambda k: sum(r[k] for r in all_r) / n
    s = {"dataset": dataset_name, "model": model_display, "total": n,
         "precision_at_5": round(avg('precision'), 4),
         "recall_at_5": round(avg('recall'), 4),
         "f1_at_5": round(avg('f1'), 4)}
    sp = os.path.join(RESULTS_DIR, f"{dataset_name}_{model_key}_summary.json")
    with open(sp, 'w', encoding='utf-8') as f:
        json.dump(s, f, ensure_ascii=False, indent=2)
    print(f"  => P@5={s['precision_at_5']:.4f}  R@5={s['recall_at_5']:.4f}  F1@5={s['f1_at_5']:.4f}")
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="all",
                   choices=["civil_code", "criminal_law", "work_injury", "all_first_turn", "all"])
    p.add_argument("--model", default="all",
                   choices=["deepseek-v3.2", "qwen3-8b", "deepseek-v4-flash", "deepseek-v4-pro", "all"])
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    sf_key = os.getenv("siliconflow_key", "") or os.getenv("SILICON", "")
    ds_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not sf_key and not ds_key:
        print("ERROR: No API key found (siliconflow_key or DEEPSEEK_API_KEY)"); sys.exit(1)

    ds_list = (["civil_code", "criminal_law", "work_injury"] if args.dataset == "all"
               else [args.dataset])
    mk_list = (["deepseek-v3.2", "qwen3-8b"] if args.model == "all"
               else [args.model])

    summaries = []
    for ds in ds_list:
        for mk in mk_list:
            s = run_dataset(ds, mk, max_workers=args.workers)
            if s: summaries.append(s)

    print("\n" + "=" * 70)
    print("TABLE 2 RIGHT HALF — FINAL RESULTS")
    print("=" * 70)
    print(f"{'Dataset':<20} {'Model':<28} {'P@5':>8} {'R@5':>8} {'F1@5':>8}")
    print("-" * 70)
    for s in summaries:
        print(f"{s['dataset']:<20} {s['model']:<28} {s['precision_at_5']:>8.4f} {s['recall_at_5']:>8.4f} {s['f1_at_5']:>8.4f}")
    print("-" * 70)


if __name__ == "__main__":
    main()
