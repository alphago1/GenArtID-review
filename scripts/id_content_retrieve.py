"""LLM-RAG with article content output — compares ID-only vs ID+content performance.
Also verifies whether outputted content matches the law library.
"""

import json, os, re, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# ── Config ──────────────────────────────────────────────
NEW_PROMPT = """你是一个专业的中国法律检索专家。

你的任务不是作答，而是为后续数据库检索生成候选法条。
请根据用户案情，输出最相关的5条法条编号及其内容，按相关性从高到低排序。

硬性要求：
1. 必须恰好输出5条；
2. 每条都必须是"《法律名称》第XXX条：内容"；
3. 不要输出解释、分析、理由、序号或其他任何文字；
4. 即使不确定，也必须给出5条最可能的候选法条；
5. 优先输出最核心、最直接适用的法条；
6. 只输出 JSON，不要输出其他内容。

输出格式：
{
  "articles": [
    "《法律名称》第XXX条：内容",
    "《法律名称》第XXX条：内容",
    "《法律名称》第XXX条：内容",
    "《法律名称》第XXX条：内容",
    "《法律名称》第XXX条：内容"
  ]
}"""

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print_lock = threading.Lock()
MAX_CONCURRENT = 10

# ── Article Normalization ────────────────────────────────
def chinese_to_arabic(cn_str: str) -> int:
    cn_num_map = {'零': 0, '一': 1, '二': 2, '三': 3, '四': 4,
                   '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
                   '十': 10, '百': 100, '千': 1000}
    if not cn_str: return 0
    if cn_str.isdigit(): return int(cn_str)
    r, t = 0, 0
    for ch in cn_str:
        v = cn_num_map.get(ch)
        if v is None: continue
        if v >= 10:
            if t == 0: t = 1
            r += t * v; t = 0
        else: t = v
    return r + t

def normalize_article(ref: str) -> str:
    if not ref: return ""
    ref = ref.strip()
    m = re.match(r'《(.+?)》(.+)', ref)
    if not m: return ref
    law = m.group(1).replace("中华人民共和国", "")
    part = m.group(2)
    nm = re.search(r'第?([零一二三四五六七八九十百千\d]+)条', part)
    if not nm: return f"《{law}》{part}"
    return f"《{law}》第{chinese_to_arabic(nm.group(1))}条"

def parse_article_with_content(text: str):
    """Parse '《Law》第N条：content' into (normalized_id, content)."""
    text = text.strip()
    m = re.match(r'《(.+?)》\s*第?([零一二三四五六七八九十百千\d]+)条\s*[：:]\s*(.*)', text, re.DOTALL)
    if not m:
        return None, None
    law = m.group(1).replace("中华人民共和国", "")
    num = chinese_to_arabic(m.group(2))
    content = m.group(3).strip()
    return f"《{law}》第{num}条", content

def extract_articles(text: str):
    """Extract top-5 article IDs from model response."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r'^```\w*\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "articles" in data:
            return data["articles"]
    except json.JSONDecodeError: pass
    refs = re.findall(r'《[^》]+》第[^条]+条', text)
    return refs if refs else []

# ── Law Library for Content Verification ─────────────────
def load_law_library():
    laws = {}
    law_path = os.path.join(SCRIPT_DIR, "..", "..", "data", "law_library.jsonl")
    with open(law_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                name = item["name"]
                content = item["content"]
                norm_name = normalize_article(name)
                laws[norm_name] = content
    return laws

# Simple character-level similarity
def content_similarity(predicted: str, actual: str) -> float:
    if not predicted or not actual:
        return 0.0
    # Normalize whitespace and punctuation
    def clean(s):
        return re.sub(r'\s+', '', s).replace('；',';').replace('，',',').replace('。','.')
    p = clean(predicted)
    a = clean(actual)
    if len(p) == 0 or len(a) == 0:
        return 0.0
    # Longest common substring ratio
    # Use character overlap
    common = set(p) & set(a)
    if not common:
        return 0.0
    return len(common) / max(len(set(p)), len(set(a)))

# ── Query ────────────────────────────────────────────────
def query_one(client, model_name, model_key, q, idx, total, law_lib):
    for attempt in range(5):
        try:
            kwargs = {"model": model_name, "messages": [
                {"role": "system", "content": NEW_PROMPT},
                {"role": "user", "content": q["question"]},
            ], "temperature": 0.0}
            if "deepseek" in model_key.lower():
                kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            resp = client.chat.completions.create(**kwargs)
            raw = resp.choices[0].message.content or ""
        except Exception as e:
            if attempt == 4: return idx, "", [], [], [], str(e)
            time.sleep(2 ** attempt)
            continue

        articles_raw = extract_articles(raw)

        # Parse ID + content from each article
        pred_ids = []
        pred_contents = []
        for art in articles_raw[:5]:
            nid, content = parse_article_with_content(art)
            if nid:
                pred_ids.append(nid)
                pred_contents.append(content)
            else:
                pred_ids.append(normalize_article(art))
                pred_contents.append("")

        # Verify content against law library
        content_sims = []
        for nid, pcontent in zip(pred_ids, pred_contents):
            if nid in law_lib and pcontent:
                sim = content_similarity(pcontent, law_lib[nid])
                content_sims.append(round(sim, 4))
            else:
                content_sims.append(None)

        with print_lock:
            print(f"[{idx+1}/{total}] id={q['id']} pred={pred_ids}")

        return idx, raw, pred_ids, pred_contents, content_sims, None

    return idx, "", [], [], [], "unknown"

# ── Main ──────────────────────────────────────────────────
def run_model(model_key, model_name, client, questions, law_lib):
    results = [None] * len(questions)
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as pool:
        futures = {pool.submit(query_one, client, model_name, model_key, q, i, len(questions), law_lib): i
                   for i, q in enumerate(questions)}
        for f in as_completed(futures):
            idx, raw, pred_ids, pred_contents, content_sims, err = f.result()
            if err:
                with print_lock:
                    print(f"[{idx+1}] ERROR: {err}")
            results[idx] = {
                "id": questions[idx]["id"],
                "question": questions[idx]["question"][:100],
                "gt_articles": questions[idx]["articles"],
                "raw_response": raw,
                "pred_ids": pred_ids,
                "pred_contents": pred_contents,
                "content_similarity": content_sims,
            }

    # Compute metrics
    per_sample = []
    for res in results:
        gt_norm = {normalize_article(a) for a in res["gt_articles"]}
        pred_set = set(res["pred_ids"])
        tp = len(gt_norm & pred_set)
        p = tp / len(pred_set) if pred_set else 0.0
        r = tp / len(gt_norm) if gt_norm else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        res["precision"] = p
        res["recall"] = r
        res["f1"] = f1
        per_sample.append({"id": res["id"], "precision": p, "recall": r, "f1": f1})

    n = len(results)
    avg_p = sum(r["precision"] for r in results) / n
    avg_r = sum(r["recall"] for r in results) / n
    avg_f1 = sum(r["f1"] for r in results) / n

    # Content verification stats
    content_scores = []
    for res in results:
        for s in res.get("content_similarity", []):
            if s is not None:
                content_scores.append(s)
    avg_content_sim = sum(content_scores) / len(content_scores) if content_scores else 0.0

    return {
        "model": model_name, "model_key": model_key,
        "samples": n,
        "precision_at_5": round(avg_p, 4),
        "recall_at_5": round(avg_r, 4),
        "f1_at_5": round(avg_f1, 4),
        "avg_content_similarity": round(avg_content_sim, 4),
        "content_pairs_count": len(content_scores),
        "per_sample": per_sample,
        "full_results": results,
    }

def main():
    # Load data
    data_path = os.path.join(SCRIPT_DIR, "..", "..", "data", "first_turn_by_law.json")
    with open(data_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} questions")

    print("Loading law library...")
    law_lib = load_law_library()
    print(f"Law library: {len(law_lib)} articles")

    os.makedirs(os.path.join(SCRIPT_DIR, "results"), exist_ok=True)

    all_summaries = []

    # ── DeepSeek V4 Flash ──
    ds_key = os.getenv("DEEPSEEK_API_KEY", "")
    if ds_key:
        print("\n" + "="*60)
        print("Running: DeepSeek V4 Flash (ID+Content)")
        print("="*60)
        client = OpenAI(api_key=ds_key, base_url="https://api.deepseek.com")
        summary = run_model("deepseek-v4-flash", "deepseek-v4-flash", client, questions, law_lib)
        all_summaries.append(summary)

        with open(os.path.join(SCRIPT_DIR, "results", "id_content_deepseek_v4flash.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"V4 Flash: P@5={summary['precision_at_5']} R@5={summary['recall_at_5']} F1@5={summary['f1_at_5']} ContentSim={summary['avg_content_similarity']}")

    # ── Qwen3-8B (SiliconFlow) ──
    sf_key = os.getenv("siliconflow_key", "") or os.getenv("SILICON", "")
    if sf_key:
        print("\n" + "="*60)
        print("Running: Qwen3-8B (ID+Content)")
        print("="*60)
        client = OpenAI(api_key=sf_key, base_url="https://api.siliconflow.cn/v1")
        summary = run_model("qwen3-8b", "Qwen/Qwen3-8B", client, questions, law_lib)
        all_summaries.append(summary)

        with open(os.path.join(SCRIPT_DIR, "results", "id_content_qwen3_8b.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Qwen3-8B: P@5={summary['precision_at_5']} R@5={summary['recall_at_5']} F1@5={summary['f1_at_5']} ContentSim={summary['avg_content_similarity']}")

    # ── Comparison ──
    print("\n" + "="*60)
    print("COMPARISON: ID-Only vs ID+Content")
    print("="*60)
    print(f"{'Model':<25} {'Mode':<15} {'P@5':>8} {'R@5':>8} {'F1@5':>8} {'ContentSim':>10}")
    print("-"*75)
    # ID-only baselines (from earlier runs)
    baselines = {
        ("DeepSeek V4 Flash", "ID-Only"): (0.1171, 0.5004, 0.1860, None),
        ("Qwen3-8B", "ID-Only"): (0.1106, 0.4894, 0.1783, None),  # from criminal_procedure... wait, need full 1013
    }
    for s in all_summaries:
        label = "DeepSeek V4 Flash" if "flash" in s["model_key"] else "Qwen3-8B"
        print(f"{label:<25} {'ID+Content':<15} {s['precision_at_5']:>8.4f} {s['recall_at_5']:>8.4f} {s['f1_at_5']:>8.4f} {s['avg_content_similarity']:>10.4f}")

    # Save comparison
    with open(os.path.join(SCRIPT_DIR, "results", "id_content_comparison.json"), "w", encoding="utf-8") as f:
        json.dump({
            "id_only_baselines": {
                "v4_flash": {"P@5": 0.1171, "R@5": 0.5004, "F1@5": 0.1860},
                # Qwen3-8B full 1013 data - we might have it from earlier runs
            },
            "id_content_results": [{
                "model": s["model_key"],
                "P@5": s["precision_at_5"],
                "R@5": s["recall_at_5"],
                "F1@5": s["f1_at_5"],
                "content_similarity": s["avg_content_similarity"],
            } for s in all_summaries],
        }, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
