"""
Table 3 ID-Only: 重新跑 DeepSeek V4 Flash + Qwen3-8B 在 1013 条 LexRAG 问题上的检索结果
输出: ID-Only 的 P@5/R@5/F1@5
"""
import json, os, re, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── API Keys ──
DS_KEY = os.getenv("DEEPSEEK_API_KEY", "")
SF_KEY = os.getenv("SILICON", "") or os.getenv("SILICONFLOW_API_KEY", "")
if not DS_KEY: print("WARNING: DEEPSEEK_API_KEY not set, skipping DeepSeek")
if not SF_KEY: print("WARNING: SILICON not set, skipping Qwen3-8B")

from openai import OpenAI

# ── Paths ──
DATA_FILE = r"D:\download\LexRAG-main-new\LexRAG-main\data\first_turn_by_law.json"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── ID-Only Prompt (identical to paper Table 1) ──
ID_ONLY_PROMPT = """你是一个专业的中国法律检索专家。

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
{"articles": ["《法律名称》第XXX条", ...]}
"""

# ── Chinese numeral normalization ──
CN_M = {'零':0,'〇':0,'○':0,'一':1,'二':2,'两':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9}
CN_U = {'十':10,'百':100,'千':1000,'万':10000}
def c2i(t):
    if not t: return None
    t = str(t).strip().translate(str.maketrans('０１２３４５６７８９', '0123456789'))
    if t.isdigit(): return int(t)
    tot = sec = num = 0
    for c in t:
        if c in CN_M: num = CN_M[c]
        elif c in CN_U:
            u = CN_U[c]
            if u == 10000: sec = (sec + (num or 0)) * u; tot += sec; sec = num = 0
            else:
                if num == 0: num = 1
                sec += num * u; num = 0
        else: return None
    return tot + sec + num

def normalize_article(ref):
    if not ref: return ""
    ref = ref.strip(); ref = re.sub(r"\s+", "", ref)
    m = re.match(r'《(.+?)》(.+)', ref)
    if not m: return ref
    law_name = m.group(1).strip().replace("中华人民共和国", "")
    article_part = m.group(2).strip()
    num_m = re.search(r'第?([零一二三四五六七八九十百千万\d〇○两]+)条', article_part)
    if not num_m: return f"《{law_name}》{article_part}"
    n = c2i(num_m.group(1))
    if n is None: return f"《{law_name}》{article_part}"
    return f"《{law_name}》第{n}条"

def extract_articles(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r'^```(?:json)?\s*', '', text); text = re.sub(r'\s*```$', '', text)
    articles = []
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            arts = obj.get("articles", [])
            if isinstance(arts, list): articles = [normalize_article(a) for a in arts if a]
    except: pass
    if not articles:
        found = re.findall(r'《[^》]+》第[零一二三四五六七八九十百千\d〇○两]+条', text)
        articles = [normalize_article(a) for a in found[:5]]
    seen, unique = set(), []
    for a in articles:
        if a and a not in seen: seen.add(a); unique.append(a)
    return unique[:5]

# ── Load data ──
with open(DATA_FILE, 'r', encoding='utf-8') as f:
    raw = json.load(f)
questions = []
for entry in raw:
    q = entry.get('question', '')
    gt = entry.get('articles', entry.get('gt_articles', []))
    if q:
        questions.append({
            'id': str(entry.get('id', len(questions))),
            'q': q.strip(),
            'gt': gt if isinstance(gt, list) else [gt]
        })
print(f"Loaded {len(questions)} questions")

# ── Run one model ──
lock = threading.Lock()

def run_model(model_name, model_key, client, output_prefix, max_workers=8):
    out_jsonl = os.path.join(OUT_DIR, f"{output_prefix}_per_sample.jsonl")
    out_summary = os.path.join(OUT_DIR, f"{output_prefix}_summary.json")

    done_ids = set()
    if os.path.exists(out_jsonl):
        with open(out_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                try: done_ids.add(json.loads(line)['id'])
                except: pass

    pending = [q for q in questions if q['id'] not in done_ids]
    print(f"\nModel: {model_name} | Done: {len(done_ids)} | Pending: {len(pending)}")
    if not pending:
        print("All done, computing summary from existing data...")
    else:
        def proc(q):
            msgs = [
                {"role": "system", "content": ID_ONLY_PROMPT},
                {"role": "user", "content": q['q']}
            ]
            raw = ""
            for attempt in range(5):
                try:
                    kwargs = {"model": model_name, "messages": msgs, "temperature": 0.0, "max_tokens": 512}
                    if "deepseek" in model_key.lower():
                        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
                    resp = client.chat.completions.create(**kwargs)
                    raw = resp.choices[0].message.content or ""
                    break
                except Exception as e:
                    msg = str(e)
                    if any(s in msg for s in ["429","503","502","rate"]):
                        time.sleep(min(30, 2**attempt + 1)); continue
                    raw = "【ERROR】"; break

            preds = extract_articles(raw)
            gt_norm = [normalize_article(a) for a in q['gt']]
            gt_set = set(gt_norm)
            pred_set = set(preds)
            tp = len(gt_set & pred_set)
            p = tp / len(pred_set) if pred_set else 0
            r = tp / len(gt_set) if gt_set else 0
            f1 = 2*p*r/(p+r) if (p+r) > 0 else 0

            with lock:
                print(f"  [{q['id']}] P={p:.4f} R={r:.4f} F1={f1:.4f}")

            return {
                "id": q['id'], "q": q['q'][:200],
                "gt_articles": gt_norm,
                "pred_articles": preds,
                "precision": round(p, 6), "recall": round(r, 6), "f1": round(f1, 6),
                "raw_response": raw[:500]
            }

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(proc, q): q for q in pending}
            for fut in as_completed(futs):
                r = fut.result()
                with open(out_jsonl, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Compute summary
    all_r = [json.loads(l) for l in open(out_jsonl, 'r', encoding='utf-8') if l.strip()]
    golds = [r for r in all_r if r['gt_articles']]
    n = len(golds)
    mp = round(sum(r['precision'] for r in golds)/n, 4) if n else 0
    mr = round(sum(r['recall'] for r in golds)/n, 4) if n else 0
    mf = round(sum(r['f1'] for r in golds)/n, 4) if n else 0

    summary = {
        "model": model_name, "dataset": "LexRAG 1013",
        "samples": len(all_r), "with_gold": n,
        "P@5": mp, "R@5": mr, "F1@5": mf
    }
    with open(out_summary, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n  FINAL: P@5={mp} R@5={mr} F1@5={mf}")
    return summary

# ── Run DeepSeek V4 Flash ──
if DS_KEY:
    ds_client = OpenAI(api_key=DS_KEY, base_url="https://api.deepseek.com")
    run_model("deepseek-chat", "deepseek-v4-flash", ds_client, "id_only_deepseek")

# ── Run Qwen3-8B ──
if SF_KEY:
    sf_client = OpenAI(api_key=SF_KEY, base_url="https://api.siliconflow.cn/v1")
    run_model("Qwen/Qwen3-8B", "qwen3-8b", sf_client, "id_only_qwen3")

print("\n=== DONE ===")
