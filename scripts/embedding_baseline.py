"""
Run Qwen3-Embedding-8B via SiliconFlow API for LexRAG + LawBench.
Saves embeddings for reuse.
"""
import json, os, re, time, numpy as np

ROOT = r"D:\download\LexRAG-main-new"
OUT = os.path.join(ROOT, "reproduction_runs", "api_full_runs")
os.makedirs(OUT, exist_ok=True)

SILICON_KEY = os.getenv("SILICON", "") or os.getenv("SILICONFLOW_API_KEY", "")
EMBED_MODEL = "Qwen/Qwen3-Embedding-8B"
BASE_URL = "https://api.siliconflow.cn/v1"
BATCH_SIZE = 50

if not SILICON_KEY:
    print("ERROR: SILICON not set"); exit(1)

from openai import OpenAI
client = OpenAI(api_key=SILICON_KEY, base_url=BASE_URL)

# ── Normalization ──
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

def norm_for_match(name):
    """Normalize for matching: convert Chinese→Arabic numerals"""
    name = re.sub(r'\s+', '', str(name)).replace('《','').replace('》','')
    name = re.sub(r'第([零一二三四五六七八九十百千万\d〇○两]+)条', lambda m: f'第{c2i(m.group(1))}条', name)
    return name

# ── Load law library ──
LAW_PATH = os.path.join(ROOT, "LexRAG-main", "data", "law_library.jsonl")
law_names = []
law_texts = []
with open(LAW_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip(): continue
        e = json.loads(line)
        law_names.append(e.get('name', ''))
        law_texts.append(e.get('content', ''))

print(f"Law library: {len(law_names)} articles")

# ── Embed law articles (with cache) ──
LAW_CACHE = os.path.join(OUT, "qwen3_emb_law_cache.npy")
if os.path.exists(LAW_CACHE):
    print(f"Loading law embeddings from cache: {LAW_CACHE}")
    law_embs = np.load(LAW_CACHE)
else:
    print(f"Embedding {len(law_texts)} law articles via SiliconFlow...")
    all_embs = []
    for i in range(0, len(law_texts), BATCH_SIZE):
        batch = law_texts[i:i+BATCH_SIZE]
        for attempt in range(5):
            try:
                resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
                all_embs.extend([d.embedding for d in resp.data])
                break
            except Exception as e:
                if attempt == 4: raise
                time.sleep(2 ** attempt)
        if (i + BATCH_SIZE) % 500 == 0:
            print(f"  Embedded {min(i+BATCH_SIZE, len(law_texts))}/{len(law_texts)}")
    law_embs = np.array(all_embs, dtype=np.float32)
    np.save(LAW_CACHE, law_embs)
    print(f"Law embeddings saved: {law_embs.shape}")

# Normalize law embeddings
law_norm = law_embs / (np.linalg.norm(law_embs, axis=1, keepdims=True) + 1e-8)

# ── Run embedding retrieval on a dataset ──
def run_embedding_retrieval(dataset_items, dataset_name, output_prefix):
    Q_CACHE = os.path.join(OUT, f"{output_prefix}_q_embs.npy")

    questions = [it['q'] for it in dataset_items]

    if os.path.exists(Q_CACHE):
        print(f"Loading question embeddings from cache")
        q_embs = np.load(Q_CACHE)
    else:
        print(f"Embedding {len(questions)} questions for {dataset_name}...")
        all_q_embs = []
        for i in range(0, len(questions), BATCH_SIZE):
            batch = questions[i:i+BATCH_SIZE]
            for attempt in range(5):
                try:
                    resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
                    all_q_embs.extend([d.embedding for d in resp.data])
                    break
                except Exception as e:
                    if attempt == 4: raise
                    time.sleep(2 ** attempt)
            if (i + BATCH_SIZE) % 200 == 0:
                print(f"  Embedded {min(i+BATCH_SIZE, len(questions))}/{len(questions)}")
        q_embs = np.array(all_q_embs, dtype=np.float32)
        np.save(Q_CACHE, q_embs)

    q_norm = q_embs / (np.linalg.norm(q_embs, axis=1, keepdims=True) + 1e-8)

    # Compute cosine similarity and top-5
    print(f"Computing top-5 retrieval for {dataset_name}...")
    dense_scores = q_norm @ law_norm.T  # [n_questions, n_laws]

    total_p = total_r = total_f1 = 0
    n_with_gold = 0
    results = []

    for i, item in enumerate(dataset_items):
        top5_idx = np.argsort(dense_scores[i])[::-1][:5]
        pred_articles = [normalize_article(law_names[j]) for j in top5_idx]

        gt = item['gt']
        gt_set = set(norm_for_match(a) for a in gt)
        pred_set = set(norm_for_match(a) for a in pred_articles)

        if not gt_set:
            results.append({'id': item['id'], 'pred': pred_articles, 'gt': gt, 'P': 0, 'R': 0, 'F1': 0})
            continue

        n_with_gold += 1
        tp = len(gt_set & pred_set)
        pr = tp / len(pred_set) if pred_set else 0
        rc = tp / len(gt_set)
        f1 = 2*pr*rc/(pr+rc) if (pr+rc) > 0 else 0
        total_p += pr; total_r += rc; total_f1 += f1
        results.append({'id': item['id'], 'pred': pred_articles, 'gt': gt, 'P': pr, 'R': rc, 'F1': f1})

    avg_p = round(total_p / n_with_gold, 4) if n_with_gold else 0
    avg_r = round(total_r / n_with_gold, 4) if n_with_gold else 0
    avg_f1 = round(total_f1 / n_with_gold, 4) if n_with_gold else 0

    summary = {
        "model": EMBED_MODEL, "provider": "SiliconFlow API",
        "dataset": dataset_name, "samples": len(dataset_items),
        "with_gold": n_with_gold,
        "P@5": avg_p, "R@5": avg_r, "F1@5": avg_f1
    }

    with open(os.path.join(OUT, f"{output_prefix}_summary.json"), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{dataset_name} Embedding Retrieval:")
    print(f"  P@5={avg_p}, R@5={avg_r}, F1@5={avg_f1}")
    return summary

# ── Load LexRAG data ──
LEX_FIRST_TURN = os.path.join(ROOT, "LexRAG-main", "data", "first_turn_by_law.json")
with open(LEX_FIRST_TURN, 'r', encoding='utf-8') as f:
    lex_data = json.load(f)

lex_items = []
for entry in lex_data:
    q = entry.get('question', '')
    gt = entry.get('articles', entry.get('gt_articles', []))
    if q:
        lex_items.append({'id': str(entry.get('id', len(lex_items))), 'q': q.strip(),
                         'gt': gt if isinstance(gt, list) else [gt]})
print(f"\nLexRAG items: {len(lex_items)}")

# ── Load LawBench data ──
LB_PATH = os.path.join(ROOT, "experiments", "lawbench", "data", "lawbench_processed_v2.json")
with open(LB_PATH, 'r', encoding='utf-8') as f:
    lb_data = json.load(f)

lb_items = []
for i, entry in enumerate(lb_data):
    q = entry.get('user', '')
    gt = entry.get('article', [])
    if q and gt:
        lb_items.append({'id': str(i), 'q': q.strip(), 'gt': gt if isinstance(gt, list) else [gt]})
print(f"LawBench items: {len(lb_items)}")

# ── Run on both datasets ──
print("\n" + "=" * 60)
s1 = run_embedding_retrieval(lex_items, "LexRAG (1013)", "lexrag_1013_embedding_qwen3")
print("Paper LexRAG Embedding: P@5=0.0492 R@5=0.2178 F1@5=0.0789")

print("\n" + "=" * 60)
s2 = run_embedding_retrieval(lb_items, "LawBench (439)", "lawbench_439_embedding_qwen3")
print("Paper LawBench Embedding: P@5=0.0779 R@5=0.2338 F1@5=0.1120")

# Save combined
combined = {"lexrag": s1, "lawbench": s2}
with open(os.path.join(OUT, "embedding_all_summaries.json"), 'w', encoding='utf-8') as f:
    json.dump(combined, f, ensure_ascii=False, indent=2)
print("\n=== EMBEDDING BASELINE COMPLETE ===")
