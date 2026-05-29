"""
BM25 + Hybrid (BM25 + Qwen3-Emb-8B) baselines for both datasets.
Runs on LexRAG (1013) and Lawbench (439).
"""

import json, os, re, math, time, argparse
import numpy as np
from collections import Counter
from openai import OpenAI

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..", "..")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results", "table1")
os.makedirs(RESULTS_DIR, exist_ok=True)

LAW_LIBRARY_PATH = os.path.join(PROJECT_ROOT, "LexRAG-main", "data", "law_library.jsonl")
LAWBENCH_PATH = os.path.join(SCRIPT_DIR, "data", "lawbench_processed_v2.json")
LEXRAG_PATH = os.path.join(PROJECT_ROOT, "LexRAG-main", "data", "first_turn_by_law.json")

API_KEY = os.getenv("siliconflow_key", "") or os.getenv("SILICON", "")
EMBED_MODEL = "Qwen/Qwen3-Embedding-8B"
BASE_URL = "https://api.siliconflow.cn/v1"

# ── Normalization ───────────────────────────────────────
CN_MAP = {'零':0,'一':1,'二':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9,'十':10,'百':100,'千':1000}

def c2a(cn):
    if not cn: return 0
    if cn.isdigit(): return int(cn)
    r, t = 0, 0
    for ch in cn:
        if ch in CN_MAP:
            v = CN_MAP[ch]
            if v >= 10:
                if t == 0: t = 1
                r += t * v; t = 0
            else: t = v
    return r + t

def norm_art(ref):
    if not ref: return ""
    ref = ref.strip()
    m = re.match(r'《(.+?)》(.+)', ref)
    if not m: return ref
    law = m.group(1).replace("中华人民共和国", "")
    part = m.group(2)
    nm = re.search(r'第?([零一二三四五六七八九十百千\d]+)条', part)
    if not nm: return f"《{law}》{part}"
    return f"《{law}》第{c2a(nm.group(1))}条"

def compute_metrics(gt, pred, k=5):
    gt_set = set(norm_art(a) for a in gt)
    pred_set = set(norm_art(a) for a in pred[:k])
    if not gt_set: return {"P": 0.0, "R": 0.0, "F1": 0.0}
    tp = len(gt_set & pred_set)
    p = tp / len(pred_set) if pred_set else 0.0
    r = tp / len(gt_set)
    f1 = 2*p*r/(p+r) if p+r > 0 else 0.0
    return {"P": p, "R": r, "F1": f1}

# ── Data Loading ─────────────────────────────────────────

def load_law_library():
    laws = []
    with open(LAW_LIBRARY_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                laws.append({"name": item["name"], "content": item["content"]})
    return laws

def load_dataset(path, name):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    items = []
    for i, entry in enumerate(data):
        if 'conversation' in entry:
            t0 = entry['conversation'][0]
            q = t0.get('user', '')
            gt = t0.get('article', [])
        elif 'user' in entry and 'article' in entry:
            q = entry['user']
            gt = entry['article']
        elif 'question' in entry:
            q = entry['question']
            gt = entry.get('articles', [])
        else:
            continue
        if q and gt:
            items.append({'id': str(i), 'question': q.strip(), 'gt_articles': gt if isinstance(gt, list) else [gt]})
    print(f"  {name}: {len(items)} valid items")
    return items

# ── BM25 Tokenizer ───────────────────────────────────────

def tokenize(text):
    """Simple Chinese tokenizer: character bigrams + unigrams"""
    text = re.sub(r'[^一-鿿\w]', ' ', text)
    chars = text.replace(' ', '')
    tokens = list(chars)
    # Add bigrams
    for i in range(len(chars)-1):
        tokens.append(chars[i:i+2])
    return tokens

# ── BM25 Implementation ──────────────────────────────────

class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.N = len(corpus)
        self.tokenized = [tokenize(doc) for doc in corpus]
        self.doc_len = [len(t) for t in self.tokenized]
        self.avgdl = sum(self.doc_len) / self.N if self.N > 0 else 1.0
        # DF (document frequency) for each term
        self.df = Counter()
        for tokens in self.tokenized:
            for t in set(tokens):
                self.df[t] += 1
        # Precompute TF per doc
        self.tf = [Counter(tokens) for tokens in self.tokenized]

    def score(self, query, doc_idx):
        """Score a single document against query"""
        q_tokens = tokenize(query)
        score = 0.0
        doc_tf = self.tf[doc_idx]
        doc_len = self.doc_len[doc_idx]
        for t in q_tokens:
            if t not in self.df:
                continue
            tf = doc_tf.get(t, 0)
            if tf == 0:
                continue
            df = self.df[t]
            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
            numer = tf * (self.k1 + 1)
            denom = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += idf * numer / denom
        return score

    def search(self, query, top_k=5):
        scores = [self.score(query, i) for i in range(self.N)]
        order = np.argsort(scores)[::-1][:top_k]
        return [(i, scores[i]) for i in order]

# ── Embedding API ────────────────────────────────────────

def embed_batch(client, texts, batch_size=50):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        for attempt in range(5):
            try:
                resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
                all_embs.extend([d.embedding for d in resp.data])
                break
            except Exception as e:
                if attempt == 4: raise
                time.sleep(2 ** attempt)
    return np.array(all_embs, dtype=np.float32)

# ── Run ──────────────────────────────────────────────────

def run_experiment(dataset_path, dataset_name, output_prefix):
    print(f"\n{'='*70}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*70}")

    laws = load_law_library()
    dataset = load_dataset(dataset_path, dataset_name)
    print(f"  Laws: {len(laws)}, Questions: {len(dataset)}")

    # Init client
    api_key = API_KEY
    if not api_key:
        print("  ERROR: No API key!")
        return None
    client = OpenAI(api_key=api_key, base_url=BASE_URL)

    # Build BM25
    print("  Building BM25 index...")
    law_texts = [l["content"] for l in laws]
    bm25 = BM25(law_texts)

    # Dense embeddings
    print("  Embedding laws with Qwen3-Emb-8B...")
    law_embs = embed_batch(client, law_texts, batch_size=50)

    print("  Embedding questions with Qwen3-Emb-8B...")
    q_texts = [item["question"] for item in dataset]
    q_embs = embed_batch(client, q_texts, batch_size=20)

    # Normalize dense
    law_norm = law_embs / (np.linalg.norm(law_embs, axis=1, keepdims=True) + 1e-8)
    q_norm = q_embs / (np.linalg.norm(q_embs, axis=1, keepdims=True) + 1e-8)

    # Compute all BM25 scores
    print("  Computing BM25 scores...")
    bm25_scores = np.zeros((len(dataset), len(laws)), dtype=np.float32)
    for i, item in enumerate(dataset):
        for idx, score in bm25.search(item["question"], top_k=len(laws)):
            bm25_scores[i, idx] = score
        if (i+1) % 200 == 0:
            print(f"    BM25: {i+1}/{len(dataset)}")

    # Normalize BM25 scores to [0,1] per query
    bm25_max = bm25_scores.max(axis=1, keepdims=True)
    bm25_max[bm25_max == 0] = 1.0
    bm25_norm = bm25_scores / bm25_max

    # Dense scores
    dense_scores = q_norm @ law_norm.T

    # ── BM25 Only ──
    print("  Running BM25 retrieval...")
    bm25_results = []
    for i, item in enumerate(dataset):
        top5 = np.argsort(bm25_norm[i])[::-1][:5]
        preds = [norm_art(laws[j]["name"]) for j in top5]
        m = compute_metrics(item["gt_articles"], preds)
        bm25_results.append({**m, "pred": preds})

    # ── Hybrid (α=0.5) ──
    print("  Running Hybrid (BM25 + Dense)...")
    for alpha in [0.3, 0.5, 0.7]:
        hybrid_scores = alpha * dense_scores + (1 - alpha) * bm25_norm
        results = []
        for i, item in enumerate(dataset):
            top5 = np.argsort(hybrid_scores[i])[::-1][:5]
            preds = [norm_art(laws[j]["name"]) for j in top5]
            m = compute_metrics(item["gt_articles"], preds)
            results.append({**m, "pred": preds})
        # Keep best alpha
        n = len(results)
        avg_f1 = sum(r["F1"] for r in results) / n
        print(f"    α={alpha}: F1={avg_f1:.4f}")
        if alpha == 0.5:
            hybrid_results = results
            best_alpha = alpha

    # Summarize
    def summary(results, name):
        n = len(results)
        return {
            "method": name,
            "samples": n,
            "P@5": round(sum(r["P"] for r in results)/n, 4),
            "R@5": round(sum(r["R"] for r in results)/n, 4),
            "F1@5": round(sum(r["F1"] for r in results)/n, 4),
        }

    summaries = [
        summary(bm25_results, "BM25"),
        summary(hybrid_results, "Hybrid (BM25 + Dense)"),
    ]

    print(f"\n  {'Method':<30} {'P@5':>8} {'R@5':>8} {'F1@5':>8}")
    print(f"  {'-'*55}")
    for s in summaries:
        print(f"  {s['method']:<30} {s['P@5']:>8.4f} {s['R@5']:>8.4f} {s['F1@5']:>8.4f}")

    # Save
    out = {
        "dataset": dataset_name,
        "samples": len(dataset),
        "results": {s["method"]: {"P@5": s["P@5"], "R@5": s["R@5"], "F1@5": s["F1@5"]} for s in summaries},
        "best_alpha": best_alpha,
    }
    path = os.path.join(RESULTS_DIR, f"{output_prefix}_bm25_hybrid.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {path}")

    return summaries

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all", choices=["lawbench", "lexrag", "all"])
    args = parser.parse_args()

    all_summaries = {}

    if args.dataset in ("lawbench", "all"):
        s = run_experiment(LAWBENCH_PATH, "Lawbench (439)", "lawbench")
        if s: all_summaries["lawbench"] = s

    if args.dataset in ("lexrag", "all"):
        s = run_experiment(LEXRAG_PATH, "LexRAG (1013)", "lexrag")
        if s: all_summaries["lexrag"] = s

    print(f"\n{'='*70}")
    print("BM25 + HYBRID - RESULTS")
    print(f"{'='*70}")
    for ds, summaries in all_summaries.items():
        print(f"\n--- {ds} ---")
        for s in summaries:
            print(f"  {s['method']:<30} P@5={s['P@5']:.4f} R@5={s['R@5']:.4f} F1@5={s['F1@5']:.4f}")


if __name__ == "__main__":
    main()
