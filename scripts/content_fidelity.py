"""Recompute Table 2 ID+Content accuracy using cosine similarity (text2vec, threshold 0.85)"""
import json, os, re

# Load law library
LAW_PATH = r'D:\download\LexRAG-main-new\LexRAG-main\data\law_library.jsonl'
law_by_name = {}  # normalized name -> content
with open(LAW_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip(): continue
        item = json.loads(line)
        name = item['name']
        # Normalize: strip 中华人民共和国, convert Chinese nums
        norm = re.sub(r'^《中华人民共和国', '《', name)
        norm = re.sub(r'\s+', '', norm)
        law_by_name[norm] = item['content']

print('Law library:', len(law_by_name), 'articles')

# Load text2vec
from sentence_transformers import SentenceTransformer, util
print('Loading text2vec...')
model = SentenceTransformer('shibing624/text2vec-base-chinese')
print('Model ready.')

DIR = r'D:\download\LexRAG-main-new\LexRAG-main\experiments\table2_retrieval\results'

def normalize_id(article_id):
    """Normalize pred_id to match law library name format"""
    a = article_id.strip()
    a = re.sub(r'\s+', '', a)
    a = a.replace('（', '(').replace('）', ')')
    # Already in short form like 《民法典》第401条
    # law_by_name uses full form like 《中华人民共和国民法典》第四百零一条
    # -> try both forms
    return a

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

def find_law_content(pred_id):
    """Look up article content from law library, trying multiple name formats"""
    # Try exact match first
    if pred_id in law_by_name:
        return law_by_name[pred_id]
    # Try with 中华人民共和国 prefix
    full = pred_id.replace('《', '《中华人民共和国', 1)
    if full in law_by_name:
        return law_by_name[full]
    # Try converting Chinese numerals to Arabic in the pred_id
    import re as re2
    m = re2.match(r'^(《.+?》)第(.+)条$', pred_id)
    if m:
        law_name = m.group(1)
        num_str = m.group(2)
        arabic = c2i(num_str)
        if arabic is not None:
            # Search through law_by_name for matching law and article number
            for ln, content in law_by_name.items():
                lm = re2.match(r'^(《.+?》)第(.+)条$', ln)
                if lm:
                    ln_num = c2i(lm.group(2))
                    if ln_num == arabic and lm.group(1).replace('中华人民共和国', '') == law_name.replace('中华人民共和国', ''):
                        return content
    return None

for fname in ['id_content_deepseek_v4flash.json', 'id_content_qwen3_8b.json']:
    path = os.path.join(DIR, fname)
    with open(path, 'r', encoding='utf-8') as f:
        ds = json.load(f)

    results = ds['full_results']
    total = len(results)

    total_preds = 0
    correct = 0; partial = 0; wrong = 0; missing = 0
    all_sims = []
    samples_with_correct = 0
    samples_with_any = 0
    lookup_failures = 0

    for r in results:
        pred_ids = r.get('pred_ids', [])
        pred_contents = r.get('pred_contents', [])
        has_correct = False
        has_any = False

        for i, pid in enumerate(pred_ids):
            gt_content = find_law_content(pid)
            pred_content = pred_contents[i] if i < len(pred_contents) else None

            if gt_content is None or pred_content is None:
                missing += 1
                continue

            total_preds += 1
            sim = util.cos_sim(model.encode(pred_content, convert_to_tensor=True),
                               model.encode(gt_content, convert_to_tensor=True)).item()
            all_sims.append(sim)
            has_any = True

            if sim >= 0.85:
                correct += 1
                has_correct = True
            elif sim >= 0.6:
                partial += 1
            else:
                wrong += 1

        if not has_any:
            lookup_failures += 1
        if has_correct:
            samples_with_correct += 1
        if has_any:
            samples_with_any += 1

    print()
    print('=' * 60)
    print(fname)
    print('=' * 60)
    print('Samples:', total)
    print('Lookup failures:', lookup_failures)
    print()
    print('--- Per-prediction (cosine sim, same as compare_laws.py) ---')
    total_with_content = total_preds
    print('Total with content:', total_with_content)
    print('  Correct (>=0.85): ', correct, '({:.1%})'.format(correct/total_with_content if total_with_content else 0))
    print('  Partial (0.6-0.85):', partial, '({:.1%})'.format(partial/total_with_content if total_with_content else 0))
    print('  Wrong (<0.6):     ', wrong, '({:.1%})'.format(wrong/total_with_content if total_with_content else 0))
    print('  Missing:          ', missing)
    if all_sims:
        print('  Avg sim:          ', round(sum(all_sims)/len(all_sims), 4))
    print()
    print('--- Per-sample ---')
    print('>=1 correct content: ', samples_with_correct, '({:.1%})'.format(samples_with_correct/total))
    print('>=1 any content:     ', samples_with_any, '({:.1%})'.format(samples_with_any/total))
