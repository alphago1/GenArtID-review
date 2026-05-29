"""
重现 CPL Qwen3-8B Base 检索结果 (Table 7 / Table 2 CPL 行)
输出: P@5=0.0708, R@5=0.3125, F1@5=0.1141

运行方式: 在 AutoDL 上 python cpl_base_retrieval.py
需要: Qwen3-8B 4-bit 模型 + CPL eval 数据
"""
import os, json, re, gc, torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ===== 路径配置 (AutoDL) =====
MODEL_NAME = '/root/autodl-tmp/employee_code/base_model'
EVAL_JSON  = '/root/autodl-tmp/cpl_ft/data/eval_cpl_questions.json'
OUTPUT_JSON = '/root/autodl-tmp/cpl_ft/base_retrieval_result.json'

# ===== ID-Only Prompt (与论文 Table 1 完全一致) =====
TOP5_PROMPT = """你是一个专业的中国法律检索专家。

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

# ===== 中文数字转换 + 法条规范化 =====
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

def canon(a):
    if not isinstance(a, str): return ''
    x = a.strip(); x = re.sub(r'\s+', '', x); x = x.replace('（', '(').replace('）', ')')
    x = x.translate(str.maketrans('０１２３４５６７８９', '0123456789'))
    m = re.match(r'^(《.*?》)第([0-9一二三四五六七八九十百千万零〇○两]+)条$', x)
    if not m: return x
    l = re.sub(r'^《中华人民共和国', '《', m.group(1))
    n = c2i(m.group(2))
    return f'{l}第{n}条' if n is not None else f'{l}第{m.group(2)}条'

ART_RE = r'《.+?》第[0-9０-９一二三四五六七八九十百千万零〇○两]+条'

def extract_articles(text, topk=5):
    mentions = []
    m = re.search(r'\{.*\}', text, re.S)
    if m:
        try:
            obj = json.loads(m.group())
            if isinstance(obj, dict) and isinstance(obj.get('articles'), list):
                for x in obj['articles']:
                    if isinstance(x, str): mentions.extend(re.findall(ART_RE, x))
        except: pass
    if not mentions: mentions = re.findall(ART_RE, text)[:topk]
    else:
        for m2 in re.findall(ART_RE, text):
            if len(mentions) >= topk: break
            mentions.append(m2)
    return [canon(x) for x in mentions[:topk] if canon(x)]

# ===== 加载模型 (4-bit QLoRA) =====
bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, quantization_config=bnb, device_map='auto', trust_remote_code=True)
model.eval()

# ===== 生成函数 =====
@torch.no_grad()
def generate_top5(question):
    msgs = [
        {'role': 'system', 'content': TOP5_PROMPT},
        {'role': 'user', 'content': f'/no_think\n案情：\n{question}\n\n请严格输出 JSON。'}
    ]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text, return_tensors='pt').to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, repetition_penalty=1.05,
                              eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    raw = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
    return {'raw': raw, 'pred': extract_articles(raw)}

# ===== 评估 =====
with open(EVAL_JSON, 'r', encoding='utf-8') as f:
    eval_data = json.load(f)
print(f'Eval questions: {len(eval_data)}')

results = []
for item in tqdm(eval_data, desc='Base model'):
    gen = generate_top5(item['question'])
    pred_set = set(gen['pred'])
    gold_set = set(canon(a) for a in item['articles'] if canon(a))
    tp = len(pred_set & gold_set)
    p = tp / max(len(pred_set), 1)
    r = tp / max(len(gold_set), 1)
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    results.append({
        'id': item['id'], 'question': item['question'][:100],
        'gold': list(gold_set), 'pred': gen['pred'],
        'P': round(p, 6), 'R': round(r, 6), 'F1': round(f1, 6),
        'raw': gen['raw']
    })

mp = sum(r['P'] for r in results) / len(results)
mr = sum(r['R'] for r in results) / len(results)
mf = sum(r['F1'] for r in results) / len(results)

summary = {
    'model': 'Qwen3-8B Base (4-bit)', 'samples': len(results),
    'P@5': round(mp, 4), 'R@5': round(mr, 4), 'F1@5': round(mf, 4)
}

print(f'\nP@5={mp:.4f} R@5={mr:.4f} F1@5={mf:.4f}')
print(f'Paper: P@5=0.0708 R@5=0.3125 F1@5=0.1141')

with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump({'summary': summary, 'per_sample': results}, f, ensure_ascii=False, indent=2)
print(f'Saved: {OUTPUT_JSON}')
