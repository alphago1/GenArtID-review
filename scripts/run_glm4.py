"""Table 3: THUDM/GLM-4-9B-0414 via SiliconFlow API."""
import json, re, os, time, threading, sys
from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEY = os.getenv('siliconflow_key', '') or os.getenv('SILICON', '')
if not API_KEY: print('ERROR: Set siliconflow_key'); sys.exit(1)
from openai import OpenAI
client = OpenAI(api_key=API_KEY, base_url='https://api.siliconflow.cn/v1')

MODEL = 'THUDM/GLM-4-9B-0414'; MAX_WORKERS = 20
OUTPUT_DIR = r'D:\download\LexRAG-main-new\experiments\table3_glm4_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f'Model: {MODEL}')

CN_M = {'零':0,'〇':0,'○':0,'一':1,'二':2,'两':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9}
CN_U = {'十':10,'百':100,'千':1000,'万':10000}
def c2i(t):
    if not t: return None
    t=str(t).strip().translate(str.maketrans('０１２３４５６７８９','0123456789'))
    if t.isdigit(): return int(t)
    tot=sec=num=0
    for c in t:
        if c in CN_M: num=CN_M[c]
        elif c in CN_U:
            u=CN_U[c]
            if u==10000: sec=(sec+(num or 0))*u; tot+=sec; sec=num=0
            else:
                if num==0: num=1; sec+=num*u; num=0
        else: return None
    return tot+sec+num
def canon(a):
    if not isinstance(a,str): return ''
    x=a.strip(); x=re.sub(r'\s+','',x); x=x.replace('（','(').replace('）',')')
    x=x.translate(str.maketrans('０１２３４５６７８９','0123456789'))
    m=re.match(r'^(《.*?》)第([0-9一二三四五六七八九十百千万零〇○两]+)条$',x)
    if not m: return x
    l=re.sub(r'^《中华人民共和国','《',m.group(1)); n=c2i(m.group(2))
    return f'{l}第{n}条' if n is not None else f'{l}第{m.group(2)}条'
ART_RE = r'《.+?》第[0-9０-９一二三四五六七八九十百千万零〇○两]+条'
def extract_articles(text, topk=5):
    mentions=[]
    m=re.search(r'\{.*\}',text,re.S)
    if m:
        try:
            obj=json.loads(m.group())
            if isinstance(obj,dict) and isinstance(obj.get('articles'),list):
                for x in obj['articles']:
                    if isinstance(x,str): mentions.extend(re.findall(ART_RE,x))
        except: pass
    if not mentions: mentions=re.findall(ART_RE,text)[:topk]
    else:
        for m2 in re.findall(ART_RE,text):
            if len(mentions)>=topk: break
            mentions.append(m2)
    return [canon(x) for x in mentions[:topk] if canon(x)]

TOP5 = '你是一个专业的中国法律检索专家。\n\n你的任务不是作答，而是为后续数据库检索生成候选法条。\n请根据用户案情，输出最相关的5条法条编号，按相关性从高到低排序。\n\n硬性要求：\n1. 必须恰好输出5条；\n2. 每条都必须是《法律名称》第XXX条；\n3. 不要输出解释、分析、理由、序号或其他任何文字；\n4. 即使不确定，也必须给出5条最可能的候选法条；\n5. 优先输出最核心、最直接适用的法条；\n6. 只输出 JSON，不要输出其他内容。\n\n输出格式：\n{"articles": ["《法律名称》第XXX条", ...]}'
MEM_SYS = '你是一个专业的中国法律背诵助手。请严格根据用户要求输出法条内容，不要添加解释。'
lock = threading.Lock()

def api_call(msgs, max_tok=512):
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(model=MODEL, messages=msgs, temperature=0.0, max_tokens=max_tok)
            return resp.choices[0].message.content or ''
        except Exception as e:
            if attempt == 2: return ''
            time.sleep(4 ** attempt)
    return ''

CIVIL_A = r'D:\download\LexRAG-main-new\paper_data\table2_parametric_memory\data\civil_code_articles_cleaned.json'
CPL_A   = r'D:\download\LexRAG-main-new\experiments\criminal_procedure_lora\data\criminal_procedure_articles.json'
CIVIL_E = r'D:\download\LexRAG-main-new\experiments\criminal_procedure_lora\civil_code\data\eval_civil_code_questions.json'
CPL_E   = r'D:\download\LexRAG-main-new\experiments\criminal_procedure_lora\data\eval_cpl_questions.json'
RS2 = {'分期车第二年保险没在分期公司买，他们有权扣车么？','借钱给朋友，朋友不还怎么办？有录音和转账记录。',
       '我早上手机在教室丢了，去学校看监控被拒绝了。','我的父母想把他们的安置房送给孙子，没有房产证。',
       '去别人家鱼塘钓鱼被抓了，但一条鱼都没钓到。','大学生欠了很多贷款，没钱怎么办？',
       '外祖父母过世后房产子女平分，孙辈没有继承权吧。','父亲去世银行有存款怎么取？',
       '结婚和父母住，拆迁给二套房怎么分。','买了二手房过了户受法律保护吗。'}

for law_name, art_path, out_name in [('Civil', CIVIL_A, 'civil_memory_raw_glm4.json'), ('CPL', CPL_A, 'cpl_memory_raw_glm4.json')]:
    print(f'\n=== Memory: {law_name} ===')
    with open(art_path,'r',encoding='utf-8') as f: raw = json.load(f)
    arts = sorted([{'name':canon(a['name']),'content':a['content'].strip()} for a in raw], key=lambda x:x['name'])
    print(f'Articles: {len(arts)}')
    mem = [None]*len(arts)
    def recite(idx, art):
        r = api_call([{'role':'system','content':MEM_SYS},{'role':'user','content':f'请背诵{art["name"]}。'}], 512)
        with lock:
            if (idx+1)%200==0: print(f'  [{idx+1}/{len(arts)}]')
        return idx, {'name':art['name'],'raw':r,'gt_content':art['content']}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(recite,i,a):i for i,a in enumerate(arts)}
        for f in as_completed(futs): idx,r=f.result(); mem[idx]=r
    with open(f'{OUTPUT_DIR}/{out_name}','w',encoding='utf-8') as f: json.dump(mem,f,ensure_ascii=False,indent=2)
    print(f'Done: {len(mem)}')

with open(CIVIL_E,'r',encoding='utf-8') as f: civil_qs = [q for q in json.load(f) if q['question'] not in RS2]
print(f'\n=== Retrieval: Civil ({len(civil_qs)} qs) ===')
civil_ret=[None]*len(civil_qs)
def ret_civil(idx,item):
    q=item['question']
    raw=api_call([{'role':'system','content':TOP5},{'role':'user','content':f'/no_think\n案情：\n{q}\n\n请严格输出 JSON。'}],256)
    preds=extract_articles(raw); gs=set(canon(a) for a in item['articles'] if canon(a)); ps=set(preds)
    tp=len(ps&gs); p=tp/max(len(ps),1); r=tp/max(len(gs),1); f1=2*p*r/(p+r) if p+r>0 else 0.0
    with lock:
        if (idx+1)%100==0: print(f'  [{idx+1}/{len(civil_qs)}]')
    return idx,{'id':item.get('id',idx),'q':q,'gold':list(gs),'pred':list(ps),'P':round(p,6),'R':round(r,6),'F1':round(f1,6),'raw':raw}
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
    futs={pool.submit(ret_civil,i,item):i for i,item in enumerate(civil_qs)}
    for f in as_completed(futs): idx,r=f.result(); civil_ret[idx]=r
N=len(civil_ret); cr_P=round(sum(r['P'] for r in civil_ret)/N,4); cr_R=round(sum(r['R'] for r in civil_ret)/N,4); cr_F=round(sum(r['F1'] for r in civil_ret)/N,4)
print(f'Civil: P@5={cr_P} R@5={cr_R} F1@5={cr_F}')

with open(CPL_E,'r',encoding='utf-8') as f: cpl_qs=json.load(f)
print(f'\n=== Retrieval: CPL ({len(cpl_qs)} qs) ===')
cpl_ret=[None]*len(cpl_qs)
def ret_cpl(idx,item):
    q=item['question']
    raw=api_call([{'role':'system','content':TOP5},{'role':'user','content':f'/no_think\n案情：\n{q}\n\n请严格输出 JSON。'}],256)
    preds=extract_articles(raw); gs=set(canon(a) for a in item['articles'] if canon(a)); ps=set(preds)
    tp=len(ps&gs); p=tp/max(len(ps),1); r=tp/max(len(gs),1); f1=2*p*r/(p+r) if p+r>0 else 0.0
    with lock:
        if (idx+1)%20==0: print(f'  [{idx+1}/{len(cpl_qs)}]')
    return idx,{'id':item.get('id',idx),'q':q,'gold':list(gs),'pred':list(ps),'P':round(p,6),'R':round(r,6),'F1':round(f1,6),'raw':raw}
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
    futs={pool.submit(ret_cpl,i,item):i for i,item in enumerate(cpl_qs)}
    for f in as_completed(futs): idx,r=f.result(); cpl_ret[idx]=r
N=len(cpl_ret); cp_P=round(sum(r['P'] for r in cpl_ret)/N,4); cp_R=round(sum(r['R'] for r in cpl_ret)/N,4); cp_F=round(sum(r['F1'] for r in cpl_ret)/N,4)
print(f'CPL: P@5={cp_P} R@5={cp_R} F1@5={cp_F}')

print('\n=== Memory Eval (text2vec) ===')
import torch; from sentence_transformers import SentenceTransformer, util
sim = SentenceTransformer('shibing624/text2vec-base-chinese'); PASS=0.85

def eval_mem(mem_raw,label):
    gt_texts=[r['gt_content'] for r in mem_raw]
    print(f'  Encoding {len(gt_texts)} GT...')
    gt_embs=sim.encode(gt_texts,convert_to_tensor=True,show_progress_bar=True)
    claimed=set(); exact=misplaced=rep=fail=0; results=[]
    for idx in range(len(mem_raw)):
        r=mem_raw[idx]; raw=r['raw']
        if not raw: fail+=1; results.append({'idx':idx,'name':r['name'],'raw':'','status_type':'fail','current_score':0,'best_score':0,'best_idx':-1}); continue
        gen_emb=sim.encode(raw,convert_to_tensor=True)
        cur=util.cos_sim(gen_emb,gt_embs[idx]).item()
        scores=util.cos_sim(gen_emb,gt_embs)[0]; bi=torch.argmax(scores).item(); bs=scores[bi].item()
        if cur>=PASS: claimed.add(idx); result={'idx':idx,'name':r['name'],'raw':raw[:200],'current_score':cur,'best_score':bs,'best_idx':bi,'status_type':'exact'}; exact+=1
        elif bs>=PASS:
            if bi in claimed: result={'idx':idx,'name':r['name'],'raw':raw[:200],'current_score':cur,'best_score':bs,'best_idx':bi,'status_type':'repetition'}; rep+=1
            else: result={'idx':idx,'name':r['name'],'raw':raw[:200],'current_score':cur,'best_score':bs,'best_idx':bi,'status_type':'misplaced'}; misplaced+=1; claimed.add(bi)
        else: result={'idx':idx,'name':r['name'],'raw':raw[:200],'current_score':cur,'best_score':bs,'best_idx':bi,'status_type':'fail'}; fail+=1
        results.append(result)
    t=len(mem_raw); v=exact+misplaced
    return {'effective_memory':round(v/t,4),'invalid_duplicate':round(rep/t,4),'error_hallucination':round(fail/t,4),'total':t},results

with open(f'{OUTPUT_DIR}/civil_memory_raw_glm4.json','r',encoding='utf-8') as f: civil_mem=json.load(f)
with open(f'{OUTPUT_DIR}/cpl_memory_raw_glm4.json','r',encoding='utf-8') as f: cpl_mem=json.load(f)
cm_s, cm_r = eval_mem(civil_mem,'Civil')
print(f'Civil: Eff={cm_s["effective_memory"]} Dup={cm_s["invalid_duplicate"]} Err={cm_s["error_hallucination"]}')
cpm_s, cpm_r = eval_mem(cpl_mem,'CPL')
print(f'CPL:   Eff={cpm_s["effective_memory"]} Dup={cpm_s["invalid_duplicate"]} Err={cpm_s["error_hallucination"]}')

with open(f'{OUTPUT_DIR}/civil_memory_eval_glm4.json','w',encoding='utf-8') as f: json.dump({'summary':cm_s,'per_article':cm_r},f,ensure_ascii=False)
with open(f'{OUTPUT_DIR}/cpl_memory_eval_glm4.json','w',encoding='utf-8') as f: json.dump({'summary':cpm_s,'per_article':cpm_r},f,ensure_ascii=False)
with open(f'{OUTPUT_DIR}/civil_retrieval_glm4.json','w',encoding='utf-8') as f: json.dump({'summary':{'P@5':cr_P,'R@5':cr_R,'F1@5':cr_F,'samples':len(civil_ret)},'per_sample':civil_ret},f,ensure_ascii=False)
with open(f'{OUTPUT_DIR}/cpl_retrieval_glm4.json','w',encoding='utf-8') as f: json.dump({'summary':{'P@5':cp_P,'R@5':cp_R,'F1@5':cp_F,'samples':len(cpl_ret)},'per_sample':cpl_ret},f,ensure_ascii=False)

print('\n'+'='*60)
print('TABLE 3: GLM-4-9B-0414')
print('='*60)
print(f'| Civil Code | GLM-4-9B-0414 | {cm_s["effective_memory"]:.4f} | {cm_s["invalid_duplicate"]:.4f} | {cm_s["error_hallucination"]:.4f} | {cr_P:.4f} | {cr_R:.4f} | {cr_F:.4f} |')
print(f'| CPL | GLM-4-9B-0414 | {cpm_s["effective_memory"]:.4f} | {cpm_s["invalid_duplicate"]:.4f} | {cpm_s["error_hallucination"]:.4f} | {cp_P:.4f} | {cp_R:.4f} | {cp_F:.4f} |')

combined={'model':MODEL,'civil_code':{'memory':cm_s,'retrieval':{'P@5':cr_P,'R@5':cr_R,'F1@5':cr_F}},'cpl':{'memory':cpm_s,'retrieval':{'P@5':cp_P,'R@5':cp_R,'F1@5':cp_F}}}
with open(f'{OUTPUT_DIR}/table3_glm4.json','w',encoding='utf-8') as f: json.dump(combined,f,ensure_ascii=False,indent=2)
print(f'\nSaved to {OUTPUT_DIR}/')
print('DONE.')
