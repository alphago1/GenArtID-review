"""Portable GenArtID core: corpus-only preparation, ID binding and scoring.

All commands are offline except the explicitly opted-in generate-api command.
No account-key creation, model downloads, SSH, or service control is included.
"""
import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import urllib.request

from ga_corpus import load_corpus
from ga_build_sft import build_ratio_dataset,RATIOS_F_R
from ga_identifiers import canon_fixed
from ga_parsing import parse_fixed
from ga_scoring import per_question_scores
from ga_split import load_questions,build_split,split_manifest_hash
from ga_memory_recall_probe import recitation_prompts

ROOT=Path(__file__).resolve().parent


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def rows(path):
    return [json.loads(l) for l in Path(path).read_text(encoding='utf-8').splitlines() if l.strip()]


def prepare(args):
    articles=load_corpus(Path(args.corpus),args.domain)
    out=Path(args.out)
    manifest=read(out/'build_manifest.json') if (out/'build_manifest.json').exists() else {'schema':'review-sft/1','inputs':{},'outputs':[]}
    manifest['inputs'][args.domain]=sha(args.corpus)
    manifest['outputs']=[e for e in manifest['outputs'] if e['domain']!=args.domain]
    for ratio in RATIOS_F_R:
        dataset=build_ratio_dataset(args.domain,articles,ratio)
        name=dataset['stats']['ratio_label']
        path=out/args.domain/f'ratio_{name}.json'
        save(path,dataset)
        manifest['outputs'].append({'kind':'ratio','domain':args.domain,'path':str(path),'sha256':sha(path)})
    save(out/'build_manifest.json',manifest)
    return {'domain':args.domain,'articles':len(articles),'slots_per_ratio':len(articles)*4,'corpus_sha256':sha(args.corpus)}


def split(args):
    qs=load_questions(Path(args.questions),args.domain,canon_fixed)
    manifest=build_split(qs,args.domain,seed=args.seed)
    manifest['split_manifest_hash']=split_manifest_hash(manifest)
    save(args.out,manifest)
    return manifest['counts']


def evaluate(args):
    observed=rows(args.predictions)
    if len({str(r['item_id']) for r in observed})!=len(observed):raise ValueError('Duplicate item_id')
    gold_rows=rows(args.gold)
    gold={str(r['item_id']):{canon_fixed(x) for x in r['gold']} for r in gold_rows}
    if len(gold)!=len(gold_rows) or any('' in s or not s for s in gold.values()):raise ValueError('Duplicate/invalid/empty gold')
    if {str(r['item_id']) for r in observed}!=set(gold):raise ValueError('Refuse incomplete or extra predictions')
    total=Fraction();precision=Fraction();details=[]
    for r in observed:
        iid=str(r['item_id']);parsed=parse_fixed(r['raw_response']);pred=parsed.predictions
        hit=len(set(pred[:5]) & gold[iid]);total+=Fraction(hit,len(gold[iid]));precision+=Fraction(hit,5)
        details.append({'item_id':iid,'pred':pred,'parse_status':parsed.status,'scores':per_question_scores(pred,list(gold[iid]))})
    result={'n':len(gold),'Recall@5_exact':str(total/len(gold)),'Recall@5':float(total/len(gold)),
            'P@5_fixed':float(precision/len(gold)),'details':details,'parser':'fixed_v2','gold_view':'full_gold_all'}
    save(args.out,result);return {k:v for k,v in result.items() if k!='details'}


def bind(args):
    if args.domain=='all':
        source=Path(args.corpus)
        articles=rows(source) if source.suffix=='.jsonl' else read(source)
        pool={}
        for a in articles:
            iid=canon_fixed(a.get('name',a.get('id','')))
            text=a.get('content')
            if not iid or iid in pool or not isinstance(text,str) or not text.strip():
                raise ValueError('Invalid/duplicate article in multi-law binding corpus')
            pool[iid]=text
    else:
        pool={a.canon_id:a.repaired_content for a in load_corpus(Path(args.corpus),args.domain)}
    parsed=parse_fixed(args.answer)
    return {'parse_status':parsed.status,'bound':[{'id':iid,'text':pool[iid]} for iid in parsed.predictions if iid in pool],
            'unbound':[iid for iid in parsed.predictions if iid not in pool]}


def generate(args):
    if args.task=='recitation':
        arts=load_corpus(Path(args.input),args.domain)
        jobs=[{'item_id':p['item_id'],'messages':[{'role':'system','content':p['system']},
                {'role':'user','content':p['user']}]} for p in recitation_prompts(arts,args.domain)]
    else:
        profile=read(ROOT/'base_ft_eval_profile.json')
        jobs=[{'item_id':str(r['id']),'messages':[{'role':'system','content':profile['prompt']['system_prompt_verbatim_notebook_top5']},
                {'role':'user','content':profile['prompt']['user_template'].format(question=r['question'])}]} for r in read(args.input)]
    if not args.execute:return {'planned_jobs':len(jobs),'paid_requests_executed':0,'task':args.task}
    key=os.environ.get('GENARTID_API_KEY','')
    if not key:raise ValueError('Set GENARTID_API_KEY locally; never store it in repository files')
    target=Path(args.out)
    done={r['item_id'] for r in rows(target)} if target.exists() else set()
    target.parent.mkdir(parents=True,exist_ok=True)
    for j in jobs:
        if j['item_id'] in done:continue
        p={'model':args.model,'messages':j['messages'],'temperature':0.0,'max_tokens':2048 if args.task=='recitation' else 256,
           'stream':False,'n':1,'frequency_penalty':0.0}
        if args.provider:p['provider']={'only':[args.provider],'allow_fallbacks':False,'data_collection':'deny'}
        req=urllib.request.Request(args.api_base.rstrip('/')+'/chat/completions',data=json.dumps(p,ensure_ascii=False).encode('utf-8'),
                                   headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
        # No quality-based retries. Errors stop; successful rows persist once.
        with urllib.request.urlopen(req,timeout=120) as response:body=json.load(response)
        row={'item_id':j['item_id'],'raw_response':body['choices'][0]['message'].get('content') or '',
             'payload_sha256':hashlib.sha256(json.dumps(p,sort_keys=True,ensure_ascii=False).encode('utf-8')).hexdigest(),'response':body}
        with target.open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
    return {'completed':len(jobs),'task':args.task}


def memory(args):
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    from ga_local_memory_exec import Text2vecEncoder,score_memory_run
    identity=read(ROOT/'scorer_identity.json')
    for name,entry in identity['encoder_files'].items():
        if sha(Path(args.encoder)/name)!=entry['sha256']:raise ValueError('Pinned encoder file mismatch: '+name)
    arts=load_corpus(Path(args.corpus),args.domain)
    rs=rows(args.predictions)
    outputs={r['item_id']:r['raw_response'] for r in rs}
    if len(outputs)!=len(rs) or set(outputs)!={a.canon_id for a in arts}:raise ValueError('Memory coverage/duplicate mismatch')
    encoder=Text2vecEncoder(args.encoder,identity,num_threads=4)
    result=score_memory_run('api',args.domain,arts,outputs,encoder)
    save(args.out,result)
    return {k:result[k] for k in ['n','eff','dup','err','memory_rate']}


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    x=sub.add_parser('prepare-sft');x.add_argument('--domain',choices=['civil','criminal_law'],required=True);x.add_argument('--corpus',required=True);x.add_argument('--out',default=str(ROOT/'artifacts/sft_corpus_only'))
    x=sub.add_parser('split');x.add_argument('--domain',choices=['civil','criminal_law'],required=True);x.add_argument('--questions',required=True);x.add_argument('--seed',type=int,default=20260910);x.add_argument('--out',required=True)
    x=sub.add_parser('evaluate');x.add_argument('--predictions',required=True);x.add_argument('--gold',required=True);x.add_argument('--out',required=True)
    x=sub.add_parser('bind');x.add_argument('--domain',choices=['civil','criminal_law','all'],default='all');x.add_argument('--corpus',required=True);x.add_argument('--answer',required=True)
    x=sub.add_parser('generate-api');x.add_argument('--task',choices=['retrieval','recitation'],required=True);x.add_argument('--domain',choices=['civil','criminal_law'],required=True);x.add_argument('--input',required=True);x.add_argument('--model',required=True);x.add_argument('--provider');x.add_argument('--api-base',default='https://openrouter.ai/api/v1');x.add_argument('--out',required=True);x.add_argument('--execute',action='store_true')
    x=sub.add_parser('memory');x.add_argument('--domain',choices=['civil','criminal_law'],required=True);x.add_argument('--corpus',required=True);x.add_argument('--predictions',required=True);x.add_argument('--encoder',required=True);x.add_argument('--out',required=True)
    a=p.parse_args();fn={'prepare-sft':prepare,'split':split,'evaluate':evaluate,'bind':bind,'generate-api':generate,'memory':memory}[a.command]
    print(json.dumps(fn(a),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
