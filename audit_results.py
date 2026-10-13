"""Audit the released numerical evidence without weights, GPU or network."""
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent

def load(path):return json.loads((ROOT/path).read_text(encoding='utf-8'))

def main():
    summary=load('results/memory_recall_runs.json')
    points=load('fig3_points.json')['points']
    expected_runs=set()
    for point in points:
        keys=[f'glm0414_seed{seed}' for seed in [20260910,20260911,20260912]] if point['model_run_key']=='glm0414_ssl' else [point['model_run_key']]
        expected_runs.update((key,point['domain']) for key in keys)
    assert len(summary['runs'])==14
    assert {(r['run_key'],r['domain']) for r in summary['runs']}==expected_runs
    expected_files={f'{key}_{domain}.json' for key,domain in expected_runs}
    for folder in ['per_question','per_article']:
        assert {p.name for p in (ROOT/'results'/folder).glob('*.json')}==expected_files
    questions=articles=0
    for run in summary['runs']:
        name=run['run_key']+'_'+run['domain']
        r=load('results/per_question/'+name+'.json');m=load('results/per_article/'+name+'.json')
        assert len(r)==run['n_questions'] and len(m)==run['n_articles']
        assert len({x['item_id'] for x in r})==len(r) and len({x['item_id'] for x in m})==len(m)
        recalls=[]
        for row in r:
            assert len(row['predicted_ids'])<=5
            correct=len(set(row['predicted_ids'])&set(row['gold_ids']));den=len(set(row['gold_ids']))
            value=Fraction(correct,den)
            assert correct==row['correct_count'] and den==row['gold_count']
            assert value==Fraction(row['R@5_exact']) and abs(correct/5-row['P@5_fixed'])<1e-12
            recalls.append(value)
        mean=sum(recalls,Fraction())/len(recalls)
        assert mean==Fraction(run['R@5_exact']) and abs(float(mean)-run['R@5'])<1e-12
        classes=Counter(x['class'] for x in m)
        for row in m:
            expected='Err.' if row['empty_output'] else 'Eff.' if row['s_target']>=.85 else 'Dup.' if row['s_other']>=.85 else 'Err.'
            assert row['class']==expected
        assert all(classes[k]==v for k,v in run['counts'].items())
        rate=Fraction(classes['Eff.'],len(m))
        assert rate==Fraction(run['M_exact']) and abs(float(rate)-run['M'])<1e-12
        questions+=len(r);articles+=len(m)
    checkpoints=load('checkpoints/manifest.json')['adapters'];assert len(checkpoints)==6
    for adapter in checkpoints:
        selected=load('selection/'+adapter['domain']+'_seed'+str(adapter['seed'])+'.json')
        best=min(selected['candidates'],key=lambda c:(-c['dev_recall_at_5'],c['global_step']))
        assert best['global_step']==adapter['selected_step']==selected['selected_global_step']
        split=load('splits/'+adapter['domain']+'.json')
        assert not set(split['dev_ids'])&set(split['test_ids'])
        rows=load('results/per_question/glm0414_seed'+str(adapter['seed'])+'_'+adapter['domain']+'.json')
        assert {x['item_id'] for x in rows}=={str(x) for x in split['test_ids']}
    for point in points:
        key=point['model_run_key'];domain=point['domain']
        candidates=[r for r in summary['runs'] if r['domain']==domain and (r['run_key']==key or key=='glm0414_ssl_mean' and r['run_key'].startswith('glm0414_seed'))]
        if not candidates:
            # The displayed pooled SSL point has a separate registry key.
            candidates=[r for r in summary['runs'] if r['domain']==domain and 'ssl' in key.lower() and r['run_key'].startswith('glm0414_seed')]
        assert candidates,(key,domain)
        assert abs(statistics.mean(r['M'] for r in candidates)-point['m_mean'])<1e-12
        assert abs(statistics.mean(r['R@5'] for r in candidates)-point['r_mean'])<1e-12
    manifest_path=ROOT/'release_manifest.json'
    if manifest_path.exists():
        for entry in load('release_manifest.json')['files']:
            p=ROOT/entry['path']
            assert p.stat().st_size==entry['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==entry['sha256']
    print(json.dumps({'status':'PASS','paired_M_R_runs':len(summary['runs']),'retrieval_rows':questions,'memory_rows':articles,
        'dev_selection_records':6,'figure_points':10,'model_weights_loaded':False}))

if __name__=='__main__':main()
