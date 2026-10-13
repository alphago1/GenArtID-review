"""Small CPU fixtures; these are not experimental observations."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from ga_parsing import parse_fixed
from ga_corpus import load_corpus
from ga_build_sft import build_ratio_dataset
from ga_memory_recall_probe import classify_memory
import review_cli


class CoreTests(unittest.TestCase):
    def test_original_top5_slots_not_refilled(self):
        s=json.dumps({'articles':['《民法典》第1条','《民法典》第1条',None,'bad','《民法典》第2条','《民法典》第3条']},ensure_ascii=False)
        self.assertEqual(parse_fixed(s).predictions,['《民法典》第1条','《民法典》第2条'])

    def test_fixed_precision(self):
        self.assertEqual(review_cli.per_question_scores(['《民法典》第1条'],['《民法典》第1条'])['P@5_fixed'],0.2)

    def test_corpus_only_slots(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'articles.json';p.write_text(json.dumps([{'name':'《民法典》第一条','content':'单元测试正文甲。'},{'name':'《民法典》第二条','content':'单元测试正文乙。'}],ensure_ascii=False),encoding='utf-8')
            result=build_ratio_dataset('civil',load_corpus(p,'civil'),(1,1))
            self.assertEqual(result['stats']['total_slots'],8)
            self.assertEqual(result['stats']['f_slots'],result['stats']['r_slots'])
            self.assertTrue(all(s['source_article_ids'] for s in result['samples']))

    def test_target_first_memory(self):
        self.assertEqual(classify_memory(False,0.85,0.99),'Eff.')
        self.assertEqual(classify_memory(False,0.84,0.85),'Dup.')
        self.assertEqual(classify_memory(False,0.84,0.84),'Err.')
        self.assertEqual(classify_memory(True,None,None),'Err.')

    def test_partial_eval_refused(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'pred.jsonl';g=Path(td)/'gold.jsonl'
            p.write_text('',encoding='utf-8');g.write_text(json.dumps({'item_id':'1','gold':['《民法典》第1条']})+'\n',encoding='utf-8')
            with self.assertRaises(ValueError):review_cli.evaluate(SimpleNamespace(predictions=p,gold=g,out=Path(td)/'out.json'))


if __name__=='__main__':unittest.main()
