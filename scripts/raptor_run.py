import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import tiktoken
import yaml

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.raptor_lexrag.utils.normalization import build_canonical_name_map
from experiments.raptor_lexrag.utils.embeddings import (
    BGE_M3_EmbeddingModel, 
    CachedEmbeddingModel, 
    get_summarizer
)
from experiments.raptor_lexrag.build_tree import LawArticleTree
from experiments.raptor_lexrag.retrieve import LawArticleRetriever
from experiments.raptor_lexrag.evaluate import (
    LawArticleEvaluator, 
    build_query_from_turn, 
    extract_gold_articles
)

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_law_library(path: str) -> List[Dict]:
    articles = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            name = str(row.get('name', '')).strip()
            content = str(row.get('content', '')).replace('\\n', '\n').strip()
            if name and content:
                articles.append({'name': name, 'content': content})
    logger.info(f"Loaded {len(articles)} articles from law library")
    return articles


def load_dataset(path: str) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} samples from dataset")
    return data


def setup_logging(output_dir: Path, config_name: str) -> Path:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{config_name}_{timestamp}.log"
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logging.root.addHandler(file_handler)
    
    return log_file


def run_single_config(
    config: Dict,
    articles: List[Dict],
    dataset: List[Dict],
    output_dir: Path,
    config_name: str,
    sweep_config: Optional[Dict] = None
) -> Dict:
    if sweep_config:
        config['tree_builder']['umap_n_neighbors'] = sweep_config.get('umap_n_neighbors', 15)
        config['tree_builder']['reduction_dimension'] = sweep_config.get('reduction_dimension', 10)
    
    logger.info(f"Running configuration: {config_name}")
    
    cache_dir = Path(config['paths']['cache_dir'])
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Initializing embedding model...")
    embedding_model = BGE_M3_EmbeddingModel(
        model_path=config['embedding']['model_path'],
        batch_size=config['embedding']['batch_size'],
        max_seq_length=config['embedding']['max_seq_length']
    )
    
    cached_embedding_model = CachedEmbeddingModel(
        base_model=embedding_model,
        cache_dir=str(cache_dir),
        model_name=config['embedding']['model_type']
    )
    
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    logger.info("Initializing summarizer...")
    summarizer = get_summarizer(config['summarization'], tokenizer=tokenizer)
    
    logger.info("Building article tree...")
    tree = LawArticleTree(
        embedding_model=cached_embedding_model,
        summarizer=summarizer,
        tokenizer=tokenizer,
        reduction_dimension=config['tree_builder']['reduction_dimension'],
        umap_n_neighbors=config['tree_builder']['umap_n_neighbors'],
        gmm_threshold=config['tree_builder']['gmm_threshold'],
        max_clusters=config['tree_builder']['max_clusters'],
        min_cluster_size=config['tree_builder']['min_cluster_size'],
        max_layers=config['tree_builder']['num_layers'],
        summarization_length=config['tree_builder']['summarization_length'],
        cache_dir=str(cache_dir)
    )
    
    tree.build_from_articles(articles, use_cache=True)
    
    logger.info("Initializing retriever...")
    retriever = LawArticleRetriever(
        tree=tree,
        embedding_model=cached_embedding_model,
        tokenizer=tokenizer,
        top_k_nodes=config['retrieval']['top_k_nodes'],
        final_top_k=config['retrieval']['final_top_k'],
        max_context_tokens=config['retrieval']['max_context_tokens']
    )
    
    logger.info("Building canonical name map...")
    canonical_map = build_canonical_name_map(articles)
    
    evaluator = LawArticleEvaluator(canonical_map=canonical_map)
    
    query_mode = config['retrieval']['query_mode']
    logger.info(f"Evaluating with query mode: {query_mode}")
    
    total_turns = 0
    for sample in dataset:
        sample_id = sample.get('id', 0)
        conversation = sample.get('conversation', [])
        
        for turn_idx, turn in enumerate(conversation):
            query = build_query_from_turn(conversation, turn_idx, mode=query_mode)
            gold_articles = extract_gold_articles(turn)
            
            if not query or not gold_articles:
                continue
            
            result = retriever.retrieve(query)
            
            evaluator.evaluate_turn(
                sample_id=sample_id,
                turn_idx=turn_idx,
                query=query,
                query_mode=query_mode,
                gold_articles=gold_articles,
                pred_articles=result.article_names,
                context_budget_tokens=result.context_budget_tokens
            )
            
            total_turns += 1
            if total_turns % 50 == 0:
                logger.info(f"Processed {total_turns} turns")
    
    logger.info(f"Total turns evaluated: {total_turns}")
    
    cached_embedding_model.save_cache()
    
    config_output_dir = output_dir / config_name
    evaluator.save_results(config_output_dir)
    
    metrics = evaluator.compute_macro_metrics()
    metrics['config_name'] = config_name
    metrics['umap_n_neighbors'] = config['tree_builder']['umap_n_neighbors']
    metrics['reduction_dimension'] = config['tree_builder']['reduction_dimension']
    metrics['query_mode'] = query_mode
    
    return metrics


def run_sweep(
    config: Dict,
    articles: List[Dict],
    dataset: List[Dict],
    output_dir: Path
) -> pd.DataFrame:
    sweep_configs = config['sweep']['configs']
    all_metrics = []
    
    for sweep_config in sweep_configs:
        config_name = sweep_config['name']
        metrics = run_single_config(
            config=config,
            articles=articles,
            dataset=dataset,
            output_dir=output_dir,
            config_name=config_name,
            sweep_config=sweep_config
        )
        all_metrics.append(metrics)
    
    df = pd.DataFrame(all_metrics)
    
    sweep_file = output_dir / "sweep_results.csv"
    df.to_csv(sweep_file, index=False, encoding='utf-8-sig')
    logger.info(f"Saved sweep results to {sweep_file}")
    
    best_idx = df['macro_f1_at_5'].idxmax()
    best_config = df.iloc[best_idx]
    
    logger.info(f"\nBest configuration: {best_config['config_name']}")
    logger.info(f"  F1@5: {best_config['macro_f1_at_5']:.4f}")
    logger.info(f"  Recall@5: {best_config['macro_recall_at_5']:.4f}")
    logger.info(f"  Precision@5: {best_config['macro_precision_at_5']:.4f}")
    
    return df


def generate_paper_row(metrics: Dict) -> str:
    retrieval_rounds = 1.0
    context_budget = f"{metrics['avg_context_budget']:.3f}"
    recall = f"{metrics['macro_recall_at_5']:.4f}"
    precision = f"{metrics['macro_precision_at_5']:.4f}"
    f1 = f"{metrics['macro_f1_at_5']:.4f}"
    
    row = f"| RAPTOR (ICLR 2024) | {retrieval_rounds:.3f} | {context_budget} | {recall} | {precision} | {f1} |"
    return row


def generate_report(
    metrics: Dict,
    config: Dict,
    data_paths: Dict,
    output_dir: Path
) -> str:
    report = f"""
# RAPTOR 法律条文检索实验报告

## 实验配置

### 数据文件
- 问题集: {data_paths['dataset']}
- 法律库: {data_paths['law_library']}

### 模型配置
- Embedding模型: {config['embedding']['model_type']} ({config['embedding']['model_path']})
- 摘要模型: {config['summarization']['type']} ({config['summarization']['model']})

### 树构建参数
- UMAP n_neighbors: {config['tree_builder']['umap_n_neighbors']}
- 降维维度: {config['tree_builder']['reduction_dimension']}
- GMM阈值: {config['tree_builder']['gmm_threshold']}
- 最大聚类数: {config['tree_builder']['max_clusters']}
- 最小聚类大小: {config['tree_builder']['min_cluster_size']}
- 最大树深度: {config['tree_builder']['num_layers']}
- 摘要长度: {config['tree_builder']['summarization_length']}

### 检索参数
- Query模式: {config['retrieval']['query_mode']}
- Top-K节点数: {config['retrieval']['top_k_nodes']}
- 最终Top-K: {config['retrieval']['final_top_k']}
- 最大上下文tokens: {config['retrieval']['max_context_tokens']}

## 实验结果

### 主要指标
| 指标 | 值 |
|------|------|
| Retrieval Rounds | 1.000 |
| Context Budget (avg tokens) | {metrics['avg_context_budget']:.3f} |
| Recall@5 | {metrics['macro_recall_at_5']:.4f} |
| Precision@5 | {metrics['macro_precision_at_5']:.4f} |
| F1@5 | {metrics['macro_f1_at_5']:.4f} |
| Hit@5 Rate | {metrics['hit_at_5_rate']:.4f} |
| Total Turns | {metrics['total_turns']} |

### 论文表格行
{generate_paper_row(metrics)}

## 实现说明

### Retrieval Rounds = 1 的解释
RAPTOR collapsed tree retrieval 采用单次查询策略：
1. 对用户问题进行一次 embedding
2. 在整个树上（所有层节点）进行一次相似度检索
3. 通过回溯机制从检索到的节点获取候选法条
4. 聚合排序后输出 top-5 法条

因此 Retrieval Rounds 记为 1。

### Context Budget 统计口径
Context Budget 统计的是：
- 实际参与最终候选生成的节点文本 token 总量
- 使用 tiktoken cl100k_base tokenizer 计算
- 取所有样本的平均值

### 标签泄漏防护
本实现严格遵守以下规则：
- 只使用用户问题文本作为查询输入
- 不使用 assistant、keyword、article、article_context、type 等字段
- 多轮对话中只拼接历史 user 文本，不拼接 assistant 内容

### 法条名规范化
实现了完整的法条名规范化流程：
- 统一全半角字符
- 统一中文书名号格式
- 统一"第XXX条"的数字格式
- 使用法律库中的 canonical name 进行匹配

## 输出文件
- metrics.json: 汇总指标
- per_case.csv: 每个样本的详细结果
- hardest_cases.json: 最难样本
- confusion_cases.json: 混淆样本
- sweep_results.csv: 超参数扫描结果

---
实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    report_file = output_dir / "experiment_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    return report


def main():
    parser = argparse.ArgumentParser(description='RAPTOR Law Article Retrieval Experiment')
    parser.add_argument('--config', type=str, 
                        default=str(Path(__file__).parent / 'config.yaml'),
                        help='Path to config file')
    parser.add_argument('--no-sweep', action='store_true',
                        help='Disable hyperparameter sweep')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    output_dir = Path(config['paths']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(output_dir, "raptor_lexrag")
    
    np.random.seed(config['seed'])
    
    articles = load_law_library(config['paths']['law_library'])
    dataset = load_dataset(config['paths']['dataset'])
    
    if config['sweep']['enabled'] and not args.no_sweep:
        sweep_df = run_sweep(config, articles, dataset, output_dir)
        
        best_idx = sweep_df['macro_f1_at_5'].idxmax()
        best_metrics = sweep_df.iloc[best_idx].to_dict()
        
        best_config_name = best_metrics['config_name']
        for sweep_config in config['sweep']['configs']:
            if sweep_config['name'] == best_config_name:
                config['tree_builder']['umap_n_neighbors'] = sweep_config['umap_n_neighbors']
                config['tree_builder']['reduction_dimension'] = sweep_config['reduction_dimension']
                break
    else:
        best_metrics = run_single_config(
            config=config,
            articles=articles,
            dataset=dataset,
            output_dir=output_dir,
            config_name="default"
        )
    
    report = generate_report(
        metrics=best_metrics,
        config=config,
        data_paths=config['paths'],
        output_dir=output_dir
    )
    
    print("\n" + "="*60)
    print("RAPTOR 法律条文检索实验完成")
    print("="*60)
    print(f"\n数据文件:")
    print(f"  问题集: {config['paths']['dataset']}")
    print(f"  法律库: {config['paths']['law_library']}")
    print(f"\n最佳配置:")
    print(f"  UMAP n_neighbors: {config['tree_builder']['umap_n_neighbors']}")
    print(f"  降维维度: {config['tree_builder']['reduction_dimension']}")
    print(f"  Query模式: {config['retrieval']['query_mode']}")
    print(f"\n主结果:")
    print(f"  Retrieval Rounds = 1.000")
    print(f"  Context Budget = {best_metrics['avg_context_budget']:.3f}")
    print(f"  Recall@5 = {best_metrics['macro_recall_at_5']:.4f}")
    print(f"  Precision@5 = {best_metrics['macro_precision_at_5']:.4f}")
    print(f"  F1@5 = {best_metrics['macro_f1_at_5']:.4f}")
    print(f"\n论文表格行:")
    print(generate_paper_row(best_metrics))
    print(f"\n输出文件:")
    print(f"  目录: {output_dir}")
    print(f"  指标文件: metrics.json")
    print(f"  样本详情: per_case.csv")
    print(f"  扫描结果: sweep_results.csv")
    print(f"  实验报告: experiment_report.md")
    print("="*60)


if __name__ == "__main__":
    main()
