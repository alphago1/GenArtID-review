import json
import re
import os
import sys
import torch
from sentence_transformers import SentenceTransformer, util

# 1. 加载预训练的中文语义模型
# 'shibing624/text2vec-base-chinese' 是一个效果很好的中文语义向量模型
MODEL_NAME = "shibing624/text2vec-base-chinese"

def load_model():
    print(f"正在加载模型：{MODEL_NAME} ...")
    try:
        return SentenceTransformer(MODEL_NAME)
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("尝试使用 'paraphrase-multilingual-MiniLM-L12-v2' 作为备选...")
        return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def load_standard_data(path):
    """加载官方标准法条 (JSON List format)"""
    print(f"正在读取标准文件: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    standard_dict = {}
    # Check if it's a list (as seen in inspection) or dict
    if isinstance(data, list):
        for item in data:
            # Assuming id starts at 0 for Article 1
            # We can also try to parse "name": "《...》第一条"
            # For robustness, let's try to extract number from name if possible, 
            # otherwise fallback to id + 1.
            
            # Simple mapping: id 0 -> 1, id 1 -> 2
            article_num = item.get('id') + 1
            content = item.get('content', '').strip()
            standard_dict[str(article_num)] = content
    elif isinstance(data, dict):
        # In case the user provided a dict format file I didn't see
        standard_dict = {str(k): v for k, v in data.items()}
            
    print(f"标准库加载完成，共 {len(standard_dict)} 条。")
    return standard_dict

def parse_generated_markdown(path):
    """解析模型生成的 Markdown 文件"""
    print(f"正在解析生成文件: {path}")
    generated_data = {}
    if not os.path.exists(path):
        print(f"文件不存在: {path}")
        return generated_data

    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 正则匹配：## 第N条\n\n内容
    # Modified to be more flexible with spaces
    pattern = r"## 第\s*(\d+)\s*条\s*\n+(.*?)(?=\n## 第|\Z)"
    matches = re.findall(pattern, content, re.DOTALL)
    
    for no, text in matches:
        generated_data[int(no)] = text.strip()
    
    print(f"解析完成，共 {len(generated_data)} 条。")
    return generated_data

def calculate_similarity(model, text1, text2):
    """计算两个文本的余弦相似度"""
    if not text1 or not text2:
        return 0.0
    embeddings = model.encode([text1, text2], convert_to_tensor=True)
    cosine_score = util.cos_sim(embeddings[0], embeddings[1])
    return cosine_score.item()

def compare_file(model, standard_data, generated_file, report_title, report_path):
    print(f"\n========== 开始比对: {report_title} ==========")
    generated_data = parse_generated_markdown(generated_file)
    
    # Pre-compute standard embeddings for global search (for detecting misalignments)
    print("正在为标准库构建语义索引（用于检测错位）...")
    standard_items = []
    for k, v in standard_data.items():
        standard_items.append({'no': int(k), 'text': v})
    
    # Sort by article number
    standard_items.sort(key=lambda x: x['no'])
    
    standard_texts = [item['text'] for item in standard_items]
    standard_nos = [item['no'] for item in standard_items]
    
    # Encode all standard texts
    standard_embeddings = model.encode(standard_texts, convert_to_tensor=True)
    
    results = []
    
    # Store all match candidates first to process "claims"
    # Structure: {gen_no: {'gen_text': str, 'best_match_idx': int, 'best_score': float, 'best_std_no': int, 'current_score': float}}
    match_candidates = {}
    
    print("正在计算所有条目的最佳匹配...")
    
    # 1. First Pass: Calculate Scores for ALL generated articles
    for no in sorted(generated_data.keys()):
        gen_text = generated_data[no]
        gen_embedding = model.encode(gen_text, convert_to_tensor=True)
        
        # Calculate Score with the Intended Article Number
        current_score = 0.0
        current_idx = -1
        try:
            current_idx = standard_nos.index(no)
            current_score = util.cos_sim(gen_embedding, standard_embeddings[current_idx]).item()
        except ValueError:
            pass
            
        # Global Search
        cos_scores = util.cos_sim(gen_embedding, standard_embeddings)[0]
        best_match_idx = torch.argmax(cos_scores).item()
        best_score = cos_scores[best_match_idx].item()
        best_match_no = standard_nos[best_match_idx]
        
        match_candidates[no] = {
            'gen_text': gen_text,
            'best_match_idx': best_match_idx,
            'best_score': best_score,
            'best_std_no': best_match_no,
            'best_std_text': standard_texts[best_match_idx],
            'current_score': current_score,
            'current_std_text': standard_texts[current_idx] if current_idx != -1 else ""
        }

    # 2. Second Pass: Determine Status based on Claims
    # Priority: Exact Matches claim first. Then First-Come-First-Served for Misplaced.
    
    claimed_std_nos = set()
    final_results = []
    
    PASS_THRESHOLD = 0.85
    FAIL_THRESHOLD = 0.60
    
    sorted_gen_nos = sorted(match_candidates.keys())
    
    # Step 2a: Mark Exact Matches
    for no in sorted_gen_nos:
        cand = match_candidates[no]
        # If current position matches well enough, it claims the spot
        if cand['current_score'] >= PASS_THRESHOLD:
            claimed_std_nos.add(no)
            cand['status_type'] = 'exact'
            cand['status_text'] = "✅ 完美/合格"
    
    # Step 2b: Process others (Misplaced vs Repetition vs Fail)
    for no in sorted_gen_nos:
        cand = match_candidates[no]
        
        # Skip if already determined as exact
        if cand.get('status_type') == 'exact':
            final_results.append(cand)
            continue
            
        best_std_no = cand['best_std_no']
        best_score = cand['best_score']
        
        if best_score >= PASS_THRESHOLD:
            # It's a high quality match, but is it a duplicate?
            if best_std_no in claimed_std_nos:
                # Already claimed (by an exact match OR a previous misplaced match)
                cand['status_type'] = 'repetition'
                cand['status_text'] = f"🔁 重复 (第{best_std_no}条已被占用)"
            else:
                # Not claimed yet, so this is a valid "Misplaced" memory
                cand['status_type'] = 'misplaced'
                cand['status_text'] = f"🔀 错位 (似第{best_std_no}条)"
                claimed_std_nos.add(best_std_no)
        else:
            # Low score
            cand['status_type'] = 'fail'
            if best_score >= FAIL_THRESHOLD:
                cand['status_text'] = "❌ 严重偏差"
            else:
                cand['status_text'] = "🚫 幻觉"
        
        final_results.append(cand)
        
        # Log
        if cand['status_type'] == 'misplaced':
             print(f"第{no}条 -> 错位匹配第{cand['best_std_no']}条")
        elif cand['status_type'] == 'repetition':
             # Only print repetition if it's the first few times to avoid spam?
             pass

    # 3. Generate Report Data
    results = []
    for no in sorted_gen_nos:
        cand = match_candidates[no]
        results.append({
            "条号": no,
            "原位相似度": cand['current_score'],
            "最佳匹配条号": cand['best_std_no'],
            "最佳相似度": cand['best_score'],
            "状态": cand['status_text'],
            "类型": cand['status_type'],
            "生成内容": cand['gen_text'],
            "标准内容": cand['current_std_text'],
            "最佳匹配内容": cand['best_std_text']
        })

    # 统计报告
    total = len(results)
    if total == 0:
        print("没有找到可比对的条目。")
        return

    exact_match_count = sum(1 for r in results if r['类型'] == 'exact')
    misplaced_count = sum(1 for r in results if r['类型'] == 'misplaced')
    repetition_count = sum(1 for r in results if r['类型'] == 'repetition')
    fail_count = sum(1 for r in results if r['类型'] == 'fail')
    
    # Valid Memory = Exact + Misplaced (Unique)
    valid_memory_count = exact_match_count + misplaced_count
    
    avg_best_score = sum(r['最佳相似度'] for r in results) / total
    
    summary = f"\n---------- {report_title} 校验报告 (去重版) ----------\n"
    summary += f"总条数：{total}\n"
    summary += f"✅ 有效记忆 (内容匹配且唯一): {valid_memory_count} ({(valid_memory_count)/total*100:.2f}%)\n"
    summary += f"   - 完全合格: {exact_match_count}\n"
    summary += f"   - 有效错位: {misplaced_count}\n"
    summary += f"� 无效重复 (车轱辘话): {repetition_count} ({(repetition_count)/total*100:.2f}%)\n"
    summary += f"❌ 错误/幻觉: {fail_count} ({(fail_count)/total*100:.2f}%)\n"
    summary += "==================================================\n"
    
    print(summary)
    
    # Save to file
    with open(report_path, "a", encoding="utf-8") as f:
        f.write(summary)
        
        f.write("\n### 1. 有效错位列表 (First Occurrence)\n\n")
        for r in results:
            if r['类型'] == 'misplaced':
                f.write(f"#### 第{r['条号']}条 -> 实际是第{r['最佳匹配条号']}条\n")
                f.write(f"- **[生成]**: {r['生成内容']}\n")
                f.write(f"- **[最佳匹配]**: {r['最佳匹配内容']}\n\n")
        
        f.write("\n### 2. 重复/车轱辘话列表 (Repetitions)\n\n")
        # Group repetitions by what they are repeating
        rep_dict = {}
        for r in results:
            if r['类型'] == 'repetition':
                target = r['最佳匹配条号']
                if target not in rep_dict: rep_dict[target] = []
                rep_dict[target].append(r['条号'])
        
        for target, sources in rep_dict.items():
            f.write(f"#### 第{target}条 被重复使用于: {sources}\n")
            # Sample one content
            sample_r = next(r for r in results if r['条号'] == sources[0])
            f.write(f"- **[生成内容示例]**: {sample_r['生成内容'][:100]}...\n\n")

        f.write("\n### 3. 错误/幻觉列表\n\n")
        for r in results:
            if r['类型'] == 'fail':
                f.write(f"#### 第{r['条号']}条: {r['状态']}\n")
                f.write(f"- **[生成]**: {r['生成内容']}\n\n")
        f.write("\n---\n")

def main():
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(base_dir)  # recitation_check/

    # Clear previous report
    report_path = os.path.join(root_dir, "reports", "comparison_report_v3.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 民法典背诵校验报告 (去重版)\n\n")

    # 文件路径
    standard_file = os.path.join(root_dir, "data", "civil_code_articles_cleaned.json")
    deepseek_file = os.path.join(root_dir, "outputs", "recite_raw", "civil_code_deepseek_sf.md")
    recite_file = os.path.join(root_dir, "outputs", "recite_raw", "civil_code_recite.md")
    lora_file = os.path.join(root_dir, "outputs", "recite_raw", "civil_code_recite_lora.md")
    
    # 1. Load Model
    model = load_model()
    
    # 2. Load Standard Data
    standard_data = load_standard_data(standard_file)
    
    # 3. Compare Deepseek File
    compare_file(model, standard_data, deepseek_file, "DeepSeek V3", report_path)

    # 4. Compare Recite File
    compare_file(model, standard_data, recite_file, "Qwen3-8B", report_path)

    # 5. Compare LoRA File
    compare_file(model, standard_data, lora_file, "LoRA 微调版", report_path)

if __name__ == "__main__":
    main()
