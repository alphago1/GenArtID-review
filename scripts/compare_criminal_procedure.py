import json
import re
import os
import torch
from sentence_transformers import SentenceTransformer, util

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
    print(f"正在读取标准文件: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    standard_dict = {}
    if isinstance(data, list):
        for item in data:
            name = item.get('name', '')
            match = re.search(r'第([零一二三四五六七八九十百千]+)条', name)
            if match:
                cn_num = match.group(1)
                article_num = cn_to_int(cn_num)
            else:
                print(f"Warning: Could not parse article number from '{name}', skipping.")
                continue
            content = item.get('content', '').strip()
            standard_dict[str(article_num)] = content

    print(f"标准库加载完成，共 {len(standard_dict)} 条。")
    return standard_dict

def cn_to_int(cn_str):
    cn_map = {
        '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
        '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
        '零': 0, '百': 100, '千': 1000
    }
    if not cn_str: return 0
    if cn_str in cn_map: return cn_map[cn_str]
    result = 0
    temp = 0
    for char in cn_str:
        if char not in cn_map:
            continue
        val = cn_map[char]
        if val >= 10:
            if val > temp:
                result += val * (temp if temp > 0 else 1)
                temp = 0
            else:
                result += temp
                temp = val
        else:
            temp = val
    result += temp
    return result

def parse_generated_markdown(path):
    print(f"正在解析生成文件: {path}")
    generated_data = {}
    if not os.path.exists(path):
        print(f"文件不存在: {path}")
        return generated_data
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    pattern = r"## 第\s*(\d+)\s*条\s*\n+(.*?)(?=\n## 第|\Z)"
    matches = re.findall(pattern, content, re.DOTALL)
    for no, text in matches:
        generated_data[int(no)] = text.strip()
    print(f"解析完成，共 {len(generated_data)} 条。")
    return generated_data

def compare_file(model, standard_data, generated_file, report_title, report_path):
    generated_data = parse_generated_markdown(generated_file)

    print("正在为标准库构建语义索引...")
    standard_items = []
    for k, v in standard_data.items():
        standard_items.append({'no': int(k), 'text': v})
    standard_items.sort(key=lambda x: x['no'])
    standard_texts = [item['text'] for item in standard_items]
    standard_nos = [item['no'] for item in standard_items]

    if not standard_texts:
        print("标准库为空，无法比对。")
        return

    standard_embeddings = model.encode(standard_texts, convert_to_tensor=True)

    match_candidates = {}
    print("正在计算最佳匹配...")

    for no in sorted(generated_data.keys()):
        gen_text = generated_data[no]
        gen_embedding = model.encode(gen_text, convert_to_tensor=True)

        current_score = 0.0
        current_idx = -1
        try:
            current_idx = standard_nos.index(no)
            current_score = util.cos_sim(gen_embedding, standard_embeddings[current_idx]).item()
        except ValueError:
            pass

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

    claimed_std_nos = set()
    final_results = []
    PASS_THRESHOLD = 0.85
    FAIL_THRESHOLD = 0.60

    sorted_gen_nos = sorted(match_candidates.keys())

    for no in sorted_gen_nos:
        cand = match_candidates[no]
        if cand['current_score'] >= PASS_THRESHOLD:
            claimed_std_nos.add(no)
            cand['status_type'] = 'exact'
            cand['status_text'] = "✅ 完美/合格"

    for no in sorted_gen_nos:
        cand = match_candidates[no]
        if cand.get('status_type') == 'exact':
            final_results.append(cand)
            continue

        best_std_no = cand['best_std_no']
        best_score = cand['best_score']

        if best_score >= PASS_THRESHOLD:
            if best_std_no in claimed_std_nos:
                cand['status_type'] = 'repetition'
                cand['status_text'] = f"🔁 重复 (第{best_std_no}条已被占用)"
            else:
                cand['status_type'] = 'misplaced'
                cand['status_text'] = f"🔀 错位 (似第{best_std_no}条)"
                claimed_std_nos.add(best_std_no)
        else:
            cand['status_type'] = 'fail'
            cand['status_text'] = "❌ 严重偏差" if best_score >= FAIL_THRESHOLD else "🚫 幻觉"

        final_results.append(cand)

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

    total = len(results)
    if total == 0:
        print("没有条目。")
        return

    exact_count = sum(1 for r in results if r['类型'] == 'exact')
    misplaced_count = sum(1 for r in results if r['类型'] == 'misplaced')
    repetition_count = sum(1 for r in results if r['类型'] == 'repetition')
    fail_count = sum(1 for r in results if r['类型'] == 'fail')

    valid_count = exact_count + misplaced_count

    summary = f"\n---------- {report_title} 校验报告 (去重版) ----------\n"
    summary += f"总条数：{total}\n"
    summary += f"✅ 有效记忆: {valid_count} ({(valid_count)/total*100:.2f}%)\n"
    summary += f"   - 完全合格: {exact_count}\n"
    summary += f"   - 有效错位: {misplaced_count}\n"
    summary += f"🔁 无效重复: {repetition_count} ({(repetition_count)/total*100:.2f}%)\n"
    summary += f"❌ 错误/幻觉: {fail_count} ({(fail_count)/total*100:.2f}%)\n"
    summary += "==================================================\n"

    print(summary)

    with open(report_path, "a", encoding="utf-8") as f:
        f.write(f"\n# 刑事诉讼法背诵校验报告 - {report_title}\n\n")
        f.write(summary)

        f.write("\n### 1. 有效错位列表\n\n")
        for r in results:
            if r['类型'] == 'misplaced':
                f.write(f"#### 第{r['条号']}条 -> 实际是第{r['最佳匹配条号']}条\n")
                f.write(f"- **[生成]**: {r['生成内容']}\n")
                f.write(f"- **[最佳匹配]**: {r['最佳匹配内容']}\n\n")

        f.write("\n### 2. 重复/车轱辘话列表\n\n")
        rep_dict = {}
        for r in results:
            if r['类型'] == 'repetition':
                target = r['最佳匹配条号']
                if target not in rep_dict: rep_dict[target] = []
                rep_dict[target].append(r['条号'])
        for target, sources in rep_dict.items():
            f.write(f"#### 第{target}条 被重复使用于: {sources}\n")
            sample = next(r for r in results if r['条号'] == sources[0])
            f.write(f"- **[内容]**: {sample['生成内容'][:100]}...\n\n")

        f.write("\n### 3. 错误/幻觉列表\n\n")
        for r in results:
            if r['类型'] == 'fail':
                f.write(f"#### 第{r['条号']}条: {r['状态']}\n")
                f.write(f"- **[生成]**: {r['生成内容']}\n\n")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(base_dir)

    report_path = os.path.join(root_dir, "reports", "criminal_procedure_comparison_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 刑事诉讼法背诵校验报告汇总\n\n")

    standard_file = os.path.join(root_dir, "data", "中华人民共和国刑事诉讼法.json")
    qwen_file = os.path.join(root_dir, "outputs", "recite_raw", "criminal_procedure_recite.md")
    deepseek_file = os.path.join(root_dir, "outputs", "recite_raw", "criminal_procedure_deepseek_flash.md")

    model = load_model()
    standard_data = load_standard_data(standard_file)

    if os.path.exists(qwen_file):
        compare_file(model, standard_data, qwen_file, "Qwen3-8B 刑事诉讼法", report_path)
    else:
        print(f"Qwen output not found: {qwen_file}")

    if os.path.exists(deepseek_file):
        compare_file(model, standard_data, deepseek_file, "DeepSeek V4 Flash 刑事诉讼法", report_path)
    else:
        print(f"DeepSeek output not found: {deepseek_file}")

if __name__ == "__main__":
    main()
