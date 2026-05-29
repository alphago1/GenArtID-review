import json
import random
import os

# ==========================================
# 1. 配置与参数
# ==========================================
# ⚠️ 请确保这两个文件名与你本地的文件名一致
CIVIL_CODE_FILE = 'civil_code_articles_cleaned.json'  # 你的 1260 条法条原文文件 (包含 name 和 content 字段)
COT_QA_FILE = 'civil_code_remaining_rewritten.json'      # 你已经生成好的 200 条带 <think> 的 QA 文件
OUTPUT_FILE = 'llama_factory_27b_dataset.json' # 最终喂给 LLaMA-Factory 的文件

# ==========================================
# 2. 模板库 (极简配方)
# ==========================================
# 轨道一：纯背诵模板 (给 27B 建立硬核记忆)
FORWARD_TEMPLATES = [
    "请准确复述{name}的内容。",
    "{name}是怎么规定的？",
    "查询法条：{name}的完整内容是什么？",
    "请背诵{name}。"
]

RAG_SYSTEM_PROMPT = """你是一个专业的中国法律检索专家。
请仔细分析用户的案情，并在 <think> 和 </think> 标签内写下你的法律推理过程。
思考结束后，请根据案情的复杂程度，列出最相关的 1 到 3 条《中华人民共和国民法典》法条编号以构成完整的法律适用逻辑。
格式要求（按主次顺序，每行一条）：
《中华人民共和国民法典》第XXX条
《中华人民共和国民法典》第YYY条"""

def build_dataset():
    final_dataset = []
    
    # ==========================================
    # 处理轨道一：1260 条法条纯记忆注入
    # ==========================================
    print(f"📖 正在读取法条记忆库: {CIVIL_CODE_FILE} ...")
    if os.path.exists(CIVIL_CODE_FILE):
        with open(CIVIL_CODE_FILE, 'r', encoding='utf-8') as f:
            civil_data = json.load(f)
            
        valid_articles = 0
        for item in civil_data:
            name = item.get("name", "").strip()
            content = item.get("content", "").strip()
            
            if name and content:
                # 随机选一个背诵指令，防止指令格式过拟合
                instruction = random.choice(FORWARD_TEMPLATES).format(name=name)
                final_dataset.append({
                    "instruction": instruction,
                    "input": "",
                    "output": content
                })
                valid_articles += 1
        print(f"   ✅ 成功提取法条记忆数据: {valid_articles} 条")
    else:
        print(f"   ❌ 未找到文件 {CIVIL_CODE_FILE}，请检查路径！")

    # ==========================================
    # 处理轨道二：200 条黄金 CoT 推理数据
    # ==========================================
    print(f"🧠 正在读取 CoT 推理库: {COT_QA_FILE} ...")
    if os.path.exists(COT_QA_FILE):
        with open(COT_QA_FILE, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
            
        valid_qas = 0
        for item in qa_data:
            try:
                # 解析你提供的特定数据结构
                conv = item.get("conversation", [])[0]
                user_query = conv.get("user", "").strip()
                assistant_reply = conv.get("assistant", "").strip()
                
                if user_query and assistant_reply:
                    final_dataset.append({
                        "instruction": RAG_SYSTEM_PROMPT,
                        "input": user_query,
                        "output": assistant_reply
                    })
                    valid_qas += 1
            except Exception as e:
                print(f"   ⚠️ 跳过一条格式错误的 QA 数据: {e}")
        print(f"   ✅ 成功提取 CoT 逻辑对齐数据: {valid_qas} 条")
    else:
        print(f"   ❌ 未找到文件 {COT_QA_FILE}，请检查路径！")

    # ==========================================
    # 核心魔法：深度洗牌 (Shuffle)
    # ==========================================
    # 为什么必须打乱？因为如果模型先集中背完 1260 条，再集中做 200 条题，
    # 梯度下降方向会发生剧烈偏移，导致“灾难性遗忘”。打乱能让记忆和逻辑在神经元中完美交织。
    print("🔀 正在对数据集进行深度洗牌 (Shuffle) ...")
    random.seed(42) # 固定随机种子，保证每次运行结果一致
    random.shuffle(final_dataset)

    # ==========================================
    # 保存与质检
    # ==========================================
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=2)

    print("\n" + "="*50)
    print("🎉 终极微调数据集构建完毕！")
    print(f"📊 总数据量: {len(final_dataset)} 条")
    print(f"💾 已保存至: {OUTPUT_FILE}")
    print("="*50)

    # 打印前两条看看效果，确认混合成功
    if len(final_dataset) >= 2:
        print("\n🔍 抽样预览 (前 2 条数据):")
        for i in range(2):
            print(f"\n--- 样本 {i+1} ---")
            print(f"【Instruction】:\n{final_dataset[i]['instruction'][:100]}...")
            if final_dataset[i]['input']:
                print(f"【Input】:\n{final_dataset[i]['input']}")
            print(f"【Output (截断展示)】:\n{final_dataset[i]['output'][:150]}...")

if __name__ == '__main__':
    build_dataset()

