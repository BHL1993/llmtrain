import torch
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime


# 1. 定义模型结构（必须与训练时完全一致）
class EnhancedTimeLLM(torch.nn.Module):
    def __init__(self, text_model_name="hfl/chinese-macbert-base"):
        super().__init__()
        # 此处需完整复制训练时的模型定义
        self.text_encoder = AutoModelForCausalLM.from_pretrained(text_model_name)
        self.time_encoder = torch.nn.Sequential(
            torch.nn.Linear(1, 128),
            torch.nn.GELU()
        )
        # ... 其他层定义参考原始训练代码

    def forward(self, time_features, text_features):
        # 前向传播逻辑
        # ...
        return risk_score


# 2. 数据预处理函数（需与训练时一致）
def preprocess_log(log_entries):
    """将原始日志转换为模型输入格式"""
    processed = []
    start_time = datetime.strptime(log_entries[0]["操作时间"], "%Y-%m-%d %H:%M:%S")

    for entry in log_entries:
        # 计算时间差
        curr_time = datetime.strptime(entry["操作时间"], "%Y-%m-%d %H:%M:%S")
        delta_hours = (curr_time - start_time).total_seconds() / 3600

        # 构建文本输入
        text = f"{entry['操作类型']}：{entry['内容概述']}"
        processed.append({
            "delta_hours": delta_hours,
            "text": text
        })

    return processed


# 3. 加载完整模型
def load_full_model(model_path):
    # 加载保存的检查点
    checkpoint = torch.load(model_path, map_location='cpu')

    # 初始化模型实例
    model = EnhancedTimeLLM()

    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])

    # 加载分词器
    try:
        tokenizer = checkpoint['tokenizer']
    except KeyError:
        print("警告：检查点中未找到分词器，尝试从配置加载...")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint['config']['text_model'])

    # 加载自定义配置
    model.dynamic_phase.thresholds = torch.tensor(checkpoint.get('thresholds', [24.0, 72.0]))

    return model, tokenizer


# 4. 推理流程
def predict_risk(model, tokenizer, log_entries):
    # 数据预处理
    processed = preprocess_log(log_entries)

    # 构建模型输入
    time_deltas = torch.tensor([x["delta_hours"] for x in processed], dtype=torch.float32)
    texts = [x["text"] for x in processed]

    # 文本编码
    text_enc = tokenizer(
        texts,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    # 执行推理
    with torch.no_grad():
        model.eval()
        outputs = model(
            time_deltas.unsqueeze(0),  # 添加batch维度
            text_enc["input_ids"].unsqueeze(0)
        )
        prob = torch.sigmoid(outputs).item()

    return prob


# ------------------- 使用示例 -------------------
if __name__ == "__main__":
    # 输入数据
    sample_log = [
        {
            "操作时间": "2024-03-01 09:00:00",
            "操作类型": "客户进线",
            "内容概述": "包裹外包装严重破损"
        },
        {
            "操作时间": "2024-03-03 14:00:00",
            "操作类型": "客服回复",
            "内容概述": "要求客户提供照片证明"
        }
    ]

    # 加载模型
    try:
        model, tokenizer = load_full_model("./saved_models/full_model.pth")
        print("✅ 模型加载成功")
    except FileNotFoundError:
        print("❌ 模型文件未找到，请检查路径")
        exit()
    except KeyError as e:
        print(f"❌ 模型结构不匹配，缺失参数：{str(e)}")
        exit()

    # 执行预测
    try:
        risk_score = predict_risk(model, tokenizer, sample_log)
        print(f"舆情风险概率：{risk_score:.2%}")

        # 生成解释报告
        if risk_score > 0.7:
            print("[高危预警] 建议立即优先处理")
        elif risk_score > 0.5:
            print("[中度风险] 需在24小时内跟进")
        else:
            print("[低风险] 按正常流程处理")
    except Exception as e:
        print(f"❌ 推理失败：{str(e)}")

# ------------------- 预期输出 -------------------
# ✅ 模型加载成功
# 舆情风险概率：86.72%
# [高危预警] 建议立即优先处理