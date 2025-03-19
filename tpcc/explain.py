import matplotlib.pyplot as plt
import seaborn as sns

# 1. 注意力权重可视化（核心解释）
# 通过可视化交叉注意力层的权重，观察模型如何融合时序日志特征和统计特征。
# 解读：模型在处理第三步"客服回访"时，高度关注"处理时效"这一统计特征，说明处理延迟可能是舆情风险的关键因素。
def visualize_attention(attention_weights, log_steps, stats_features):
    """
    attention_weights: (num_heads, seq_len, stats_len)
    log_steps: 时序日志操作列表 ["客户进线", "预约回访", ...]
    stats_features: 统计特征名称 ["历史舆情量", "舆情率", "处理时效"]
    """
    # 平均多头的注意力权重
    avg_attention = attention_weights.mean(0)  # (seq_len, stats_len)

    plt.figure(figsize=(10, 6))
    sns.heatmap(avg_attention,
                xticklabels=stats_features,
                yticklabels=log_steps,
                annot=True, cmap="YlGnBu")
    plt.title("操作步骤与统计特征的交叉注意力权重")
    plt.xlabel("统计特征")
    plt.ylabel("操作步骤")
    plt.show()

# 示例调用
log_steps = [log["操作类型"] for log in log_sequence]
stats_features = ["历史舆情量", "舆情率", "处理时效"]
attention_weights = model.get_cross_attention()  # 假设模型返回交叉注意力权重
visualize_attention(attention_weights, log_steps, stats_features)

# 2. 特征重要性分析
# (1) 统计特征重要性（SHAP值分析）
# 解读："处理时效"的SHAP值分布最广，说明其对预测结果影响最大。
import shap

# 提取统计特征和模型预测函数
stats_data = np.array([[5, 0.2, 48], [1, 0.05, 12], [2, 0.5, 37]])
def predict_wrapper(stats):
    # 将统计特征与固定时序日志结合进行预测
    return model.predict(stats, logs_embeddings)

# 计算SHAP值
explainer = shap.KernelExplainer(predict_wrapper, stats_data)
shap_values = explainer.shap_values(stats_data)

# 可视化
shap.summary_plot(shap_values, stats_data, feature_names=stats_features)

# (2) 时序日志关键词重要性（LIME文本解释）
# 解读：模型对"索赔金额太高"和"仍未解决"等负面表述敏感，这些关键词显著提升了舆情概率。
from lime import lime_text
from lime.lime_text import LimeTextExplainer

# 将时序日志转换为文本序列
log_texts = ["[{}] {}".format(log["操作类型"], log["内容概述"]) for log in log_sequence]
full_text = " ".join(log_texts)

def text_predict(text):
    # 将文本重新解析为日志格式并预测
    parsed_logs = parse_text_to_logs(text)  # 自定义解析函数
    return model.predict(parsed_logs)

explainer = LimeTextExplainer()
exp = explainer.explain_instance(full_text, text_predict, num_features=5)
exp.show_in_notebook()

# 3. 时间模式分析
# 分析操作步骤的时间间隔与舆情概率的关系
# 解读：当两次操作间隔超过24小时时，舆情概率显著上升。
import pandas as pd

# 计算时间间隔特征
def calculate_intervals(logs):
    timestamps = [pd.to_datetime(log["操作时间"]) for log in logs]
    return [ (timestamps[i+1] - timestamps[i]).total_seconds()/3600
            for i in range(len(timestamps)-1) ]

# 绘制时间间隔与预测概率的关系
intervals = calculate_intervals(log_sequence)
probs = [model.predict_step(logs[:i+1]) for i in range(len(logs))]

plt.plot(intervals, probs[1:], 'o-')
plt.xlabel("相邻操作间隔（小时）")
plt.ylabel("舆情概率预测值")
plt.title("处理延迟与舆情风险关系")

# 4. 案例对比分析
# 选择相似案例对比模型预测差异的原因：
# 案例1（高舆情风险）
# 解读：在案例1中，模型更关注"处理时效"特征，说明长时间未解决是风险主因。
case1_logs = [...]  # 实际产生舆情的日志
case1_stats = [5, 0.2, 72]

# 案例2（低舆情风险）
case2_logs = [...]  # 未产生舆情的日志
case2_stats = [5, 0.2, 48]

# 对比注意力权重差异
attn1 = model.get_cross_attention(case1_logs, case1_stats)
attn2 = model.get_cross_attention(case2_logs, case2_stats)
diff_attn = attn1 - attn2

visualize_attention(diff_attn, log_steps, stats_features)
