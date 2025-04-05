import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

# 嗯，用户现在的问题是关于如何构建一个模型，能够同时利用事件序列中的文本信息、时间间隔以及用户维度的历史特征来进行舆情预测。之前已经讨论过多模态深度学习方案，并且用户现在想把问题转化为预测下一个事件是否为舆情类型。用户提供的样本数据包含详细的事件操作序列，每个事件有时间、类型、沟通文本以及两个时间间隔，还有用户级别的历史进线和舆情次数。
#
# 首先，我需要确认用户的需求。用户希望整合多种数据源：文本（沟通文本）、时间间隔（两个时间间隔）、事件类型序列以及用户的历史统计特征。目标是根据这些信息预测下一个事件是否是舆情。需要设计一个模型，能够有效融合这些不同类型的数据。
#
# 接下来，回顾之前的对话。之前给出的方案包括多模态模型，使用BERT处理文本，LSTM处理事件序列，全连接处理统计特征。用户现在提供的样本数据更具体，需要根据这个样本调整模型结构。
#
# 用户提到的事件操作序列中的每个事件都有时间、类型、沟通文本、两个时间间隔。此外，用户级别的特征如历史进线次数和舆情次数也需要考虑。需要将这些信息整合到模型中。
#
# 首先，文本信息处理。每个事件都有沟通文本，可能需要为每个事件的文本生成嵌入，然后在整个序列中整合这些嵌入。例如，使用BERT对每个事件的沟通文本进行编码，得到每个事件的文本特征向量。
#
# 然后是事件类型和时间间隔。事件类型可以编码为嵌入向量，时间间隔可以作为数值特征输入。用户的历史统计特征可以作为全局特征输入。
#
# 需要考虑如何将这些不同特征融合。可能的方案是，对于每个事件，将类型嵌入、时间间隔数值特征、文本嵌入进行拼接，形成每个事件的特征向量。然后将事件序列输入到LSTM或Transformer中，捕捉时序模式。同时，用户的历史特征可以在序列处理之后，与序列的最终状态拼接，再输入到分类层。
#
# 此外，用户的历史特征可能对预测结果有重要影响，比如历史舆情次数多的用户可能更容易再次产生舆情。因此，需要将这些用户级别的特征有效地融入到模型中。
#
# 可能需要分几个部分处理数据：
#
# 1. **事件级别的特征处理**：
#
# - 类型编码：将事件类型（进线、跟进、舆情）转换为嵌入向量。
#
# - 时间间隔处理：将两个时间间隔作为数值特征，可能进行归一化。
#
# - 文本处理：使用预训练模型（如BERT）对沟通文本进行编码，得到文本特征。
#
# 每个事件的特征由类型嵌入、时间间隔、文本嵌入拼接而成。
#
# 2. **序列建模**：
#
# - 使用LSTM或Transformer处理事件序列，捕捉时序依赖。每个时间步输入的是上述拼接的事件特征。
#
# 3. **用户级别特征融合**：
#
# - 将用户的历史进线次数、历史舆情次数等统计特征进行处理（如通过全连接层），然后与序列的最终状态拼接，输入到分类层。
#
# 4. **分类层**：
#
# - 结合序列输出和用户特征，进行二分类预测（下一个事件是否为舆情）。
#
# 需要考虑如何处理可变长度的事件序列，比如使用填充和掩码。另外，文本处理部分可能需要预计算BERT嵌入以减少训练时的计算负担，特别是在处理长序列时。
#
# 另外，用户可能面临数据不平衡的问题，因为舆情事件可能较少。需要在损失函数中处理类别权重，或者在采样时进行过采样/欠采样。
#
# 在模型训练时，需要将每个事件的特征整合成一个张量，然后将整个序列输入到序列模型中。用户级别的特征则可以在序列处理完成后与之结合。
#
# 可能还需要考虑如何将用户特征与序列特征有效地交互，比如使用注意力机制或其他融合方法，而不仅仅是拼接。
#
# 总结来说，模型架构可能包括以下几个部分：
#
# - **事件特征编码器**：处理每个事件的类型、时间间隔、文本。
#
# - **序列编码器**：处理事件序列，捕捉时序模式。
#
# - **用户特征编码器**：处理用户历史统计特征。
#
# - **特征融合与分类**：将序列编码和用户特征融合，进行分类预测。
#
# 需要确保各个部分的数据维度和处理流程正确，同时考虑计算效率和模型的可训练性。例如，使用BERT处理每个事件的文本可能会增加计算量，可以考虑使用较小的预训练模型或对文本进行截断。
#
# 最后，需要给出具体的代码示例，展示如何定义模型、处理数据以及训练流程。可能需要使用PyTorch或TensorFlow框架，根据用户之前的代码偏好，用户之前接受的是TensorFlow代码，但最后一次对话中使用了PyTorch，可能需要确认，但看起来用户可能更倾向于PyTorch，但需要保持一致性。



class MultiModalEventModel(nn.Module):
    def __init__(self, num_event_types, user_feat_dim, bert_model_name='bert-base-chinese'):
        super().__init__()
        # 文本编码器 (每个事件的沟通文本)
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.text_proj = nn.Linear(768, 64)  # 降维

        # 事件类型编码
        self.event_embed = nn.Embedding(num_event_types, 32)

        # 时间间隔编码
        self.time_encoder = nn.Sequential(
            nn.Linear(2, 16),  # 输入: [距离上次间隔, 距离首次间隔]
            nn.ReLU()
        )

        # 序列编码器 (Bi-LSTM)
        self.lstm = nn.LSTM(
            input_size=64 + 32 + 16,  # 文本+类型+时间
            hidden_size=128,
            bidirectional=True,
            batch_first=True
        )

        # 用户特征编码
        self.user_encoder = nn.Sequential(
            nn.Linear(user_feat_dim, 64),
            nn.ReLU()
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(128 * 2 + 64, 64),  # LSTM输出(双向) + 用户特征
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, batch):
        # 处理每个事件的文本、类型、时间
        event_features = []
        for event in batch['events']:
            # 文本编码
            text_feat = self.bert(**event['text']).last_hidden_state[:, 0, :]
            text_feat = self.text_proj(text_feat)

            # 事件类型编码
            type_feat = self.event_embed(event['type'])

            # 时间编码
            time_feat = self.time_encoder(event['time_intervals'])

            # 拼接特征
            event_feat = torch.cat([text_feat, type_feat, time_feat], dim=-1)
            event_features.append(event_feat)

        # 构建序列 (batch_size, seq_len, hidden_dim)
        sequence = torch.stack(event_features, dim=1)

        # LSTM编码序列
        lstm_out, _ = self.lstm(sequence)  # lstm_out: (batch, seq_len, 256)

        # 取序列最后一个状态
        seq_feature = lstm_out[:, -1, :]

        # 用户特征编码
        user_feat = self.user_encoder(batch['user_features'])

        # 融合特征
        combined = torch.cat([seq_feature, user_feat], dim=-1)

        # 分类
        return self.classifier(combined)