import math

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig


# 好的，用户现在问的是是否可以用Transformer替代之前模型中使用的LSTM。我需要先回顾之前的对话，看看之前的设计是怎样的。用户之前的数据结构包括事件操作序列，每个事件有类型、时间间隔、沟通文本，还有用户维度的历史信息。之前的模型用了多模态结构，结合BERT处理文本，LSTM处理事件序列，再加上用户特征。现在用户想知道是否可以用Transformer代替LSTM。
#
# 首先，我需要理解为什么用户会想用Transformer。可能的原因包括Transformer在长序列处理上的优势，或者想提升模型的表现。但也要考虑Transformer的缺点，比如计算资源需求高，数据量是否足够，以及实现复杂度。
#
# 接下来，我需要比较LSTM和Transformer在用户场景下的适用性。用户的事件序列可能长度不一，但通常物流场景中的事件序列可能不会特别长，可能几十到几百步？如果序列较长，Transformer的自注意力机制可能更有效捕捉长期依赖。而LSTM在处理长序列时可能会有梯度消失的问题。
#
# 然后，考虑用户的数据量。Transformer通常需要更多的数据来训练，尤其是因为其参数较多。如果用户的数据量较小，可能LSTM更合适。但如果数据量足够，Transformer可能表现更好。
#
# 另外，用户的需求是否对实时性有要求。Transformer的推理速度可能比LSTM慢，尤其是在没有GPU的情况下。如果部署环境有资源限制，这点需要考虑。
#
# 接下来，我需要设计用Transformer替代LSTM后的模型结构。需要将原来的LSTM层换成Transformer编码器层。同时，需要注意位置编码，因为Transformer本身不具备处理序列顺序的能力，需要添加位置编码。而原来的模型中可能已经通过LSTM的结构隐式处理了序列顺序。
#
# 另外，用户的数据中每个事件已经包含了时间间隔信息，这些时间信息是否需要作为位置编码的一部分，还是单独处理。可能需要将时间间隔信息融入到Transformer的输入中，比如作为额外的特征向量与事件嵌入拼接。
#
# 还需要考虑如何处理变长序列。在之前的LSTM模型中，可能通过填充来处理变长序列，而Transformer同样需要处理填充，使用注意力掩码来忽略填充部分的影响。
#
# 然后，我需要给出具体的代码示例，展示如何修改原来的模型，将LSTM替换为Transformer。同时，需要解释代码中的关键部分，比如自注意力层的配置、位置编码的实现、以及如何整合多模态特征。
#
# 还需要讨论这种替换后的优缺点，比如模型容量增加，可能提升性能，但训练时间和资源消耗也会增加。此外，可能需要调整超参数，如注意力头的数量、层数等，以适应具体任务。
#
# 最后，要给出建议，比如在数据量充足的情况下尝试Transformer，或者在资源受限时使用更轻量级的Transformer变体，如DistilBERT或更小的层数。同时，提醒用户在实际应用中进行实验，比较两种模型的性能，选择最适合的方案。
#
# 可能用户还关心如何在实际部署中优化Transformer模型，比如模型蒸馏、量化或使用更高效的注意力机制，如线性注意力。这些也可以在回答中提及，帮助用户全面考虑替换后的影响和优化方向。

class EventTransformerModel(nn.Module):
    def __init__(self, num_event_types, user_feat_dim,
                 d_model=128, nhead=4, num_layers=3):
        super().__init__()

        # 文本编码（冻结BERT前几层）
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        for param in self.bert.parameters()[:6]:  # 冻结前6层
            param.requires_grad = False
        self.text_proj = nn.Linear(768, 64)

        # 事件类型和时间编码
        self.event_embed = nn.Embedding(num_event_types, 32)
        self.time_encoder = nn.Linear(2, 32)

        # Transformer编码器
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=256
            ),
            num_layers=num_layers
        )

        # 位置编码（考虑时间间隔）
        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            max_len=500,
            time_aware=True  # 使用实际时间间隔调整位置编码
        )

        # 用户特征编码
        self.user_encoder = nn.Sequential(
            nn.Linear(user_feat_dim, 64),
            nn.ReLU()
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model + 64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, batch):
        # 事件特征融合
        event_feats = []
        for event in batch['events']:
            # 文本特征 [batch, 64]
            text_feat = self.bert(**event['text']).last_hidden_state[:, 0, :]
            text_feat = self.text_proj(text_feat)

            # 事件类型 [batch, 32]
            type_feat = self.event_embed(event['type'])

            # 时间特征 [batch, 32]
            time_feat = self.time_encoder(event['time_intervals'])

            # 合并 [batch, 128]
            combined = torch.cat([text_feat, type_feat, time_feat], dim=-1)
            event_feats.append(combined)

        # 构建序列 [batch, seq_len, 128]
        src = torch.stack(event_feats, dim=1)

        # 加入时间感知的位置编码
        src = self.pos_encoder(src, batch['time_intervals'])

        # Transformer编码 [batch, seq_len, d_model]
        src = src.permute(1, 0, 2)  # Transformer需要(seq_len, batch, dim)
        memory = self.transformer(src)
        seq_feat = memory[-1, :, :]  # 取最后一个事件

        # 用户特征融合
        user_feat = self.user_encoder(batch['user_features'])
        combined = torch.cat([seq_feat, user_feat], dim=-1)

        return self.classifier(combined)


class PositionalEncoding(nn.Module):
    """时间感知的位置编码"""

    def __init__(self, d_model, max_len=500, time_aware=True):
        super().__init__()
        self.time_aware = time_aware              # 是否启用时间感知模式
        self.scale = nn.Parameter(torch.ones(1))  # 可学习的缩放因子

    def forward(self, x, time_intervals):
        # x: [batch, seq_len, d_model]
        # time_intervals: [batch, seq_len, 2]

        batch_size, seq_len, _ = x.size()

        # 使用累计时间间隔生成位置编码
        if self.time_aware:
            # 使用距离首次事件的时间间隔作为位置基数
            pos = time_intervals[:, :, 1].unsqueeze(-1)  # [batch, seq_len, 1]
        else:
            pos = torch.arange(seq_len, device=x.device).float()

        # 生成频率项 (div_term)
        # 实现细节：
        #
        # torch.arange(0, x.size(-1), 2)：生成偶数值索引[0, 2, 4, ..., d_model - 2]。
        #
        # -math.log(10000.0) / x.size(-1)：计算缩放因子。
        #
        # torch.exp：将线性值映射为指数空间。
        div_term = torch.exp(torch.arange(0, x.size(-1), 2) * (-math.log(10000.0) / x.size(-1)))  # 形状: [d_model/2]

        # 计算相位 (pe)
        # 正弦和余弦函数：
        #
        # pos（形状 [batch, seq_len, 1]）与 div_term（形状 [d_model/2]）相乘，通过广播得到形状 [batch, seq_len, d_model/2]。
        #
        # 对每个元素分别计算正弦和余弦，拼接后得到形状 [batch, seq_len, d_model]。
        pe = pos * div_term  # 广播后形状: [batch, seq_len, d_model/2]
        pe = torch.cat([torch.sin(pe), torch.cos(pe)], dim=-1)  # [batch, seq_len, d_model]

        # 融合位置编码
        # 缩放与相加：
        # self.scale：可学习参数，调整位置编码的幅度。
        #
        # pe[:, :, :x.size(-1)]：确保 pe 的最后一维与 x 的嵌入维度一致（防止维度不匹配）。
        #
        # 最终将位置编码与输入 x 相加，增强输入的时间感知能力。

        return x + self.scale * pe[:, :, :x.size(-1)]