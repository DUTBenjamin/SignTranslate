# CNN_LSTM.py
"""
手语识别模型定义文件，包含：
1. FeatureExtractor: 基于ResNet的CNN特征提取器
2. MultiModalCNNTransformerModel: 支持多模态输入的Transformer模型
"""
import torch
import torch.nn as nn
from torchvision import models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# -------------------- 特征提取器 --------------------
class FeatureExtractor(nn.Module):
    def __init__(self, model_name='resnet18'):
        super(FeatureExtractor, self).__init__()
        self.model_name = model_name

        # 加载预训练ResNet模型
        if model_name == 'resnet18':
            original_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.feature_dim = 512
        elif model_name == 'resnet50':
            original_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # 分解网络层
        self.feature_layers = nn.Sequential(
            original_model.conv1,
            original_model.bn1,
            original_model.relu,
            original_model.maxpool,
            original_model.layer1,
            original_model.layer2,
            original_model.layer3,
            original_model.layer4,
            original_model.avgpool
        )

        # 冻结所有参数（使用预训练特征）
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.feature_layers(x)
        x = torch.flatten(x, 1)
        return x


# 多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        out = self.fc_out(out)
        return out


# Transformer编码器
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query):
        attention = self.attention(value, key, query)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


# 主模型（多模态CNN-Transformer）
class MultiModalCNNTransformerModel(nn.Module):
    def __init__(self, feature_dim=638, num_classes=100, transformer_layers=3, heads=8, dropout=0.5):
        super(MultiModalCNNTransformerModel, self).__init__()

        self.feature_dim = feature_dim
        self.num_classes = num_classes

        # Transformer编码器
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=feature_dim, nhead=heads, dropout=dropout),
            num_layers=transformer_layers
        )

        # 多头注意力机制
        self.attention = MultiHeadAttention(feature_dim, heads)

        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, lengths=None):
        """
        输入参数：
        x: 特征序列 (batch_size, seq_len, feature_dim)
        lengths: 实际序列长度 (batch_size,)
        """
        batch_size, seq_len, _ = x.shape

        # 处理变长序列
        if lengths is not None:
            max_len = lengths.max().item()
            x = x[:, :max_len, :]

        # Transformer编码
        out = self.transformer(x.permute(1, 0, 2))  # Transformer输入需要是(seq_len, batch, feature)

        # 多头注意力
        attn_out = self.attention(out.permute(1, 0, 2), out.permute(1, 0, 2), out.permute(1, 0, 2))

        # 使用最后一个时间步的输出进行分类
        logits = self.classifier(attn_out[:, -1, :])

        return logits