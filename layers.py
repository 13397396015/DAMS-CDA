# model/layers.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # Linear projections
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        # Final projection
        output = self.out_proj(attn_output)

        return output, attn_weights


class MetapathAggregation(nn.Module):
    def __init__(self, embed_dim, num_metapaths, num_heads):
        super(MetapathAggregation, self).__init__()
        self.embed_dim = embed_dim
        self.num_metapaths = num_metapaths

        # Multi-head attention for each metapath
        self.attention = MultiHeadAttention(embed_dim, num_heads)

        # Metapath-level attention
        self.metapath_attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, metapath_embeddings):
        """
        Args:
            metapath_embeddings: List of embeddings from different metapaths
                                 Each with shape [batch_size, seq_len, embed_dim]
                                 or [batch_size, num_metapaths, seq_len, embed_dim]
        """
        # 检查输入形状
        if len(metapath_embeddings[0].shape) == 4:
            # 输入已经是 [batch_size, num_metapaths, seq_len, embed_dim]
            stacked_outputs = metapath_embeddings[0]
            batch_size, actual_num_metapaths, seq_len, _ = stacked_outputs.shape
        else:
            # 原始处理方式
            batch_size = metapath_embeddings[0].shape[0]
            seq_len = metapath_embeddings[0].shape[1]

            # Apply attention within each metapath view
            metapath_outputs = []
            for embeddings in metapath_embeddings:
                # Self-attention
                attn_output, _ = self.attention(embeddings, embeddings, embeddings)
                metapath_outputs.append(attn_output)

            # Stack metapath outputs
            stacked_outputs = torch.stack(metapath_outputs, dim=1)  # [batch_size, num_metapaths, seq_len, embed_dim]
            actual_num_metapaths = len(metapath_outputs)

        # Print shapes for debugging
        print(f"stacked_outputs shape: {stacked_outputs.shape}")

        # 计算实际的元路径数量
        actual_num_metapaths = stacked_outputs.shape[1]
        print(f"batch_size: {batch_size}, actual_num_metapaths: {actual_num_metapaths}, seq_len: {seq_len}")

        # Calculate metapath attention weights
        reshaped = stacked_outputs.reshape(-1, self.embed_dim)  # Flatten all dimensions except the last
        metapath_weights = self.metapath_attention(reshaped)
        metapath_weights = metapath_weights.view(batch_size, actual_num_metapaths, seq_len)
        metapath_weights = F.softmax(metapath_weights, dim=1)  # [batch_size, num_metapaths, seq_len]

        # Expand weights for broadcasting
        metapath_weights = metapath_weights.unsqueeze(-1)  # [batch_size, num_metapaths, seq_len, 1]

        # Weighted sum of metapath outputs
        weighted_sum = (stacked_outputs * metapath_weights).sum(dim=1)  # [batch_size, seq_len, embed_dim]

        return weighted_sum, metapath_weights.squeeze(-1)