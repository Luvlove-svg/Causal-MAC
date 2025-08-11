import torch
import torch.nn as nn
import torch.nn.functional as F
from causal_discovery.nsa_attention import NativeSparseAttention

class MultiHeadSparseAttention(nn.Module):
    """多头稀疏注意力适配器"""
    def __init__(self, dim, num_heads, sparsity=0.6):
        """
        Args:
            dim: 输入维度
            num_heads: 注意力头数
            sparsity: 注意力稀疏度
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # 多头稀疏注意力
        self.heads = nn.ModuleList([
            NativeSparseAttention(self.head_dim, sparsity)
            for _ in range(num_heads)
        ])
        
        # 输出投影
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, causal_mask=None):
        """
        前向传播
        Args:
            x: 输入序列 [batch_size, seq_len, dim]
            causal_mask: 因果掩码 [seq_len, seq_len]
        Returns:
            输出序列 [batch_size, seq_len, dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # 分割多头
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        x = x.permute(0, 2, 1, 3)  # [batch, heads, seq, head_dim]
        
        # 各头分别处理
        head_outputs = []
        for i, head in enumerate(self.heads):
            head_x = x[:, i, :, :]
            head_output = head(head_x, causal_mask)
            head_outputs.append(head_output)
        
        # 合并多头
        combined = torch.cat(head_outputs, dim=-1)
        combined = combined.permute(0, 2, 1)  # [batch, seq, dim]
        
        # 输出投影
        return self.out_proj(combined)

class InteractionConfidenceAttention(nn.Module):
    """IJCV 2025的交互置信度注意力"""
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=dim, 
            num_heads=num_heads,
            batch_first=True
        )
        self.confidence_proj = nn.Linear(dim, 1)
    
    def forward(self, states, actions):
        """计算状态-动作交互置信度"""
        # 状态-动作联合编码
        sa_embedding = torch.cat([states, actions], dim=-1)
        
        # 自注意力计算
        attn_output, attn_weights = self.attention(
            sa_embedding, sa_embedding, sa_embedding
        )
        
        # 计算置信度得分
        confidence = torch.sigmoid(self.confidence_proj(attn_output))
        return confidence