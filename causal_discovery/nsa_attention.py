# causal_discovery/nsa_attention.py
import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from .pc_algorithm import PCAlgorithm
from typing import List, Dict, Any

class NSAttentionCausalDiscovery(nn.Module):
    """基于稀疏注意力的因果发现"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.feature_dim = config["feature_dim"]
        self.sparsity = config.get("sparsity", 0.3)
        self.n_heads = config.get("n_heads", 4)
        self.hardware_aware = config.get("hardware_aware", True)
        
        # NSA 稀疏注意力层
        self.attn = NativeSparseAttention(
            embed_dim=self.feature_dim,
            num_heads=self.n_heads,
            sparsity=self.sparsity,
            hardware_aware=self.hardware_aware
        )
        
        # 因果图预测器
        self.graph_predictor = nn.Sequential(
            nn.Linear(self.feature_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, X: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播
        
        参数:
            X: 输入特征 (batch_size, seq_len, feature_dim)
            mask: 注意力掩码 (可选)
            
        返回:
            因果图邻接矩阵 (batch_size, seq_len, seq_len)
        """
        # 应用稀疏注意力
        attn_output, attn_weights = self.attn(X, X, X, key_padding_mask=mask, need_weights=True)
        
        # 预测因果图
        batch_size, seq_len, _ = X.shape
        causal_graphs = torch.zeros(batch_size, seq_len, seq_len, device=X.device)
        
        for i in range(seq_len):
            for j in range(seq_len):
                if i != j:  # 忽略自环
                    # 连接两个节点的特征
                    pair_features = torch.cat([X[:, i, :], X[:, j, :]], dim=1)
                    edge_prob = torch.sigmoid(self.graph_predictor(pair_features))
                    causal_graphs[:, i, j] = edge_prob.squeeze()
        
        # 应用注意力权重作为先验
        causal_graphs = causal_graphs * attn_weights
        
        return causal_graphs
    
    def learn_from_data(self, data: np.ndarray, epochs: int = 100, lr: float = 0.001):
        """从数据中学习因果模型"""
        # 转换为PyTorch张量
        data_tensor = torch.tensor(data, dtype=torch.float32)
        
        # 创建训练数据 - 预测自身作为自监督
        X = data_tensor
        y = torch.eye(data.shape[1]).unsqueeze(0).repeat(data.shape[0], 1, 1)
        
        # 优化器
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.BCELoss()
        
        # 训练循环
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # 前向传播
            pred_graphs = self(X)
            
            # 计算损失
            loss = criterion(pred_graphs, y)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")
        
        return self
    
    def get_causal_graph(self, data: np.ndarray, threshold: float = 0.5) -> nx.DiGraph:
        """获取因果图"""
        with torch.no_grad():
            data_tensor = torch.tensor(data, dtype=torch.float32)
            pred_graphs = self(data_tensor)
            
            # 平均所有样本
            adj_matrix = pred_graphs.mean(dim=0).numpy()
            
            # 应用阈值
            adj_matrix = (adj_matrix > threshold).astype(int)
            np.fill_diagonal(adj_matrix, 0)  # 移除自环
            
            # 转换为NetworkX图
            graph = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
            
            return graph

class NativeSparseAttention(nn.Module):
    """硬件感知的稀疏注意力实现"""
    
    def __init__(self, embed_dim: int, num_heads: int, sparsity: float = 0.3, 
                 hardware_aware: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.sparsity = sparsity
        self.hardware_aware = hardware_aware
        
        # 确保head_dim正确
        assert self.head_dim * num_heads == embed_dim, "embed_dim必须能被num_heads整除"
        
        # 线性变换
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # 输出投影
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # 稀疏掩码 (初始化为None，将在前向传播中计算)
        self.sparse_mask = None
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None,
                need_weights: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """前向传播"""
        # 线性变换
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # 重塑为多头格式 (batch_size, seq_len, num_heads, head_dim)
        batch_size, tgt_len, _ = query.shape
        _, src_len, _ = key.shape
        
        q = q.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # 应用键填充掩码
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
        
        # 生成稀疏掩码
        if self.sparse_mask is None or self.sparse_mask.shape != attn_scores.shape:
            self.sparse_mask = self._generate_sparse_mask(attn_scores.shape, device=attn_scores.device)
        
        # 应用稀疏掩码
        sparse_attn_scores = attn_scores.masked_fill(self.sparse_mask == 0, float('-inf'))
        
        # 计算注意力权重
        attn_weights = torch.softmax(sparse_attn_scores, dim=-1)
        
        # 注意力输出
        attn_output = torch.matmul(attn_weights, v)
        
        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        
        if need_weights:
            # 返回平均注意力权重
            attn_weights = attn_weights.mean(dim=1)
            return attn_output, attn_weights
        else:
            return attn_output, None
    
    def _generate_sparse_mask(self, shape: torch.Size, device: torch.device) -> torch.Tensor:
        """生成稀疏注意力掩码"""
        _, num_heads, tgt_len, src_len = shape
        
        # 硬件感知模式
        if self.hardware_aware:
            # 块稀疏模式 (优化GPU内存访问)
            block_size = 64
            mask = torch.zeros(tgt_len, src_len, device=device)
            
            # 创建块对角线模式
            for i in range(0, tgt_len, block_size):
                j = i // block_size * block_size
                end_i = min(i + block_size, tgt_len)
                end_j = min(j + block_size, src_len)
                mask[i:end_i, j:end_j] = 1
            
            # 添加随机稀疏性
            rand_mask = torch.rand(tgt_len, src_len, device=device) < self.sparsity
            mask = (mask | rand_mask).float()
        else:
            # 完全随机稀疏性
            mask = (torch.rand(tgt_len, src_len, device=device) < self.sparsity).float()
        
        # 扩展为多头
        mask = mask.unsqueeze(0).unsqueeze(0).repeat(1, num_heads, 1, 1)
        
        return mask