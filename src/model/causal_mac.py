import torch
import torch.nn as nn


class CausalMAC(nn.Module):
    def __init__(self, config):
        super(CausalMAC, self).__init__()
        self.config = config

        # 代理特征编码器
        self.agent_encoder = nn.Sequential(
            nn.Linear(config.agent_feature_dim, config.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(config.hidden_dim)
        )

        # 多智能体通信模块
        self.communication_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.num_heads,
                dim_feedforward=config.hidden_dim * 4,
                batch_first=True
            ) for _ in range(config.num_layers)
        ])

        # 修改预测头：移除Sigmoid，输出logits
        self.relation_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1)  # 移除Sigmoid层
        )

    def forward(self, agents):
        # 检查输入维度并确保是三维
        if agents.dim() == 2:
            agents = agents.unsqueeze(0)

        # 获取形状信息
        batch_size, num_agents, _ = agents.shape

        # 编码代理特征
        flat_agents = agents.view(-1, self.config.agent_feature_dim)
        agent_emb_flat = self.agent_encoder(flat_agents)
        agent_emb = agent_emb_flat.view(batch_size, num_agents, self.config.hidden_dim)

        # 多智能体通信
        for layer in self.communication_layers:
            agent_emb = layer(agent_emb)

        # 预测因果关系
        relations = []
        for i in range(num_agents):
            for j in range(num_agents):
                if i != j:
                    agent_i = agent_emb[:, i, :]
                    agent_j = agent_emb[:, j, :]
                    relation = self.relation_predictor(
                        torch.cat([agent_i, agent_j], dim=-1))
                    relations.append(relation)

        # 将列表转换为张量并展平
        if relations:
            return torch.stack(relations, dim=1).squeeze(-1)
        else:
            return torch.zeros(batch_size, 0, device=agents.device)

    def compute_loss(self, pred_relations, gt_relations):
        """
        使用BCEWithLogitsLoss计算损失
        """
        if pred_relations.numel() == 0 or gt_relations.numel() == 0:
            return torch.tensor(0.0, device=self.config.device)

        # 统一形状
        pred_relations = pred_relations.view(-1)
        gt_relations = gt_relations.view(-1)

        # 使用安全的损失函数
        return nn.functional.binary_cross_entropy_with_logits(
            pred_relations,
            gt_relations
        )