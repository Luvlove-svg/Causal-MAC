import torch
import torch.nn as nn
from models.attention_modules import NativeSparseAttention
from models.quantize import quantize_layer_int4

class AgentPolicy(nn.Module):
    """轻量级智能体策略网络，集成NSA注意力和INT4量化"""
    def __init__(self, obs_dim, action_dim, config):
        """
        Args:
            obs_dim: 观测空间维度
            action_dim: 动作空间维度
            config: 模型配置字典
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.quantize = config.get('quantize', True)
        
        # 编码器网络
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )
        
        # NSA通信注意力模块
        self.comm_attention = NativeSparseAttention(
            dim=256,
            sparsity=config['sparsity'],
            block_size=config['block_size']
        )
        
        # 决策网络
        self.decoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        # 应用INT4量化
        if self.quantize:
            self._apply_quantization()

    def _apply_quantization(self):
        """应用INT4量化到线性层"""
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                quantize_layer_int4(layer)
        
        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                quantize_layer_int4(layer)

    def forward(self, local_obs, messages=None):
        """
        前向传播
        Args:
            local_obs: 本地观测 [batch_size, obs_dim]
            messages: 接收的消息 [batch_size, num_messages, message_dim]
        Returns:
            action_logits: 动作logits [batch_size, action_dim]
        """
        # 处理本地观测
        x = self.encoder(local_obs)
        
        # 处理接收到的消息
        if messages is not None:
            # 添加本地观测作为自我消息
            all_messages = torch.cat([x.unsqueeze(1), messages], dim=1)
            
            # 应用稀疏注意力
            comm_output = self.comm_attention(all_messages)
            
            # 残差连接
            x = x + comm_output[:, 0]
        
        # 生成动作分布
        return self.decoder(x)

class QLoRAWrapper(nn.Module):
    """QLoRA微调适配器，用于高效参数更新"""
    def __init__(self, base_model, lora_rank=8):
        """
        Args:
            base_model: 基础模型
            lora_rank: LoRA的秩
        """
        super().__init__()
        self.base_model = base_model
        self.lora_adapters = nn.ModuleDict()
        
        # 为所有线性层添加LoRA适配器
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Linear):
                # LoRA低秩适配器
                lora_down = nn.Linear(module.in_features, lora_rank, bias=False)
                lora_up = nn.Linear(lora_rank, module.out_features, bias=False)
                
                # 初始化
                nn.init.kaiming_uniform_(lora_down.weight, a=5**0.5)
                nn.init.zeros_(lora_up.weight)
                
                self.lora_adapters[name] = nn.Sequential(lora_down, lora_up)
    
    def forward(self, local_obs, messages=None):
        """带LoRA适配器的前向传播"""
        # 基础模型前向传播
        base_output = self.base_model(local_obs, messages)
        
        # 应用LoRA适配器
        for name, adapter in self.lora_adapters.items():
            # 定位原始模块
            module = self.base_model
            for part in name.split('.'):
                module = getattr(module, part)
            
            # 获取模块输入
            with torch.no_grad():
                input = module.input_cache  # 需要在前向中缓存
            
            # 计算LoRA适配
            lora_output = adapter(input)
            
            # 更新输出
            base_output = base_output + lora_output
        
        return base_output