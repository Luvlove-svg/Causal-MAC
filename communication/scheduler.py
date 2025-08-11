# communication/scheduler.py
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from causal_discovery import CausalDiscoveryManager
import torch
import torch.nn as nn
import torch.jit

class CausalCommunicationScheduler(nn.Module):
    """基于因果图的通信调度器"""
    
    def __init__(self, causal_manager: CausalDiscoveryManager, config: Dict[str, Any]):
        """
        初始化通信调度器
        
        参数:
            causal_manager: 因果发现管理器
            config: 通信配置
        """
        super().__init__()
        self.causal_manager = causal_manager
        self.config = config
        self.threshold = config.get("causal_threshold", 0.6)
        self.min_importance = config.get("min_importance", 0.3)
        self.max_messages = config.get("max_messages", 3)
        self.message_dim = config.get("message_dim", 16)
        
        # 通信门控网络
        self.gate_network = nn.Sequential(
            nn.Linear(self.causal_manager.feature_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # 消息压缩网络
        self.compression_network = nn.Sequential(
            nn.Linear(self.causal_manager.feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.message_dim)
        )
        
        # 消息缓存
        self.message_cache = {}
    
    def should_send(self, sender_id: str, receiver_id: str, 
                   sender_obs: np.ndarray, receiver_obs: np.ndarray) -> bool:
        """基于因果图决定是否发送消息"""
        # 如果因果图未初始化，默认不发送
        if self.causal_manager.causal_graph is None:
            return False
        
        # 计算观测差异
        obs_diff = np.abs(sender_obs - receiver_obs)
        
        # 检查重要特征是否超过阈值
        important_features = []
        for feat_idx in range(len(sender_obs)):
            # 检查特征是否因果相关
            causal_strength = self.causal_manager.get_causal_relations(feat_idx, feat_idx)
            if obs_diff[feat_idx] > self.threshold and causal_strength > self.min_importance:
                important_features.append(feat_idx)
        
        # 如果有重要特征变化，可能需要通信
        if len(important_features) > 0:
            # 使用门控网络做最终决策
            input_tensor = torch.tensor(
                np.concatenate([sender_obs, receiver_obs]), 
                dtype=torch.float32
            ).unsqueeze(0)
            
            with torch.no_grad():
                send_prob = self.gate_network(input_tensor).item()
            
            return send_prob > 0.5
        
        return False
    
    def compress_message(self, obs: np.ndarray) -> np.ndarray:
        """压缩观测为消息向量"""
        # 检查缓存
        cache_key = tuple(obs)
        if cache_key in self.message_cache:
            return self.message_cache[cache_key]
        
        # 使用压缩网络
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            message = self.compression_network(obs_tensor).squeeze(0).numpy()
        
        # 更新缓存
        self.message_cache[cache_key] = message
        return message
    
    def decompress_message(self, message: np.ndarray) -> np.ndarray:
        """从消息向量重建观测（近似）"""
        # 注意：这是一个有损过程，只能重建关键特征
        # 在实际应用中，可能需要更复杂的重建网络
        return message  # 简化实现，直接返回消息
    
    def schedule_communications(self, agent_observations: Dict[str, np.ndarray]) -> List[Tuple[str, str, np.ndarray]]:
        """为所有智能体调度通信"""
        scheduled_messages = []
        agent_ids = list(agent_observations.keys())
        
        # 限制最大消息数
        max_messages = min(self.max_messages, len(agent_ids) * (len(agent_ids) - 1))
        
        for sender_id in agent_ids:
            for receiver_id in agent_ids:
                if sender_id == receiver_id:
                    continue
                
                # 检查是否应该发送消息
                if self.should_send(sender_id, receiver_id, 
                                   agent_observations[sender_id], 
                                   agent_observations[receiver_id]):
                    # 压缩消息
                    message = self.compress_message(agent_observations[sender_id])
                    scheduled_messages.append((sender_id, receiver_id, message))
                    
                    # 检查是否达到最大消息数
                    if len(scheduled_messages) >= max_messages:
                        return scheduled_messages
        
        return scheduled_messages
    
    def update_causal_graph(self, observations: Dict[str, np.ndarray]):
        """更新因果图（委托给因果管理器）"""
        self.causal_manager.update_causal_graph(observations)
    
    def reset(self):
        """重置状态（如缓存）"""
        self.message_cache = {}
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'gate_network': self.gate_network.state_dict(),
            'compression_network': self.compression_network.state_dict(),
            'config': self.config
        }, path)
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.gate_network.load_state_dict(checkpoint['gate_network'])
        self.compression_network.load_state_dict(checkpoint['compression_network'])
        self.config = checkpoint['config']