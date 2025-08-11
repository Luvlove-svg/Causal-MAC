# communication/reward_calculator.py
import torch
import torch.jit
import numpy as np
from typing import Dict, Tuple

@torch.jit.script
def counterfactual_reward(
    action_with_message: torch.Tensor,
    action_without_message: torch.Tensor,
    optimal_action: torch.Tensor,
    baseline_reward: float,
    message_cost: float
) -> Tuple[float, float]:
    """
    计算反事实通信奖励
    
    参数:
        action_with_message: 接收消息后的动作 (向量)
        action_without_message: 假设未接收消息的动作 (向量)
        optimal_action: 最优动作 (向量)
        baseline_reward: 当前环境的基础奖励
        message_cost: 发送消息的成本
        
    返回:
        sender_reward: 发送者奖励
        total_reward: 总奖励 (基础奖励 + 通信奖励)
    """
    # 计算动作改进
    improvement_with_msg = torch.norm(action_with_message - optimal_action)
    improvement_without_msg = torch.norm(action_without_message - optimal_action)
    
    # 计算消息带来的改进增益
    improvement_gain = improvement_without_msg - improvement_with_msg
    
    # 如果消息没有改变决策，惩罚发送者
    if torch.allclose(action_with_message, action_without_message, atol=1e-3):
        sender_reward = -0.5 * message_cost
    else:
        # 奖励与改进增益成正比
        sender_reward = 0.1 * improvement_gain.item() - 0.2 * message_cost
    
    # 总奖励 = 基础奖励 + 发送者奖励
    total_reward = baseline_reward + sender_reward
    
    return sender_reward, total_reward

class CounterfactualRewardCalculator:
    """管理反事实奖励计算"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.message_cost = config.get("message_cost", 0.05)
        self.improvement_weight = config.get("improvement_weight", 0.1)
        self.cost_weight = config.get("cost_weight", 0.2)
        self.jit_enabled = config.get("jit_enabled", True)
        
        # JIT编译函数
        if self.jit_enabled:
            self.calculate = torch.jit.script(self._calculate_impl)
        else:
            self.calculate = self._calculate_impl
    
    def _calculate_impl(self, 
                       action_with_message: torch.Tensor,
                       action_without_message: torch.Tensor,
                       optimal_action: torch.Tensor,
                       baseline_reward: float) -> Tuple[float, float]:
        """计算反事实奖励（内部实现）"""
        # 计算动作改进
        improvement_with_msg = torch.norm(action_with_message - optimal_action)
        improvement_without_msg = torch.norm(action_without_message - optimal_action)
        
        # 计算消息带来的改进增益
        improvement_gain = improvement_without_msg - improvement_with_msg
        
        # 如果消息没有改变决策，惩罚发送者
        if torch.allclose(action_with_message, action_without_message, atol=1e-3):
            sender_reward = -0.5 * self.message_cost
        else:
            # 奖励与改进增益成正比
            sender_reward = self.improvement_weight * improvement_gain.item() - self.cost_weight * self.message_cost
        
        # 总奖励 = 基础奖励 + 发送者奖励
        total_reward = baseline_reward + sender_reward
        
        return sender_reward, total_reward
    
    def batch_calculate(self,
                       actions_with_message: List[torch.Tensor],
                       actions_without_message: List[torch.Tensor],
                       optimal_actions: List[torch.Tensor],
                       baseline_rewards: List[float]) -> Tuple[List[float], List[float]]:
        """批量计算反事实奖励"""
        sender_rewards = []
        total_rewards = []
        
        for awm, awom, oa, br in zip(actions_with_message, actions_without_message, optimal_actions, baseline_rewards):
            sr, tr = self.calculate(awm, awom, oa, br)
            sender_rewards.append(sr)
            total_rewards.append(tr)
        
        return sender_rewards, total_rewards