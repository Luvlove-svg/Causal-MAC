import unittest
import numpy as np
import torch
from communication import scheduler, reward_calculator

class TestCommunication(unittest.TestCase):
    def setUp(self):
        # 创建模拟因果图
        self.causal_graph = np.array([
            [0.0, 0.8, 0.2],
            [0.7, 0.0, 0.9],
            [0.1, 0.6, 0.0]
        ])
        np.savez("test_causal_graph.npz", graph=self.causal_graph)
        
    def test_causal_scheduler(self):
        """测试因果调度器决策逻辑"""
        sched = scheduler.CausalScheduler("test_causal_graph.npz", threshold=0.6)
        
        # 测试有效通信 (agent0 → agent1)
        obs = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        should_send, message = sched(obs, 0, 1)
        self.assertTrue(should_send)
        self.assertIsNotNone(message)
        
        # 测试无效通信 (agent0 → agent2)
        should_send, message = sched(obs, 0, 2)
        self.assertFalse(should_send)
        self.assertIsNone(message)
    
    def test_lite_scheduler(self):
        """测试轻量级调度器"""
        lite_sched = scheduler.LiteScheduler("test_causal_graph.npz", threshold=0.6)
        
        # 测试缓存功能
        self.assertIn((0, 1), lite_sched.mask_cache)
        self.assertTrue(lite_sched.mask_cache[(0, 1)])
        self.assertFalse(lite_sched.mask_cache[(0, 2)])
        
        # 测试前向传播
        obs = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        should_send, message = lite_sched(obs, 1, 2)
        self.assertTrue(should_send)
        self.assertEqual(message.tolist(), [[2.0, 3.0]])  # 基于因果图的稀疏激活
    
    def test_counterfactual_reward(self):
        """测试反事实奖励计算"""
        calc = reward_calculator.CounterfactualRewardCalculator()
        
        # 消息没有改变决策
        receiver_action = torch.tensor([0.5, 0.5])
        counterfactual_action = torch.tensor([0.5, 0.5])
        reward = calc(0, 1, None, receiver_action, counterfactual_action)
        self.assertAlmostEqual(reward.item(), -0.2, delta=1e-6)
        
        # 消息改变了决策
        counterfactual_action = torch.tensor([0.8, 0.2])
        reward = calc(0, 1, None, receiver_action, counterfactual_action)
        self.assertAlmostEqual(reward.item(), 0.0, delta=1e-6)
    
    def test_interaction_confidence(self):
        """测试交互置信度奖励"""
        conf_attn = reward_calculator.InteractionConfidenceReward(dim=16)
        
        states = torch.randn(2, 4, 8)  # 批大小2，4个智能体，状态8维
        actions = torch.randn(2, 4, 8)  # 动作8维
        
        confidence = conf_attn(states, actions)
        self.assertEqual(confidence.shape, (2, 4))
        self.assertTrue((confidence >= 0).all())
        self.assertTrue((confidence <= 1).all())

if __name__ == '__main__':
    unittest.main()