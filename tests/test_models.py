import unittest
import torch
from models import AgentPolicy, QLoRAWrapper, quantize

class TestModels(unittest.TestCase):
    def setUp(self):
        self.obs_dim = 30
        self.action_dim = 5
        self.config = {
            'quantize': True,
            'sparsity': 0.7,
            'block_size': 64
        }
    
    def test_agent_policy_forward(self):
        """测试策略网络前向传播"""
        policy = AgentPolicy(self.obs_dim, self.action_dim, self.config)
        
        # 无消息输入
        obs = torch.randn(1, self.obs_dim)
        logits = policy(obs)
        self.assertEqual(logits.shape, (1, self.action_dim))
        
        # 有消息输入
        messages = torch.randn(2, 1, 256)  # 2条消息，每条256维
        logits = policy(obs, messages)
        self.assertEqual(logits.shape, (1, self.action_dim))
    
    def test_quantization(self):
        """测试模型量化"""
        # 创建测试模型
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 5)
        )
        
        # 保存原始权重
        original_weight = model[0].weight.data.clone()
        
        # 应用量化
        quantized_model = quantize.quantize_model_int4(model)
        
        # 验证前向传播
        x = torch.randn(3, 10)
        output = quantized_model(x)
        self.assertEqual(output.shape, (3, 5))
        
        # 验证权重是否被替换（量化）
        self.assertIsNot(model[0].weight, original_weight)
    
    def test_qlora_wrapper(self):
        """测试QLoRA微调适配器"""
        base_model = AgentPolicy(self.obs_dim, self.action_dim, self.config)
        qlora_model = QLoRAWrapper(base_model, lora_rank=4)
        
        # 验证参数数量
        base_params = sum(p.numel() for p in base_model.parameters())
        qlora_params = sum(p.numel() for p in qlora_model.parameters())
        self.assertGreater(qlora_params, base_params)
        
        # 验证前向传播
        obs = torch.randn(1, self.obs_dim)
        logits = qlora_model(obs)
        self.assertEqual(logits.shape, (1, self.action_dim))
        
        # 验证LoRA适配器存在
        self.assertTrue(any("lora_adapters" in name for name, _ in qlora_model.named_modules()))

if __name__ == '__main__':
    unittest.main()