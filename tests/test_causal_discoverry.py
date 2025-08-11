import unittest
import numpy as np
import torch
from causal_discovery import fci_pc, nsa_attention
from causal_discovery.pc_algorithm import PCAlgorithm

class TestCausalDiscovery(unittest.TestCase):
    def setUp(self):
        # 创建模拟数据
        np.random.seed(42)
        self.data = np.random.randn(100, 5)  # 100个样本，5个变量
        self.config = {
            'max_vars': 5,
            'alpha': 0.05,
            'n_jobs': 4
        }
    
    def test_pc_algorithm_basic(self):
        """测试PC算法基本功能"""
        pc = PCAlgorithm(self.config)
        graph = pc.fit(self.data)
        
        # 验证图结构
        self.assertEqual(graph.shape, (5, 5))
        self.assertTrue((graph >= 0).all())
        self.assertTrue((graph <= 1).all())
        
        # 验证对称性
        for i in range(5):
            for j in range(i+1, 5):
                self.assertEqual(graph[i, j], graph[j, i])
    
    def test_fci_pc_optimization(self):
        """测试FCI-PC优化算法"""
        fci = fci_pc.FCIPCAlgorithm(max_vars=3, alpha=0.01, n_jobs=2)
        graph = fci.fit(self.data)
        
        # 验证维度压缩
        self.assertEqual(graph.shape, (3, 3))
        
        # 验证结果一致性
        pc = PCAlgorithm({'max_vars': 3, 'alpha': 0.01})
        expected = pc.fit(self.data[:, :3])
        np.testing.assert_array_equal(graph, expected)
    
    def test_nsa_attention(self):
        """测试NSA注意力机制"""
        nsa = nsa_attention.NativeSparseAttention(dim=64, sparsity=0.7)
        
        # 创建测试输入
        x = torch.randn(2, 10, 64)  # 批大小2，序列长度10，维度64
        
        # 前向传播
        output = nsa(x)
        
        # 验证输出形状
        self.assertEqual(output.shape, (2, 10, 64))
        
        # 验证稀疏性
        attn_mask = nsa.generate_sparse_mask(x, x)
        sparsity = 1 - attn_mask.sum() / attn_mask.numel()
        self.assertAlmostEqual(sparsity, 0.7, delta=0.05)
    
    def test_nsa_cuda_kernel(self):
        """测试NSA CUDA内核（如果可用）"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        nsa = nsa_attention.NativeSparseAttention(dim=64, sparsity=0.7).cuda()
        x = torch.randn(2, 10, 64).cuda()
        
        # 前向传播
        output = nsa(x)
        self.assertEqual(output.device.type, 'cuda')
        self.assertEqual(output.shape, (2, 10, 64))

if __name__ == '__main__':
    unittest.main()