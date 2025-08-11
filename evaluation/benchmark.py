import torch
import numpy as np
from models import AgentPolicy
from evaluation.evaluator import CausalMACEvaluator

class BaselineModels:
    """基准模型实现（IC3Net和TarMAC）"""
    
    @staticmethod
    def load_ic3net(env_config, model_path=None):
        """加载IC3Net模型"""
        # 简化实现 - 实际中应从文件加载
        model = AgentPolicy(
            obs_dim=env_config['obs_dim'],
            action_dim=env_config['action_dim'],
            config={
                'quantize': False,
                'sparsity': 0.0,  # 无稀疏注意力
                'block_size': 64
            }
        )
        
        if model_path:
            model.load_state_dict(torch.load(model_path))
        
        return model
    
    @staticmethod
    def load_tarmac(env_config, model_path=None):
        """加载TarMAC模型"""
        # 简化实现 - 实际中应从文件加载
        model = AgentPolicy(
            obs_dim=env_config['obs_dim'],
            action_dim=env_config['action_dim'],
            config={
                'quantize': False,
                'sparsity': 0.0,  # 无稀疏注意力
                'block_size': 64,
                'use_tarmac': True  # 特殊标志
            }
        )
        
        if model_path:
            model.load_state_dict(torch.load(model_path))
        
        return model

class BenchmarkComparator:
    """基准方法对比工具"""
    def __init__(self, causal_mac_model, env_config, model_config, device='cuda'):
        self.causal_mac = causal_mac_model
        self.env_config = env_config
        self.model_config = model_config
        self.device = device
        
        # 加载基准模型
        self.ic3net = BaselineModels.load_ic3net(env_config)
        self.tarmac = BaselineModels.load_tarmac(env_config)
    
    def run_comparison(self, num_episodes=50):
        """运行完整对比评估"""
        results = {}
        
        # 评估Causal-MAC
        causal_evaluator = CausalMACEvaluator(self.causal_mac, self.env_config, self.model_config, self.device)
        results['causal_mac'] = causal_evaluator.evaluate(num_episodes)
        
        # 评估IC3Net
        ic3net_evaluator = CausalMACEvaluator(self.ic3net, self.env_config, self.model_config, self.device)
        results['ic3net'] = ic3net_evaluator.evaluate(num_episodes)
        
        # 评估TarMAC
        tarmac_evaluator = CausalMACEvaluator(self.tarmac, self.env_config, self.model_config, self.device)
        results['tarmac'] = tarmac_evaluator.evaluate(num_episodes)
        
        return results
    
    def generate_comparison_report(self, results):
        """生成对比报告"""
        report = "===== Multi-Agent Communication Benchmark Comparison =====\n\n"
        report += "| Metric                | Causal-MAC | IC3Net     | TarMAC     |\n"
        report += "|-----------------------|------------|------------|------------|\n"
        
        # 成功率对比
        report += self._format_row("Success Rate", results, lambda r: f"{r['success_rate']:.2%}")
        
        # 平均奖励
        report += self._format_row("Avg Reward", results, lambda r: f"{r['avg_reward']:.2f}")
        
        # 通信效率
        report += self._format_row("Invalid Msg Ratio", results, lambda r: f"{r['invalid_message_ratio']:.2%}")
        
        # 资源使用
        if 'gpu_memory_usage' in results['causal_mac']:
            report += self._format_row("GPU Mem (MB)", results, lambda r: f"{r['gpu_memory_usage']:.1f}")
        
        # 推理延迟
        report += self._format_row("Latency (ms)", results, lambda r: f"{r['inference_latency']:.2f}")
        
        return report
    
    def _format_row(self, metric_name, results, formatter):
        """格式化报告行"""
        row = f"| {metric_name:<21} | "
        row += formatter(results['causal_mac']) + " | "
        row += formatter(results['ic3net']) + " | "
        row += formatter(results['tarmac']) + " |\n"
        return row
    
    def plot_performance_comparison(self, results, save_path=None):
        """绘制性能对比图（简化实现）"""
        import matplotlib.pyplot as plt
        
        metrics = ['success_rate', 'avg_reward', 'invalid_message_ratio']
        labels = ['Success Rate', 'Average Reward', 'Invalid Message Ratio']
        
        causal_vals = [results['causal_mac'][m] for m in metrics]
        ic3net_vals = [results['ic3net'][m] for m in metrics]
        tarmac_vals = [results['tarmac'][m] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width, causal_vals, width, label='Causal-MAC')
        ax.bar(x, ic3net_vals, width, label='IC3Net')
        ax.bar(x + width, tarmac_vals, width, label='TarMAC')
        
        ax.set_ylabel('Performance')
        ax.set_title('Communication Model Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        
        # 调整无效消息比例显示
        ax.set_yscale('log' if any(v > 1 for v in ic3net_vals + tarmac_vals) else 'linear')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        return fig