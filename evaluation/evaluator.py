import torch
import numpy as np
import time
from tqdm import tqdm
from collections import defaultdict
from environments.custom_pursuit_env import CustomPursuitEnv
from models import AgentPolicy
from communication.scheduler import CausalScheduler
from configs import load_config

class CausalMACEvaluator:
    """Causal-MAC模型评估器"""
    def __init__(self, policy_model, env_config, model_config, device='cuda'):
        """
        Args:
            policy_model: 训练好的策略模型
            env_config: 环境配置
            model_config: 模型配置
            device: 计算设备
        """
        self.policy = policy_model
        self.env = CustomPursuitEnv(env_config=env_config)
        self.scheduler = CausalScheduler(model_config['scheduler']['causal_graph_path'])
        self.device = device
        self.policy.to(device)
        self.policy.eval()  # 设置为评估模式
        
        # 评估指标存储
        self.metrics = {
            'success_rate': 0.0,
            'avg_reward': 0.0,
            'steps_per_episode': 0.0,
            'message_count': defaultdict(int),
            'invalid_message_ratio': 0.0,
            'gpu_memory_usage': 0.0,
            'inference_latency': 0.0
        }
    
    def evaluate(self, num_episodes=50):
        """执行完整评估流程"""
        total_rewards = []
        success_count = 0
        total_steps = 0
        total_messages = 0
        invalid_messages = 0
        latency_sum = 0
        inference_count = 0
        
        # 记录初始显存使用
        initial_mem = torch.cuda.memory_allocated(self.device) if self.device == 'cuda' else 0
        
        for episode in tqdm(range(num_episodes), desc="Evaluating"):
            obs, _ = self.env.reset()
            episode_reward = {agent: 0 for agent in self.env.agents}
            messages = {agent: [] for agent in self.env.agents}
            done = False
            step_count = 0
            
            while not done:
                actions = {}
                new_messages = {}
                step_start = time.time()
                
                for agent in self.env.agents:
                    # 准备输入
                    agent_obs = torch.tensor(obs[agent], dtype=torch.float32).unsqueeze(0).to(self.device)
                    
                    # 获取消息
                    received_msgs = []
                    for sender, msg_list in messages.items():
                        if sender != agent and len(msg_list) > step_count:
                            msg = msg_list[step_count]
                            should_send, msg_tensor = self.scheduler(msg, sender, agent)
                            
                            # 统计消息
                            total_messages += 1
                            if not should_send:
                                invalid_messages += 1
                            
                            if should_send:
                                received_msgs.append(msg_tensor)
                    
                    # 策略网络预测动作
                    with torch.no_grad():
                        action_logits = self.policy(agent_obs, 
                                                   torch.stack(received_msgs).unsqueeze(0) if received_msgs else None)
                        action = torch.argmax(action_logits, dim=-1).item()
                    
                    actions[agent] = action
                    
                    # 生成要发送的消息
                    new_messages[agent] = agent_obs.cpu().squeeze(0).numpy()
                
                # 更新消息缓冲区
                for agent, msg in new_messages.items():
                    messages[agent].append(msg)
                
                # 环境执行一步
                next_obs, rewards, dones, truncs, infos = self.env.step(actions)
                
                # 更新状态
                obs = next_obs
                step_count += 1
                
                # 更新奖励
                for agent in rewards:
                    episode_reward[agent] += rewards[agent]
                
                # 记录延迟
                latency_sum += time.time() - step_start
                inference_count += len(self.env.agents)
                
                # 检查是否结束
                done = all(dones.values())
            
            # 记录回合指标
            avg_ep_reward = sum(episode_reward.values()) / len(episode_reward)
            total_rewards.append(avg_ep_reward)
            
            # 检查是否成功（捕获所有猎物）
            if not self.env.prey_positions:
                success_count += 1
            
            total_steps += step_count
        
        # 计算最终指标
        self.metrics['success_rate'] = success_count / num_episodes
        self.metrics['avg_reward'] = np.mean(total_rewards)
        self.metrics['steps_per_episode'] = total_steps / num_episodes
        self.metrics['invalid_message_ratio'] = invalid_messages / total_messages if total_messages > 0 else 0
        self.metrics['inference_latency'] = (latency_sum / inference_count) * 1000  # 转换为毫秒
        
        # 计算显存使用
        if self.device == 'cuda':
            final_mem = torch.cuda.memory_allocated(self.device)
            self.metrics['gpu_memory_usage'] = (final_mem - initial_mem) / (1024 ** 2)  # MB
        
        return self.metrics
    
    def compare_with_baseline(self, baseline_model, num_episodes=50):
        """与基准模型对比"""
        baseline_evaluator = CausalMACEvaluator(baseline_model, self.env.config, self.model_config, self.device)
        baseline_metrics = baseline_evaluator.evaluate(num_episodes)
        
        comparison = {
            'model': self.metrics,
            'baseline': baseline_metrics,
            'improvement': {}
        }
        
        # 计算改进百分比
        for key in self.metrics:
            if key in baseline_metrics:
                base_val = baseline_metrics[key]
                model_val = self.metrics[key]
                
                # 根据指标类型计算改进
                if 'rate' in key or 'ratio' in key or 'usage' in key:
                    improvement = (model_val - base_val) / base_val * 100
                else:
                    improvement = model_val - base_val
                
                comparison['improvement'][key] = improvement
        
        return comparison
    
    def generate_report(self, comparison=None):
        """生成评估报告"""
        report = "===== Causal-MAC Evaluation Report =====\n"
        report += f"Success Rate: {self.metrics['success_rate']:.2%}\n"
        report += f"Average Reward: {self.metrics['avg_reward']:.2f}\n"
        report += f"Steps per Episode: {self.metrics['steps_per_episode']:.1f}\n"
        report += f"Invalid Message Ratio: {self.metrics['invalid_message_ratio']:.2%}\n"
        
        if 'gpu_memory_usage' in self.metrics:
            report += f"GPU Memory Usage: {self.metrics['gpu_memory_usage']:.2f} MB\n"
        
        report += f"Inference Latency: {self.metrics['inference_latency']:.2f} ms/step\n"
        
        if comparison:
            report += "\n===== Baseline Comparison =====\n"
            for metric, imp in comparison['improvement'].items():
                if 'rate' in metric or 'ratio' in metric:
                    report += f"{metric}: {imp:+.2f}%\n"
                else:
                    report += f"{metric}: {imp:+.2f}\n"
        
        return report