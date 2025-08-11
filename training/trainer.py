import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
from tqdm import tqdm
from models import AgentPolicy, QLoRAWrapper
from communication.scheduler import CausalScheduler
from environments.custom_pursuit_env import CustomPursuitEnv
from training.replay_buffer import PrioritizedReplayBuffer
from training.utils import hard_update, soft_update
from configs import load_config

class CausalMACTrainer:
    """Causal-MAC训练框架"""
    def __init__(self, config, env_config, model_config):
        # 加载配置
        self.config = config
        self.env_config = env_config
        self.model_config = model_config
        
        # 初始化环境
        self.env = CustomPursuitEnv(env_config=env_config)
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        
        # 初始化模型
        self.policy = AgentPolicy(self.obs_dim, self.action_dim, model_config)
        self.target_policy = AgentPolicy(self.obs_dim, self.action_dim, model_config)
        hard_update(self.target_policy, self.policy)  # 硬更新目标网络
        
        # 通信调度器
        self.scheduler = CausalScheduler(model_config['scheduler']['causal_graph_path'])
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.policy.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
        
        # 使用QLoRA微调
        if config.get('use_qlora', True):
            self.policy = QLoRAWrapper(self.policy, lora_rank=config['lora_rank'])
            self.target_policy = QLoRAWrapper(self.target_policy, lora_rank=config['lora_rank'])
        
        # 经验回放池
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=config['replay_capacity'],
            alpha=config['replay_alpha'],
            beta=config['replay_beta'],
            beta_increment=config['replay_beta_increment']
        )
        
        # 训练状态
        self.total_steps = 0
        self.episode_rewards = []
        self.best_reward = -float('inf')
        
        # 移动到设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)
        self.target_policy.to(self.device)
    
    def run(self):
        """主训练循环"""
        start_time = time.time()
        
        for episode in tqdm(range(self.config['num_episodes'])):
            # 重置环境
            obs, _ = self.env.reset()
            episode_reward = {agent: 0 for agent in self.env.agents}
            messages = {agent: [] for agent in self.env.agents}
            
            # 运行一个回合
            for step in range(self.config['max_steps']):
                # 收集动作和消息
                actions = {}
                new_messages = {}
                
                for agent in self.env.agents:
                    # 准备输入
                    agent_obs = torch.tensor(obs[agent], dtype=torch.float32).unsqueeze(0).to(self.device)
                    
                    # 获取消息
                    received_msgs = []
                    for sender, msg_list in messages.items():
                        if sender != agent and len(msg_list) > step:
                            msg = msg_list[step]
                            should_send, msg_tensor = self.scheduler(msg, sender, agent)
                            if should_send:
                                received_msgs.append(msg_tensor)
                    
                    # 转换为张量
                    if received_msgs:
                        received_msgs = torch.stack(received_msgs).unsqueeze(0).to(self.device)
                    else:
                        received_msgs = None
                    
                    # 策略网络预测动作
                    with torch.no_grad():
                        action_logits = self.policy(agent_obs, received_msgs)
                        action = torch.argmax(action_logits, dim=-1).item()
                    
                    actions[agent] = action
                    
                    # 生成要发送的消息（当前观测）
                    new_messages[agent] = agent_obs.cpu().squeeze(0).numpy()
                
                # 更新消息缓冲区
                for agent, msg in new_messages.items():
                    messages[agent].append(msg)
                
                # 环境执行一步
                next_obs, rewards, dones, truncs, infos = self.env.step(actions)
                
                # 存储经验
                experience = {
                    'states': obs,
                    'actions': actions,
                    'rewards': rewards,
                    'next_states': next_obs,
                    'dones': dones
                }
                self.replay_buffer.add(**experience)
                
                # 更新状态
                obs = next_obs
                self.total_steps += 1
                
                # 更新奖励
                for agent in rewards:
                    episode_reward[agent] += rewards[agent]
                
                # 定期更新模型
                if self.total_steps % self.config['update_interval'] == 0:
                    self._update_model()
                
                # 定期更新目标网络
                if self.total_steps % self.config['target_update_interval'] == 0:
                    if self.config['target_update'] == 'hard':
                        hard_update(self.target_policy, self.policy)
                    else:
                        soft_update(self.target_policy, self.policy, self.config['tau'])
                
                # 检查是否结束
                if all(dones.values()):
                    break
            
            # 记录回合奖励
            avg_reward = sum(episode_reward.values()) / len(episode_reward)
            self.episode_rewards.append(avg_reward)
            
            # 定期保存模型
            if avg_reward > self.best_reward:
                self.best_reward = avg_reward
                self.save(f"checkpoints/best_model_{episode}.pt")
            
            # 打印进度
            if episode % self.config['log_interval'] == 0:
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Steps: {self.total_steps}")
        
        # 训练结束
        print(f"Training completed in {time.time()-start_time:.2f} seconds")
        self.save("checkpoints/final_model.pt")
        return self.episode_rewards
    
    def _update_model(self):
        """从回放池采样并更新模型"""
        if len(self.replay_buffer) < self.config['batch_size']:
            return
        
        # 采样批次
        batch = self.replay_buffer.sample(self.config['batch_size'])
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        weights = batch['weights']
        indices = batch['indices']
        
        # 转换为张量
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        
        # 计算Q值
        q_values = self._compute_q_values(states, actions)
        
        # 计算目标Q值
        with torch.no_grad():
            target_q_values = self._compute_target_q_values(next_states, rewards, dones)
        
        # 计算损失
        loss = F.mse_loss(q_values, target_q_values, reduction='none')
        loss = (loss * weights).mean()
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config['max_grad_norm'])
        
        # 优化步骤
        self.optimizer.step()
        
        # 更新优先级
        priorities = loss.detach().cpu().numpy() + 1e-6
        self.replay_buffer.update_priorities(indices, priorities)
        
        return loss.item()
    
    def _compute_q_values(self, states, actions):
        """计算当前Q值"""
        q_values = []
        for i in range(len(states)):
            state = states[i]
            action = actions[i]
            
            agent_q = []
            for agent in self.env.agents:
                # 准备输入
                agent_obs = torch.tensor(state[agent], dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # 获取Q值
                q_logits = self.policy(agent_obs)
                agent_q.append(q_logits[0, action[agent]])
            
            q_values.append(torch.stack(agent_q).mean())
        
        return torch.stack(q_values)
    
    def _compute_target_q_values(self, next_states, rewards, dones):
        """计算目标Q值"""
        target_q = []
        for i in range(len(next_states)):
            next_state = next_states[i]
            reward = rewards[i]
            done = dones[i]
            
            agent_target = []
            for agent in self.env.agents:
                # 准备输入
                next_obs = torch.tensor(next_state[agent], dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # 获取目标Q值
                with torch.no_grad():
                    next_q_logits = self.target_policy(next_obs)
                    max_next_q = next_q_logits.max(dim=-1)[0]
                
                # 计算目标值
                r = reward[agent]
                d = done[agent]
                target = r + (1 - d) * self.config['gamma'] * max_next_q.item()
                agent_target.append(target)
            
            target_q.append(torch.tensor(np.mean(agent_target), dtype=torch.float32))
        
        return torch.stack(target_q).to(self.device)
    
    def save(self, path):
        """保存模型"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'rewards': self.episode_rewards
        }, path)
    
    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = checkpoint.get('rewards', [])