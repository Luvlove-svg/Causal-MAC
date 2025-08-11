# environments/pursuit_env.py
import numpy as np
from pettingzoo.sisl import pursuit_v4
from .base_env import BaseMultiAgentEnv
from configs import loader

class PursuitEnv(BaseMultiAgentEnv):
    """追捕环境封装，基于PettingZoo的pursuit_v4"""
    
    def __init__(self, env_config: str = "pursuit_v4", render_mode: Optional[str] = None):
        super().__init__(env_config, render_mode)
        
        # 创建底层PettingZoo环境
        self.env = pursuit_v4.parallel_env(
            n_pursuers=self.n_agents,
            max_cycles=self.max_steps,
            render_mode=render_mode
        )
        
        # 重置环境以初始化
        self.env.reset()
    
    def _get_agent_observation(self, agent_id: str) -> np.ndarray:
        """获取智能体观测"""
        # 从底层环境获取观测
        raw_obs = self.env.observe(agent_id)
        
        # 使用当前场景特征增强观测
        scene_feature = self.current_scene_features[self.scene_step]
        
        # 组合观测 (原始观测 + NuScenes特征)
        # 注意: 原始观测已经是向量，我们将其与场景特征连接
        combined_obs = np.concatenate([raw_obs.flatten(), scene_feature])
        
        # 如果超过特征维度，进行截断
        if len(combined_obs) > self.feature_dim:
            combined_obs = combined_obs[:self.feature_dim]
        
        return combined_obs
    
    def _calculate_reward(self, agent_id: str) -> float:
        """计算智能体奖励"""
        # 从底层环境获取奖励
        # 注意: 在step方法中统一计算奖励
        return 0.0
    
    def step(self, actions: Dict[str, int]):
        """执行一步环境更新"""
        # 在底层环境执行动作
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        
        # 更新内部状态
        self.current_step += 1
        self.scene_step = (self.scene_step + 1) % len(self.current_scene_features)
        
        # 更新动态障碍物
        if self.dynamic_obstacles:
            self._update_dynamic_obstacles()
        
        # 增强观测
        enhanced_observations = {}
        for agent, obs in observations.items():
            scene_feature = self.current_scene_features[self.scene_step]
            combined_obs = np.concatenate([obs.flatten(), scene_feature])
            if len(combined_obs) > self.feature_dim:
                combined_obs = combined_obs[:self.feature_dim]
            enhanced_observations[agent] = combined_obs
        
        # 增强奖励 (可选)
        # 这里可以添加基于因果图的额外奖励
        
        # 增强信息
        enhanced_infos = {}
        for agent in self.agents:
            info = infos.get(agent, {})
            info["causal_graph"] = self.causal_graph
            info["scene"] = self.scenes[self.current_scene_idx]
            info["scene_step"] = self.scene_step
            enhanced_infos[agent] = info
        
        # 检查是否结束
        dones = self._get_done_flags()
        terminations = {agent: dones[agent] or terminations[agent] for agent in self.agents}
        
        return enhanced_observations, rewards, terminations, truncations, enhanced_infos
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """重置环境"""
        # 重置底层环境
        observations, infos = self.env.reset(seed=seed, options=options)
        
        # 重置父类状态
        super().reset(seed=seed, options=options)
        
        # 增强观测
        enhanced_observations = {}
        for agent, obs in observations.items():
            scene_feature = self.current_scene_features[self.scene_step]
            combined_obs = np.concatenate([obs.flatten(), scene_feature])
            if len(combined_obs) > self.feature_dim:
                combined_obs = combined_obs[:self.feature_dim]
            enhanced_observations[agent] = combined_obs
        
        # 增强信息
        enhanced_infos = {}
        for agent in self.agents:
            info = infos.get(agent, {})
            info["causal_graph"] = self.causal_graph
            info["scene"] = self.scenes[self.current_scene_idx]
            info["scene_step"] = self.scene_step
            enhanced_infos[agent] = info
        
        return enhanced_observations, enhanced_infos
    
    def render(self):
        """渲染环境"""
        return self.env.render()
    
    def close(self):
        """关闭环境"""
        self.env.close()