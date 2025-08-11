# environments/base_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import ParallelEnv
from typing import Dict, List, Tuple, Optional, Any
from configs import loader

class BaseMultiAgentEnv(ParallelEnv):
    """多智能体环境基类，提供标准接口和通用功能"""
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'name': "base_multi_agent_v1"}
    
    def __init__(self, env_config: str = "pursuit_v4", render_mode: Optional[str] = None):
        """
        初始化基础多智能体环境
        
        参数:
            env_config: 环境配置名称 (对应configs/env中的yaml文件)
            render_mode: 渲染模式 (None, 'human', 'rgb_array')
        """
        # 加载配置
        self.config = loader.get_env_config(env_config)["environment"]
        
        # 设置渲染模式
        self.render_mode = render_mode
        
        # 环境参数
        self.n_agents = self.config.get("n_agents", 4)
        self.max_steps = self.config.get("max_cycles", 1000)
        self.current_step = 0
        
        # 加载特征和因果图
        self._load_features_and_causal_graph()
        
        # 初始化智能体
        self.agents = [f"agent_{i}" for i in range(self.n_agents)]
        self.possible_agents = self.agents.copy()
        
        # 动作空间和观测空间
        self._setup_action_space()
        self._setup_observation_space()
        
        # 动态障碍物
        self.dynamic_obstacles = []
        if self.config.get("advanced", {}).get("enable_dynamic_obstacles", False):
            self._init_dynamic_obstacles()
    
    def _load_features_and_causal_graph(self):
        """从预处理的数据中加载特征和因果图"""
        from configs import loader
        feature_config = self.config["observation"]
        
        # 加载特征缓存
        feature_cache_path = f"data/processed/feature_cache/{self.config['name']}_features.npz"
        self.features = np.load(feature_cache_path, allow_pickle=True)
        
        # 加载因果图
        causal_graph_path = f"data/processed/causal_graphs/{self.config['name']}_causal_graph.npz"
        causal_data = np.load(causal_graph_path)
        self.causal_graph = causal_data["graph"]
        
        # 设置特征维度
        self.feature_dim = feature_config["feature_dim"]
        
        # 获取场景列表
        self.scenes = list(self.features.keys())
        self.current_scene_idx = 0
        self.current_scene_features = self.features[self.scenes[self.current_scene_idx]]
        self.scene_step = 0
    
    def _setup_action_space(self):
        """设置动作空间"""
        # 从配置中获取动作空间维度
        action_dim = self.config.get("action_dim", 5)
        
        # 创建动作空间
        self.action_spaces = {
            agent: spaces.Discrete(action_dim) for agent in self.agents
        }
    
    def _setup_observation_space(self):
        """设置观测空间"""
        # 创建观测空间
        self.observation_spaces = {
            agent: spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(self.feature_dim,), 
                dtype=np.float32
            ) for agent in self.agents
        }
    
    def _init_dynamic_obstacles(self):
        """初始化动态障碍物"""
        obstacle_config = self.config["dynamic_obstacles"]
        n_obstacles = obstacle_config.get("num_obstacles", 5)
        
        # 初始化障碍物位置和速度
        for _ in range(n_obstacles):
            pos = np.random.uniform(-10, 10, size=2)
            speed = np.random.uniform(0.1, 0.5)
            direction = np.random.uniform(0, 2*np.pi)
            velocity = np.array([np.cos(direction), np.sin(direction)]) * speed
            
            self.dynamic_obstacles.append({
                "position": pos,
                "velocity": velocity,
                "size": np.random.uniform(0.5, 2.0)
            })
    
    def _update_dynamic_obstacles(self):
        """更新动态障碍物位置"""
        if not self.dynamic_obstacles:
            return
        
        # 更新障碍物位置
        for obstacle in self.dynamic_obstacles:
            obstacle["position"] += obstacle["velocity"]
            
            # 边界反弹
            if np.abs(obstacle["position"][0]) > 10:
                obstacle["velocity"][0] *= -1
            if np.abs(obstacle["position"][1]) > 10:
                obstacle["velocity"][1] *= -1
    
    def _get_agent_observation(self, agent_id: str) -> np.ndarray:
        """获取智能体观测"""
        # 基础实现 - 子类应覆盖此方法
        return np.zeros(self.feature_dim, dtype=np.float32)
    
    def _calculate_reward(self, agent_id: str) -> float:
        """计算智能体奖励"""
        # 基础实现 - 子类应覆盖此方法
        return 0.0
    
    def _get_done_flags(self) -> Dict[str, bool]:
        """获取完成标志"""
        # 检查是否达到最大步数
        done = self.current_step >= self.max_steps
        return {agent: done for agent in self.agents}
    
    def _get_info(self, agent_id: str) -> Dict[str, Any]:
        """获取额外信息"""
        return {}
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """重置环境"""
        super().reset(seed=seed)
        
        # 重置状态
        self.current_step = 0
        self.current_scene_idx = (self.current_scene_idx + 1) % len(self.scenes)
        self.current_scene_features = self.features[self.scenes[self.current_scene_idx]]
        self.scene_step = 0
        
        # 重置障碍物
        if self.dynamic_obstacles:
            self._init_dynamic_obstacles()
        
        # 获取初始观测
        observations = {
            agent: self._get_agent_observation(agent) for agent in self.agents
        }
        
        # 获取信息
        infos = {agent: self._get_info(agent) for agent in self.agents}
        
        return observations, infos
    
    def step(self, actions: Dict[str, int]):
        """执行一步环境更新"""
        self.current_step += 1
        self.scene_step = (self.scene_step + 1) % len(self.current_scene_features)
        
        # 更新动态障碍物
        if self.dynamic_obstacles:
            self._update_dynamic_obstacles()
        
        # 计算奖励
        rewards = {
            agent: self._calculate_reward(agent) for agent in self.agents
        }
        
        # 获取观测
        observations = {
            agent: self._get_agent_observation(agent) for agent in self.agents
        }
        
        # 检查是否结束
        dones = self._get_done_flags()
        terminations = {agent: dones[agent] for agent in self.agents}
        truncations = {agent: False for agent in self.agents}  # 目前没有提前终止
        
        # 获取信息
        infos = {agent: self._get_info(agent) for agent in self.agents}
        
        return observations, rewards, terminations, truncations, infos
    
    def render(self):
        """渲染环境"""
        if self.render_mode is None:
            return
        
        # 基础渲染实现 - 子类应覆盖此方法
        if self.render_mode == "human":
            print(f"Step: {self.current_step}/{self.max_steps}")
        elif self.render_mode == "rgb_array":
            return np.zeros((100, 100, 3), dtype=np.uint8)
    
    def observation_space(self, agent):
        return self.observation_spaces[agent]
    
    def action_space(self, agent):
        return self.action_spaces[agent]
    
    def close(self):
        """关闭环境"""
        pass