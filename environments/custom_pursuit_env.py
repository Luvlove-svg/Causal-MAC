# environments/custom_pursuit_env.py
import numpy as np
from .base_env import BaseMultiAgentEnv
from .custom_maps import CustomMapGenerator
from configs import loader

class CustomPursuitEnv(BaseMultiAgentEnv):
    """自定义追捕环境，支持动态障碍物和复杂地图"""
    
    def __init__(self, env_config: str = "custom_map", render_mode: Optional[str] = None):
        super().__init__(env_config, render_mode)
        
        # 创建自定义地图生成器
        self.map_generator = CustomMapGenerator(self.config)
        
        # 初始化智能体位置
        self.agent_positions = self._init_agent_positions()
        
        # 初始化猎物位置
        self.prey_positions = self._init_prey_positions()
    
    def _init_agent_positions(self) -> Dict[str, np.ndarray]:
        """初始化智能体位置"""
        positions = {}
        for i, agent in enumerate(self.agents):
            # 随机位置（不在边界上）
            x = np.random.uniform(10, self.map_generator.width - 10)
            y = np.random.uniform(10, self.map_generator.height - 10)
            positions[agent] = np.array([x, y], dtype=np.float32)
        return positions
    
    def _init_prey_positions(self) -> Dict[str, np.ndarray]:
        """初始化猎物位置"""
        n_prey = self.config.get("n_prey", 5)
        prey_positions = {}
        for i in range(n_prey):
            # 随机位置（远离智能体）
            while True:
                x = np.random.uniform(10, self.map_generator.width - 10)
                y = np.random.uniform(10, self.map_generator.height - 10)
                
                # 检查是否远离所有智能体
                too_close = False
                for agent_pos in self.agent_positions.values():
                    dist = np.linalg.norm(agent_pos - np.array([x, y]))
                    if dist < 15:
                        too_close = True
                        break
                
                if not too_close:
                    prey_positions[f"prey_{i}"] = np.array([x, y], dtype=np.float32)
                    break
        
        return prey_positions
    
    def _get_agent_observation(self, agent_id: str) -> np.ndarray:
        """获取智能体观测"""
        # 获取局部障碍物地图
        agent_pos = self.agent_positions[agent_id]
        obstacle_obs = self.map_generator.get_agent_observation(agent_pos)
        
        # 获取其他智能体和猎物的相对位置
        other_agents = []
        for other_id, pos in self.agent_positions.items():
            if other_id != agent_id:
                rel_pos = pos - agent_pos
                other_agents.extend(rel_pos.tolist())
        
        preys = []
        for prey_id, pos in self.prey_positions.items():
            rel_pos = pos - agent_pos
            preys.extend(rel_pos.tolist())
        
        # 添加NuScenes场景特征
        scene_feature = self.current_scene_features[self.scene_step]
        
        # 组合所有特征
        features = [
            *obstacle_obs.flatten().tolist(),
            *other_agents,
            *preys,
            *scene_feature.tolist()
        ]
        
        # 转换为numpy数组
        obs = np.array(features, dtype=np.float32)
        
        # 如果超过特征维度，进行截断
        if len(obs) > self.feature_dim:
            obs = obs[:self.feature_dim]
        
        return obs
    
    def _calculate_reward(self, agent_id: str) -> float:
        """计算智能体奖励"""
        # 基础奖励 - 鼓励追捕猎物
        agent_pos = self.agent_positions[agent_id]
        reward = 0.0
        
        # 检查是否捕获猎物
        captured_prey = []
        for prey_id, prey_pos in self.prey_positions.items():
            dist = np.linalg.norm(agent_pos - prey_pos)
            if dist < 1.5:  # 捕获距离
                reward += self.config["reward"]["capture_reward"]
                captured_prey.append(prey_id)
        
        # 移除被捕获的猎物
        for prey_id in captured_prey:
            del self.prey_positions[prey_id]
        
        # 避免与障碍物碰撞
        obstacle_map = self.map_generator.get_obstacle_map()
        x, y = int(agent_pos[0]), int(agent_pos[1])
        if 0 <= x < self.map_generator.width and 0 <= y < self.map_generator.height:
            if obstacle_map[y, x] == 1:
                reward += self.config["reward"]["collision_penalty"]
        
        # 避免与动态障碍物碰撞
        for obstacle in self.map_generator.dynamic_obstacles:
            dist = np.linalg.norm(agent_pos - obstacle["position"])
            if dist < obstacle["size"] + 0.5:
                reward += self.config["reward"]["obstacle_collision_penalty"]
        
        # 时间惩罚
        reward += self.config["reward"]["time_penalty"]
        
        return reward
    
    def step(self, actions: Dict[str, int]):
        """执行一步环境更新"""
        # 更新智能体位置
        for agent_id, action in actions.items():
            # 解析动作
            if action == 0:  # 上
                self.agent_positions[agent_id][1] -= 1
            elif action == 1:  # 下
                self.agent_positions[agent_id][1] += 1
            elif action == 2:  # 左
                self.agent_positions[agent_id][0] -= 1
            elif action == 3:  # 右
                self.agent_positions[agent_id][0] += 1
            # 动作4: 不动
        
        # 更新猎物位置 (简单随机移动)
        for prey_id in list(self.prey_positions.keys()):
            # 随机移动
            move = np.random.choice([0, 1, 2, 3])
            if move == 0:  # 上
                self.prey_positions[prey_id][1] -= 0.5
            elif move == 1:  # 下
                self.prey_positions[prey_id][1] += 0.5
            elif move == 2:  # 左
                self.prey_positions[prey_id][0] -= 0.5
            elif move == 3:  # 右
                self.prey_positions[prey_id][0] += 0.5
            
            # 边界检查
            self.prey_positions[prey_id][0] = np.clip(
                self.prey_positions[prey_id][0], 0, self.map_generator.width)
            self.prey_positions[prey_id][1] = np.clip(
                self.prey_positions[prey_id][1], 0, self.map_generator.height)
        
        # 更新父类状态
        self.current_step += 1
        self.scene_step = (self.scene_step + 1) % len(self.current_scene_features)
        
        # 更新动态障碍物
        self.map_generator.update_dynamic_obstacles()
        
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
        
        # 如果所有猎物都被捕获，提前结束
        if not self.prey_positions:
            dones = {agent: True for agent in self.agents}
        
        terminations = {agent: dones[agent] for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        
        # 获取信息
        infos = {agent: self._get_info(agent) for agent in self.agents}
        
        return observations, rewards, terminations, truncations, infos
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """重置环境"""
        # 重置父类状态
        super().reset(seed=seed, options=options)
        
        # 重置地图生成器
        self.map_generator = CustomMapGenerator(self.config)
        
        # 重置智能体位置
        self.agent_positions = self._init_agent_positions()
        
        # 重置猎物位置
        self.prey_positions = self._init_prey_positions()
        
        # 获取初始观测
        observations = {
            agent: self._get_agent_observation(agent) for agent in self.agents
        }
        
        # 获取信息
        infos = {agent: self._get_info(agent) for agent in self.agents}
        
        return observations, infos
    
    def render(self):
        """渲染环境（文本模式）"""
        if self.render_mode == "human":
            # 简单文本渲染
            print(f"Step: {self.current_step}")
            print(f"Agents: {self.agent_positions}")
            print(f"Preys: {self.prey_positions}")
            print(f"Dynamic Obstacles: {self.map_generator.dynamic_obstacles}")
        
        return None