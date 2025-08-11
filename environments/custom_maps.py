# environments/custom_maps.py
import numpy as np
from typing import List, Dict, Any

class CustomMapGenerator:
    """自定义地图生成器，支持动态障碍物和复杂场景"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化地图生成器
        
        参数:
            config: 地图配置字典
        """
        self.config = config
        self.width = config.get("width", 100)
        self.height = config.get("height", 100)
        self.grid = np.zeros((self.height, self.width), dtype=int)
        
        # 生成基本地图
        self._generate_base_map()
        
        # 生成动态障碍物
        self.dynamic_obstacles = []
        if config.get("dynamic_obstacles", {}).get("enabled", False):
            self._generate_dynamic_obstacles()
    
    def _generate_base_map(self):
        """生成基础地图"""
        # 创建边界
        self.grid[0, :] = 1  # 上边界
        self.grid[-1, :] = 1  # 下边界
        self.grid[:, 0] = 1  # 左边界
        self.grid[:, -1] = 1  # 右边界
        
        # 添加静态障碍物
        obstacle_config = self.config.get("static_obstacles", {})
        n_obstacles = obstacle_config.get("count", 10)
        min_size = obstacle_config.get("min_size", 2)
        max_size = obstacle_config.get("max_size", 5)
        
        for _ in range(n_obstacles):
            # 随机位置和大小
            x = np.random.randint(1, self.width - max_size - 1)
            y = np.random.randint(1, self.height - max_size - 1)
            w = np.random.randint(min_size, max_size)
            h = np.random.randint(min_size, max_size)
            
            # 添加障碍物
            self.grid[y:y+h, x:x+w] = 1
    
    def _generate_dynamic_obstacles(self):
        """生成动态障碍物"""
        obstacle_config = self.config["dynamic_obstacles"]
        n_obstacles = obstacle_config.get("num_obstacles", 5)
        min_speed = obstacle_config.get("min_speed", 0.1)
        max_speed = obstacle_config.get("max_speed", 0.5)
        
        for _ in range(n_obstacles):
            # 随机位置（不在边界上）
            x = np.random.uniform(5, self.width - 5)
            y = np.random.uniform(5, self.height - 5)
            
            # 随机速度和方向
            speed = np.random.uniform(min_speed, max_speed)
            angle = np.random.uniform(0, 2 * np.pi)
            vx = np.cos(angle) * speed
            vy = np.sin(angle) * speed
            
            # 随机大小
            size = np.random.uniform(0.5, 2.0)
            
            self.dynamic_obstacles.append({
                "position": np.array([x, y], dtype=np.float32),
                "velocity": np.array([vx, vy], dtype=np.float32),
                "size": size
            })
    
    def update_dynamic_obstacles(self):
        """更新动态障碍物位置"""
        for obstacle in self.dynamic_obstacles:
            # 更新位置
            obstacle["position"] += obstacle["velocity"]
            
            # 边界检查
            x, y = obstacle["position"]
            size = obstacle["size"]
            
            # 左右边界碰撞
            if x - size < 0 or x + size > self.width:
                obstacle["velocity"][0] *= -1
                # 确保不会卡在边界
                obstacle["position"][0] = np.clip(obstacle["position"][0], size, self.width - size)
            
            # 上下边界碰撞
            if y - size < 0 or y + size > self.height:
                obstacle["velocity"][1] *= -1
                obstacle["position"][1] = np.clip(obstacle["position"][1], size, self.height - size)
    
    def get_obstacle_map(self, time_step: int = 0) -> np.ndarray:
        """获取当前时间步的障碍物地图"""
        # 创建临时地图（包含静态和动态障碍物）
        temp_map = self.grid.copy()
        
        # 添加动态障碍物
        for obstacle in self.dynamic_obstacles:
            x, y = obstacle["position"]
            size = obstacle["size"]
            
            # 计算障碍物在网格中的位置
            min_x = max(0, int(x - size))
            max_x = min(self.width, int(x + size) + 1)
            min_y = max(0, int(y - size))
            max_y = min(self.height, int(y + size) + 1)
            
            # 标记障碍物区域
            temp_map[min_y:max_y, min_x:max_x] = 1
        
        return temp_map
    
    def get_agent_observation(self, agent_pos: np.ndarray, agent_radius: int = 5) -> np.ndarray:
        """
        获取智能体局部观测
        
        参数:
            agent_pos: 智能体位置 [x, y]
            agent_radius: 观测半径
            
        返回:
            局部观测网格
        """
        # 获取完整障碍物地图
        full_map = self.get_obstacle_map()
        
        # 计算观测范围
        x, y = agent_pos
        min_x = max(0, int(x - agent_radius))
        max_x = min(self.width, int(x + agent_radius) + 1)
        min_y = max(0, int(y - agent_radius))
        max_y = min(self.height, int(y + agent_radius) + 1)
        
        # 提取局部观测
        local_obs = full_map[min_y:max_y, min_x:max_x]
        
        # 如果观测区域小于指定大小，进行填充
        obs_height, obs_width = local_obs.shape
        if obs_height < 2 * agent_radius + 1 or obs_width < 2 * agent_radius + 1:
            padded_obs = np.zeros((2 * agent_radius + 1, 2 * agent_radius + 1), dtype=int)
            start_y = agent_radius - (int(y) - min_y)
            start_x = agent_radius - (int(x) - min_x)
            padded_obs[start_y:start_y+obs_height, start_x:start_x+obs_width] = local_obs
            return padded_obs
        
        return local_obs