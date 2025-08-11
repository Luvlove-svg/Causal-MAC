# configs/__init__.py
import os
import yaml
from typing import Dict, Any

class ConfigLoader:
    def __init__(self, base_path: str = "configs"):
        self.base_path = base_path
        self.cache = {}
    
    def load(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        # 检查缓存
        if config_path in self.cache:
            return self.cache[config_path]
        
        # 构建完整路径
        full_path = os.path.join(self.base_path, config_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Config file not found: {full_path}")
        
        # 加载YAML文件
        with open(full_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 缓存结果
        self.cache[config_path] = config
        return config
    
    def merge_configs(self, *config_paths: str) -> Dict[str, Any]:
        """合并多个配置文件"""
        merged_config = {}
        for path in config_paths:
            config = self.load(path)
            self._deep_merge(merged_config, config)
        return merged_config
    
    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """深度合并两个字典"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
        return base

# 创建全局配置加载器实例
loader = ConfigLoader()

# 示例使用函数
def get_env_config(env_name: str = "pursuit_v4") -> Dict[str, Any]:
    """获取环境配置"""
    return loader.load(f"env/{env_name}.yaml")

def get_model_config(model_name: str = "causal_mac") -> Dict[str, Any]:
    """获取模型配置"""
    return loader.load(f"model/{model_name}.yaml")

def get_training_config() -> Dict[str, Any]:
    """获取训练配置"""
    return loader.load("training.yaml")

def get_full_config(env_name: str = "custom_map", 
                   model_name: str = "causal_mac") -> Dict[str, Any]:
    """获取完整配置（环境+模型+训练）"""
    env_config = get_env_config(env_name)
    model_config = get_model_config(model_name)
    training_config = get_training_config()
    
    # 合并配置
    full_config = {
        "environment": env_config.get("environment", {}),
        "model": model_config.get("model", {}),
        "training": training_config.get("training", {})
    }
    return full_config