# causal_discovery/__init__.py
from .pc_algorithm import PCAlgorithm
from .fci_pc import FCIPCAlgorithm
from .nsa_attention import NSAttentionCausalDiscovery
from .causal_integration import CausalDiscoveryManager
from configs import loader

def create_causal_discovery_manager(env_config_name: str) -> CausalDiscoveryManager:
    """创建因果发现管理器"""
    # 加载环境配置
    env_config = loader.get_env_config(env_config_name)["environment"]
    
    # 获取因果发现配置
    causal_config = env_config.get("causal_reasoning", {})
    causal_config["env_name"] = env_config_name
    causal_config["feature_dim"] = env_config["observation"]["feature_dim"]
    
    # 创建管理器
    manager = CausalDiscoveryManager(causal_config)
    return manager