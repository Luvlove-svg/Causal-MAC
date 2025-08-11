# environments/__init__.py
from .pursuit_env import PursuitEnv
from .custom_pursuit_env import CustomPursuitEnv
from configs import loader

def make_env(env_name: str, env_config: str = None, render_mode: str = None):
    """
    创建多智能体环境
    
    参数:
        env_name: 环境名称 ('pursuit', 'custom_pursuit')
        env_config: 环境配置名称 (对应configs/env中的yaml文件)
        render_mode: 渲染模式 (None, 'human', 'rgb_array')
    
    返回:
        初始化好的环境实例
    """
    # 如果没有指定配置，使用环境名称作为默认配置
    if env_config is None:
        env_config = env_name
    
    # 加载配置以检查环境类型
    config = loader.get_env_config(env_config)["environment"]
    env_type = config.get("base_environment", env_name)
    
    # 创建环境实例
    if env_type == "pursuit_v4":
        return PursuitEnv(env_config, render_mode)
    elif env_type == "custom_pursuit_map":
        return CustomPursuitEnv(env_config, render_mode)
    else:
        raise ValueError(f"未知的环境类型: {env_type}")