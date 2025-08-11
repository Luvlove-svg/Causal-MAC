# configs/validator.py
from typing import Dict, Any

class ConfigValidator:
    """配置验证工具"""
    
    REQUIRED_SECTIONS = {
        "environment": ["name", "n_agents", "max_cycles"],
        "model": ["name", "type", "input_dim", "output_dim"],
        "training": ["experiment_name", "num_episodes", "batch_size"]
    }
    
    VALID_VALUES = {
        "environment": {
            "observation.type": ["raw", "feature_extracted"],
            "advanced.use_partial_obs": [True, False]
        },
        "model": {
            "quantization": ["none", "int8", "int4"],
            "hardware.use_mixed_precision": [True, False]
        },
        "training": {
            "optimizer.type": ["adam", "adamw", "sgd", "rmsprop"],
            "exploration.type": ["epsilon_greedy", "boltzmann", "gaussian"]
        }
    }
    
    RANGE_CONSTRAINTS = {
        "environment.n_agents": (1, 20),
        "model.causal_reasoning.update_interval": (0, 1000),
        "training.learning_rate": (1e-6, 1.0)
    }
    
    def validate(self, config: Dict[str, Any], config_type: str) -> bool:
        """验证配置"""
        # 检查必需字段
        if not self._check_required_fields(config, config_type):
            return False
        
        # 检查有效值
        if not self._check_valid_values(config, config_type):
            return False
        
        # 检查范围约束
        if not self._check_range_constraints(config, config_type):
            return False
        
        return True
    
    def _check_required_fields(self, config: Dict[str, Any], config_type: str) -> bool:
        """检查必需字段是否存在"""
        required = self.REQUIRED_SECTIONS.get(config_type, [])
        for field in required:
            if field not in config:
                print(f"Missing required field: {field} in {config_type} config")
                return False
        return True
    
    def _check_valid_values(self, config: Dict[str, Any], config_type: str) -> bool:
        """检查字段值是否有效"""
        valid_values = self.VALID_VALUES.get(config_type, {})
        for path, valid_options in valid_values.items():
            # 获取嵌套值
            keys = path.split('.')
            value = config
            for key in keys:
                if key in value:
                    value = value[key]
                else:
                    break
            
            if value not in valid_options:
                print(f"Invalid value for {path}: {value}. Valid options: {valid_options}")
                return False
        return True
    
    def _check_range_constraints(self, config: Dict[str, Any], config_type: str) -> bool:
        """检查数值范围约束"""
        range_constraints = {
            k: v for k, v in self.RANGE_CONSTRAINTS.items() 
            if k.startswith(config_type)
        }
        
        for path, (min_val, max_val) in range_constraints.items():
            # 提取字段名（去掉类型前缀）
            field = path.split('.', 1)[1]
            
            # 获取值
            keys = field.split('.')
            value = config
            for key in keys:
                if key in value:
                    value = value[key]
                else:
                    break
            
            if not (min_val <= value <= max_val):
                print(f"Value out of range for {field}: {value}. Must be between {min_val} and {max_val}")
                return False
        return True