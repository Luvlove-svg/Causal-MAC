# communication/__init__.py
from .scheduler import CausalCommunicationScheduler
from .message_utils import MessageEncoder
from .reward_calculator import CounterfactualRewardCalculator
from .protocol import MCPProtocol
from causal_discovery import CausalDiscoveryManager
from configs import loader
import torch
import os

class CommunicationManager:
    """管理整个通信系统，集成调度器、协议和奖励计算"""
    
    def __init__(self, env_config_name: str):
        """
        初始化通信管理器
        
        参数:
            env_config_name: 环境配置名称
        """
        # 加载配置
        env_config = loader.get_env_config(env_config_name)["environment"]
        self.comm_config = env_config.get("communication", {})
        
        # 初始化因果管理器
        self.causal_manager = CausalDiscoveryManager(env_config_name)
        
        # 初始化组件
        self.scheduler = CausalCommunicationScheduler(self.causal_manager, self.comm_config)
        self.encoder = MessageEncoder(self.comm_config.get("encoding", {}))
        self.reward_calculator = CounterfactualRewardCalculator(self.comm_config.get("reward", {}))
        self.protocol = MCPProtocol(self.comm_config.get("protocol", {}))
        
        # 加载预训练模型（如果存在）
        self._load_pretrained_models(env_config_name)
    
    def _load_pretrained_models(self, env_config_name: str):
        """加载预训练模型"""
        model_dir = "models/communication"
        model_path = os.path.join(model_dir, f"{env_config_name}_scheduler.pth")
        
        if os.path.exists(model_path):
            self.scheduler.load_model(model_path)
            print(f"加载预训练的通信调度器: {model_path}")
    
    def initialize(self, feature_names: List[str]):
        """初始化通信系统"""
        # 初始化因果图
        self.causal_manager.initialize_causal_graph(feature_names)
        
        # 启动MCP协议
        self.protocol.start()
    
    def shutdown(self):
        """关闭通信系统"""
        self.protocol.stop()
    
    def schedule_communications(self, agent_observations: Dict[str, np.ndarray]) -> List[Tuple[str, str, np.ndarray]]:
        """调度通信并返回消息列表"""
        # 更新因果图
        self.causal_manager.update_causal_graph(agent_observations)
        
        # 调度通信
        return self.scheduler.schedule_communications(agent_observations)
    
    def process_messages(self, messages: List[Tuple[str, str, np.ndarray]]) -> Dict[str, List[np.ndarray]]:
        """处理消息并分发给接收者"""
        # 按接收者分组消息
        receiver_messages = {}
        for sender_id, receiver_id, message in messages:
            if receiver_id not in receiver_messages:
                receiver_messages[receiver_id] = []
            receiver_messages[receiver_id].append(message)
        
        return receiver_messages
    
    def encode_messages(self, messages: List[Tuple[str, str, np.ndarray]]) -> bytes:
        """编码消息列表为字节流"""
        message_data = [msg for _, _, msg in messages]
        metadata = [{"sender": s, "receiver": r} for s, r, _ in messages]
        return self.encoder.encode_batch(message_data, metadata)
    
    def decode_messages(self, encoded: bytes) -> List[Tuple[str, str, np.ndarray]]:
        """从字节流解码消息列表"""
        decoded = self.encoder.decode_batch(encoded)
        messages = []
        for (msg, meta) in decoded:
            sender = meta.get("sender", "unknown")
            receiver = meta.get("receiver", "unknown")
            messages.append((sender, receiver, msg))
        return messages
    
    def calculate_rewards(self, 
                         actions_with_message: List[torch.Tensor],
                         actions_without_message: List[torch.Tensor],
                         optimal_actions: List[torch.Tensor],
                         baseline_rewards: List[float]) -> Tuple[List[float], List[float]]:
        """计算反事实通信奖励"""
        return self.reward_calculator.batch_calculate(
            actions_with_message,
            actions_without_message,
            optimal_actions,
            baseline_rewards
        )
    
    def call_tool(self, tool_name: str, input_data: Any, metadata: Dict = None) -> Any:
        """调用外部工具"""
        return self.protocol.call_tool(tool_name, input_data, metadata)
    
    def broadcast_to_tools(self, message: Any, metadata: Dict = None, exclude: List[str] = None):
        """向所有工具广播消息"""
        self.protocol.broadcast(message, metadata, exclude)
    
    def get_tool_status(self) -> Dict[str, str]:
        """获取工具状态"""
        return self.protocol.tool_status()
    
    def save_models(self, env_config_name: str):
        """保存模型到文件"""
        model_dir = "models/communication"
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, f"{env_config_name}_scheduler.pth")
        self.scheduler.save_model(model_path)
        print(f"保存通信调度器模型到: {model_path}")
    
    def reset(self):
        """重置状态"""
        self.scheduler.reset()