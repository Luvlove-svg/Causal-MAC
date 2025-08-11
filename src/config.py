import torch
import os


class Config:
    def __init__(self):
        # 项目根目录
        self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # 数据路径
        self.data_path = os.path.join(self.root_dir, "data", "processed", "nuscenes")

        # 模型参数
        self.agent_feature_dim = 6  # x, y, 长, 宽, 高, 朝向
        self.hidden_dim = 128
        self.num_heads = 4
        self.num_layers = 3

        # 训练参数
        self.batch_size = 8
        self.learning_rate = 0.001
        self.epochs = 50
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 保存路径
        self.model_save_path = os.path.join(self.root_dir, "models", "causal_mac_nuscenes.pth")


config = Config()