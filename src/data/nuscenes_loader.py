import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class NuScenesDataset(Dataset):
    def __init__(self, data_path, mode="train", test_size=0.2, random_state=42):
        self.mode = mode
        npz_path = os.path.join(data_path, "processed_data.npz")
        data = np.load(npz_path, allow_pickle=True)

        # 获取代理特征和因果关系
        self.agent_features = data["agent_features"]
        self.causal_relations = data["causal_relations"]

        # 分割数据集
        indices = np.arange(len(self.agent_features))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=random_state)
        val_idx, test_idx = train_test_split(
            test_idx, test_size=0.5, random_state=random_state)

        if mode == "train":
            self.indices = train_idx
        elif mode == "val":
            self.indices = val_idx
        else:  # test
            self.indices = test_idx

    def __len__(self):
        return len(self.indices)

    # src/data/nuscenes_loader.py

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        agents = self.agent_features[actual_idx]
        relations = self.causal_relations[actual_idx]

        # 转换为张量
        agent_tensor = torch.tensor(agents, dtype=torch.float32)

        # 处理关系数据
        if len(relations) > 0:
            # 只取关系强度值（第三列）并展平
            relation_strength = np.array(relations)[:, 2].flatten()

            # 归一化到 [0, 1] 范围
            min_val = np.min(relation_strength)
            max_val = np.max(relation_strength)
            if max_val - min_val > 0:
                relation_strength = (relation_strength - min_val) / (max_val - min_val)

            # 裁剪确保在 [0, 1] 范围内
            relation_strength = np.clip(relation_strength, 0.0, 1.0)

            relation_tensor = torch.tensor(relation_strength, dtype=torch.float32)
        else:
            relation_tensor = torch.zeros(0, dtype=torch.float32)

        return {
            "agents": agent_tensor,  # 形状: [num_agents, features]
            "relations": relation_tensor  # 形状: [num_relations]
        }


def get_nuscenes_dataloader(data_path, batch_size=8, mode="train"):
    dataset = NuScenesDataset(data_path, mode)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == "train"),
        num_workers=0,  # Windows下建议设为0
        collate_fn=lambda x: x  # 自定义批处理
    )