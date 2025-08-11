import os
import json
import numpy as np
from nuscenes.nuscenes import NuScenes
import logging
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocess_nuscenes.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("NuScenesPreprocessor")


def preprocess_nuscenes(data_path: str, output_path: str):
    """
    预处理NuScenes数据集，提取关键特征
    """
    logger.info("初始化NuScenes数据集...")
    try:
        nusc = NuScenes(version='v1.0-mini', dataroot=data_path, verbose=True)
        logger.info(f"数据集加载成功，包含 {len(nusc.scene)} 个场景")
    except Exception as e:
        logger.error(f"初始化NuScenes失败: {str(e)}")
        return

    os.makedirs(output_path, exist_ok=True)

    # 存储处理后的数据
    processed_data = {
        "scene_tokens": [],
        "sample_tokens": [],
        "agent_features": [],
        "context_features": [],
        "causal_relations": []
    }

    logger.info("开始处理场景和样本数据...")

    # 遍历所有场景
    for scene_idx, scene in tqdm(enumerate(nusc.scene), total=len(nusc.scene), desc="处理场景"):
        scene_token = scene['token']
        processed_data["scene_tokens"].append(scene_token)

        # 获取场景的第一个样本
        sample_token = scene['first_sample_token']
        sample_count = 0

        # 遍历场景中的所有样本
        while sample_token:
            try:
                sample = nusc.get('sample', sample_token)
                processed_data["sample_tokens"].append(sample_token)
                sample_count += 1

                # 提取代理特征（如车辆、行人）
                agent_features = []
                for ann_token in sample['anns']:
                    ann = nusc.get('sample_annotation', ann_token)
                    # 提取位置、尺寸和方向
                    agent_features.append([
                        ann['translation'][0],  # x
                        ann['translation'][1],  # y
                        ann['size'][0],  # 长度
                        ann['size'][1],  # 宽度
                        ann['size'][2],  # 高度
                        ann['rotation'][2]  # 朝向角 (yaw)
                    ])

                # 提取上下文特征（如交通灯、道路）
                context_features = []
                # 改为使用 'anns' 而不是 'instances'
                for ann_token in sample['anns']:
                    ann = nusc.get('sample_annotation', ann_token)
                    # 提取相关特征作为上下文
                    context_features.append([
                        ann['attribute_tokens'],  # 属性
                        ann['visibility_token'],  # 可见性
                        # 添加其他需要的特征
                    ])

                # 提取因果关系（基于运动轨迹）
                causal_relations = []
                if len(agent_features) > 1:
                    for i in range(len(agent_features)):
                        for j in range(len(agent_features)):
                            if i != j:
                                # 计算两个代理之间的距离
                                pos_i = np.array(agent_features[i][:2])
                                pos_j = np.array(agent_features[j][:2])
                                dist = np.linalg.norm(pos_i - pos_j)
                                causal_relations.append([i, j, float(dist)])

                processed_data["agent_features"].append(agent_features)
                processed_data["context_features"].append(context_features)
                processed_data["causal_relations"].append(causal_relations)

                # 移动到下一个样本
                sample_token = sample['next'] if 'next' in sample else None

            except Exception as e:
                logger.error(f"处理样本 {sample_token} 时出错: {str(e)}")
                sample_token = None  # 跳过此样本

    logger.info(f"保存处理后的数据到 {output_path}...")
    try:
        # 保存为JSON
        with open(os.path.join(output_path, "processed_data.json"), "w") as f:
            json.dump(processed_data, f, indent=2)

        # 保存为NumPy格式以便快速加载
        # 注意: 使用dtype=object处理不规则形状的数据
        np.savez(
            os.path.join(output_path, "processed_data.npz"),
            agent_features=np.array(processed_data["agent_features"], dtype=object),
            context_features=np.array(processed_data["context_features"], dtype=object),
            causal_relations=np.array(processed_data["causal_relations"], dtype=object)
        )

        logger.info(f"预处理完成! 共处理 {len(processed_data['scene_tokens'])} 个场景, "
                    f"{len(processed_data['sample_tokens'])} 个样本")

    except Exception as e:
        logger.error(f"保存数据时出错: {str(e)}")


if __name__ == "__main__":
    # 路径配置
    raw_data_path = r"D:\我的大学\ccf\agent\Causal-MAC\data\nuscenes\v1.0-mini"
    processed_path = r"D:\我的大学\ccf\agent\Causal-MAC\data\processed\nuscenes"

    # 确保路径存在
    os.makedirs(processed_path, exist_ok=True)

    logger.info(f"原始数据路径: {raw_data_path}")
    logger.info(f"处理后数据路径: {processed_path}")

    preprocess_nuscenes(raw_data_path, processed_path)