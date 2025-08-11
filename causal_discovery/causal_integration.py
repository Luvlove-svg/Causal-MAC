# causal_discovery/causal_integration.py
import numpy as np
import networkx as nx
from typing import Dict, Any, List, Tuple
from .pc_algorithm import PCAlgorithm
from .fci_pc import FCIPCAlgorithm
from .nsa_attention import NSAttentionCausalDiscovery
from configs import loader
import torch
import os

class CausalDiscoveryManager:
    """管理因果发现过程，与多智能体环境集成"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化因果发现管理器
        
        参数:
            config: 因果发现配置
        """
        self.config = config
        self.method = config.get("method", "fci-pc")
        self.update_interval = config.get("update_interval", 100)
        self.feature_dim = config.get("feature_dim", 30)
        self.causal_graph = None
        self.feature_names = []
        
        # 初始化因果发现方法
        if self.method == "pc":
            self.discovery_engine = PCAlgorithm(config)
        elif self.method == "fci-pc":
            self.discovery_engine = FCIPCAlgorithm(config)
        elif self.method == "nsa":
            self.nsa_model = NSAttentionCausalDiscovery({
                "feature_dim": self.feature_dim,
                "sparsity": config.get("sparsity", 0.3),
                "n_heads": config.get("n_heads", 4),
                "hardware_aware": config.get("hardware_aware", True)
            })
            # 加载预训练模型或训练新模型
            self._load_or_train_model()
        else:
            raise ValueError(f"未知的因果发现方法: {self.method}")
    
    def _load_or_train_model(self):
        """加载或训练NSA模型"""
        model_path = "models/causal_nsa.pth"
        
        if os.path.exists(model_path):
            # 加载预训练模型
            self.nsa_model.load_state_dict(torch.load(model_path))
            print("加载预训练的NSA因果模型")
        else:
            # 从数据训练新模型
            print("训练新的NSA因果模型...")
            # 这里应该从实际数据加载，但为简化使用随机数据
            # 在实际应用中，应加载预处理的因果发现数据
            train_data = np.random.randn(1000, self.feature_dim)  # 1000个样本，每个特征维度
            self.nsa_model.learn_from_data(train_data, epochs=100)
            torch.save(self.nsa_model.state_dict(), model_path)
    
    def initialize_causal_graph(self, feature_names: List[str]):
        """初始化因果图"""
        self.feature_names = feature_names
        
        # 如果已有保存的因果图，加载它
        graph_path = f"data/processed/causal_graphs/{self.config['env_name']}_causal_graph.npz"
        if os.path.exists(graph_path):
            data = np.load(graph_path)
            adj_matrix = data['graph']
            self.causal_graph = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
            # 设置节点标签
            for i, name in enumerate(feature_names):
                self.causal_graph.nodes[i]['label'] = name
            print(f"从 {graph_path} 加载预计算的因果图")
        else:
            # 使用随机图初始化
            self.causal_graph = nx.DiGraph()
            for i, name in enumerate(feature_names):
                self.causal_graph.add_node(i, label=name)
            print("初始化空因果图")
    
    def update_causal_graph(self, observations: Dict[str, np.ndarray]):
        """使用新的观测数据更新因果图"""
        # 如果未达到更新间隔，跳过
        if self.update_interval > 0 and self.env_step % self.update_interval != 0:
            return
        
        # 收集所有智能体的观测数据
        all_obs = []
        for agent_obs in observations.values():
            if len(agent_obs) == self.feature_dim:
                all_obs.append(agent_obs)
        
        if len(all_obs) < 10:  # 最少需要一些样本
            return
        
        # 转换为numpy数组
        data = np.array(all_obs)
        
        # 运行因果发现
        if self.method in ["pc", "fci-pc"]:
            new_graph = self.discovery_engine.run(data, self.feature_names)
        elif self.method == "nsa":
            with torch.no_grad():
                data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
                pred_graph = self.nsa_model(data_tensor)[0].mean(dim=0).numpy()
                # 应用阈值
                threshold = self.config.get("nsa_threshold", 0.6)
                adj_matrix = (pred_graph > threshold).astype(int)
                np.fill_diagonal(adj_matrix, 0)  # 移除自环
                new_graph = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
                # 设置节点标签
                for i, name in enumerate(self.feature_names):
                    new_graph.nodes[i]['label'] = name
        
        # 合并新旧图 (简单平均)
        self._merge_graphs(new_graph)
        
        # 保存更新后的图
        self._save_causal_graph()
    
    def _merge_graphs(self, new_graph: nx.DiGraph, alpha: float = 0.7):
        """合并新旧因果图"""
        if self.causal_graph is None:
            self.causal_graph = new_graph
            return
        
        # 创建新邻接矩阵
        nodes = sorted(self.causal_graph.nodes)
        n = len(nodes)
        old_adj = nx.to_numpy_array(self.causal_graph, nodelist=nodes)
        new_adj = nx.to_numpy_array(new_graph, nodelist=nodes)
        
        # 合并邻接矩阵
        merged_adj = alpha * old_adj + (1 - alpha) * new_adj
        
        # 应用阈值
        threshold = self.config.get("merge_threshold", 0.5)
        merged_adj = (merged_adj > threshold).astype(int)
        
        # 创建新图
        self.causal_graph = nx.from_numpy_array(merged_adj, create_using=nx.DiGraph)
        for i, node in enumerate(nodes):
            self.causal_graph.nodes[i]['label'] = self.causal_graph.nodes[node]['label']
    
    def _save_causal_graph(self):
        """保存因果图到文件"""
        if self.causal_graph is None:
            return
        
        # 确保目录存在
        os.makedirs("data/processed/causal_graphs", exist_ok=True)
        
        # 转换为邻接矩阵
        nodes = sorted(self.causal_graph.nodes)
        adj_matrix = nx.to_numpy_array(self.causal_graph, nodelist=nodes)
        
        # 保存为npz文件
        graph_path = f"data/processed/causal_graphs/{self.config['env_name']}_causal_graph.npz"
        np.savez_compressed(graph_path, graph=adj_matrix)
    
    def get_causal_relations(self, feature_i: int, feature_j: int) -> float:
        """获取两个特征之间的因果关系强度"""
        if self.causal_graph is None:
            return 0.0
        
        # 检查是否有直接边
        if self.causal_graph.has_edge(feature_i, feature_j):
            return 1.0
        
        # 检查是否有间接路径
        try:
            # 计算最短路径长度
            path_length = nx.shortest_path_length(self.causal_graph, source=feature_i, target=feature_j)
            return 1.0 / (path_length + 1)  # 路径越长，关系越弱
        except nx.NetworkXNoPath:
            return 0.0
    
    def should_communicate(self, agent_i: int, agent_j: int, 
                          obs_i: np.ndarray, obs_j: np.ndarray) -> bool:
        """基于因果图决定两个智能体是否应该通信"""
        if self.causal_graph is None:
            return False
        
        # 计算观测差异
        diff = np.abs(obs_i - obs_j)
        
        # 检查每个特征是否因果相关
        for feat_idx in range(len(obs_i)):
            # 检查特征是否在因果图中相关
            causal_strength = self.get_causal_relations(feat_idx, feat_idx)
            
            # 如果特征差异大且因果相关，可能需要通信
            if diff[feat_idx] > 0.5 and causal_strength > 0.3:
                return True
        
        return False
    
    def visualize_causal_graph(self, save_path: str = None):
        """可视化因果图"""
        if self.causal_graph is None:
            print("因果图未初始化")
            return
        
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 10))
        
        # 获取节点标签
        labels = {node: data['label'] for node, data in self.causal_graph.nodes(data=True)}
        
        # 绘制图
        pos = nx.spring_layout(self.causal_graph, seed=42)
        nx.draw_networkx_nodes(self.causal_graph, pos, node_size=700, node_color='lightblue')
        nx.draw_networkx_edges(self.causal_graph, pos, arrowstyle='->', arrowsize=20)
        nx.draw_networkx_labels(self.causal_graph, pos, labels, font_size=10)
        
        plt.title("Causal Graph")
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"保存因果图到 {save_path}")
        else:
            plt.show()
    
    def to_adjacency_matrix(self) -> np.ndarray:
        """将因果图转换为邻接矩阵"""
        if self.causal_graph is None:
            return np.zeros((self.feature_dim, self.feature_dim))
        
        nodes = sorted(self.causal_graph.nodes)
        return nx.to_numpy_array(self.causal_graph, nodelist=nodes)