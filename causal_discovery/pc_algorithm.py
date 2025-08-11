# causal_discovery/pc_algorithm.py
import numpy as np
from typing import List, Tuple, Dict, Set, Optional
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, chi2_contingency
from joblib import Parallel, delayed
import networkx as nx
import time
from configs import loader

class PCAlgorithm:
    """Peter-Clark (PC) 算法实现，用于因果发现"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化PC算法
        
        参数:
            config: 因果发现配置
        """
        self.config = config
        self.alpha = config.get("alpha", 0.05)  # 显著性水平
        self.independence_test = config.get("independence_test", "pearson")
        self.max_condition_size = config.get("max_condition_size", 3)
        self.parallel = config.get("parallel", True)
        self.n_jobs = config.get("n_jobs", -1)
        self.verbose = config.get("verbose", False)
        
        # 用于缓存独立性测试结果
        self.cache = {}
        
    def _conditional_independence_test(self, data: np.ndarray, i: int, j: int, 
                                      cond_set: Optional[Set[int]] = None) -> Tuple[bool, float]:
        """执行条件独立性测试"""
        # 生成缓存键
        cache_key = (i, j, frozenset(cond_set)) if cond_set else (i, j, None)
        
        # 检查缓存
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # 提取相关数据
        X = data[:, i]
        Y = data[:, j]
        
        # 条件集为空 - 无条件独立性测试
        if cond_set is None or len(cond_set) == 0:
            if self.independence_test == "pearson":
                corr, p_value = pearsonr(X, Y)
                independent = p_value > self.alpha
            elif self.independence_test == "chi2":
                # 离散化数据
                X_bin = np.digitize(X, bins=np.histogram_bin_edges(X, bins=5))
                Y_bin = np.digitize(Y, bins=np.histogram_bin_edges(Y, bins=5))
                contingency = np.zeros((len(np.unique(X_bin)), len(np.unique(Y_bin))))
                for x, y in zip(X_bin, Y_bin):
                    contingency[x, y] += 1
                chi2, p_value, dof, _ = chi2_contingency(contingency)
                independent = p_value > self.alpha
            else:
                raise ValueError(f"未知的独立性检验方法: {self.independence_test}")
        else:
            # 条件集不为空
            cond_indices = sorted(cond_set)
            Z = data[:, cond_indices]
            
            if self.independence_test == "pearson":
                # 使用偏相关
                from sklearn.covariance import EmpiricalCovariance
                cov = EmpiricalCovariance().fit(np.column_stack([X, Y, Z]))
                corr_matrix = cov.covariance_
                partial_corr = self._partial_correlation(corr_matrix, 0, 1, list(range(2, 2+len(cond_indices)))
                n = len(X)
                t_stat = partial_corr * np.sqrt(n - 2 - len(cond_indices)) / np.sqrt(1 - partial_corr**2)
                p_value = 2 * (1 - t.cdf(np.abs(t_stat), df=n - 2 - len(cond_indices)))
                independent = p_value > self.alpha
            elif self.independence_test == "chi2":
                # 使用条件卡方检验 (简化实现)
                independent = True
                # 在实际应用中，这里应实现更精确的条件独立性检验
            else:
                raise ValueError(f"未知的独立性检验方法: {self.independence_test}")
        
        # 缓存结果
        self.cache[cache_key] = (independent, p_value)
        return independent, p_value
    
    def _partial_correlation(self, cov_matrix, i, j, cond_indices):
        """计算偏相关系数"""
        # 计算协方差矩阵的子矩阵
        indices = [i, j] + cond_indices
        sub_cov = cov_matrix[np.ix_(indices, indices)]
        
        # 计算逆协方差矩阵
        inv_sub_cov = np.linalg.inv(sub_cov)
        
        # 偏相关系数公式
        p_corr = -inv_sub_cov[0, 1] / np.sqrt(inv_sub_cov[0, 0] * inv_sub_cov[1, 1])
        return p_corr
    
    def _estimate_skeleton(self, data: np.ndarray, variables: List[str]) -> nx.Graph:
        """估计骨架图"""
        n_vars = data.shape[1]
        graph = nx.complete_graph(n_variables=n_vars, create_using=nx.Graph)
        node_labels = {i: var for i, var in enumerate(variables)}
        nx.set_node_attributes(graph, node_labels, "label")
        
        # 分离集
        sep_set = { (i, j): set() for i, j in graph.edges }
        
        # 按条件集大小分层搜索
        for cond_size in range(0, self.max_condition_size + 1):
            edges = list(graph.edges)
            remove_edges = []
            
            if self.parallel:
                # 并行处理边
                results = Parallel(n_jobs=self.n_jobs)(
                    delayed(self._test_edge)(data, i, j, graph, cond_size)
                    for i, j in edges
                )
                
                for (i, j), (independent, p_value, found_sep_set) in zip(edges, results):
                    if independent:
                        remove_edges.append((i, j))
                        sep_set[(i, j)] = found_sep_set
            else:
                # 串行处理边
                for i, j in edges:
                    independent, p_value, found_sep_set = self._test_edge(data, i, j, graph, cond_size)
                    if independent:
                        remove_edges.append((i, j))
                        sep_set[(i, j)] = found_sep_set
            
            # 移除独立的边
            graph.remove_edges_from(remove_edges)
            
            if self.verbose:
                print(f"条件集大小 {cond_size}: 移除了 {len(remove_edges)} 条边")
        
        return graph, sep_set
    
    def _test_edge(self, data, i, j, graph, cond_size):
        """测试一条边的条件独立性"""
        # 获取i和j的相邻节点（不包括彼此）
        adj_i = set(graph.neighbors(i)) - {j}
        adj_j = set(graph.neighbors(j)) - {i}
        possible_cond_sets = adj_i | adj_j
        
        # 尝试所有大小为cond_size的条件集组合
        independent = False
        p_value = 1.0
        found_sep_set = None
        
        # 如果条件集大小大于可能的邻居数，跳过
        if cond_size > len(possible_cond_sets):
            return independent, p_value, found_sep_set
        
        # 生成所有可能的条件集
        from itertools import combinations
        for cond_set in combinations(possible_cond_sets, cond_size):
            cond_set = set(cond_set)
            independent, p_val = self._conditional_independence_test(data, i, j, cond_set)
            if independent:
                found_sep_set = cond_set
                p_value = min(p_value, p_val)
                break  # 找到一个分离集即可
        
        return independent, p_value, found_sep_set
    
    def _orient_edges(self, graph: nx.Graph, sep_set: Dict[Tuple[int, int], Set[int]]) -> nx.DiGraph:
        """定向边（v-结构）"""
        digraph = graph.to_directed()
        
        # 识别v-结构 (X → Z ← Y)
        for pair in sep_set:
            i, j = pair
            for k in set(graph.nodes) - {i, j}:
                if k in graph.neighbors(i) and k in graph.neighbors(j):
                    if k not in sep_set[(i, j)]:
                        # 确保没有冲突的边
                        if digraph.has_edge(k, i):
                            digraph.remove_edge(k, i)
                        if digraph.has_edge(k, j):
                            digraph.remove_edge(k, j)
                        digraph.add_edge(i, k)
                        digraph.add_edge(j, k)
        
        # 应用其他定向规则 (避免循环)
        self._apply_orientation_rules(digraph)
        
        return digraph
    
    def _apply_orientation_rules(self, digraph: nx.DiGraph):
        """应用PC算法的定向规则"""
        changed = True
        while changed:
            changed = False
            
            # 规则1: 如果 A → B - C 且 A 和 C 不相邻，则定向为 B → C
            for b in list(digraph.nodes):
                for a in digraph.predecessors(b):
                    for c in digraph.neighbors(b):
                        if c == a or digraph.has_edge(a, c) or digraph.has_edge(c, a):
                            continue
                        if not digraph.has_edge(b, c) and not digraph.has_edge(c, b):
                            digraph.add_edge(b, c)
                            changed = True
                        elif digraph.has_edge(c, b):
                            digraph.remove_edge(c, b)
                            digraph.add_edge(b, c)
                            changed = True
            
            # 规则2: 避免新环
            try:
                nx.find_cycle(digraph, orientation="ignore")
                # 如果发现环，移除一条边
                for cycle in nx.simple_cycles(digraph):
                    if len(cycle) > 2:
                        digraph.remove_edge(cycle[0], cycle[1])
                        changed = True
                        break
            except nx.NetworkXNoCycle:
                pass
        
        # 确保无环
        while True:
            try:
                # 如果无环，退出循环
                nx.find_cycle(digraph, orientation="ignore")
                # 如果发现环，移除一条边
                for edge in digraph.edges:
                    digraph.remove_edge(*edge)
                    try:
                        nx.find_cycle(digraph, orientation="ignore")
                        # 仍然有环，恢复边
                        digraph.add_edge(*edge)
                    except nx.NetworkXNoCycle:
                        changed = True
                        break
            except nx.NetworkXNoCycle:
                break
    
    def run(self, data: np.ndarray, variables: List[str]) -> nx.DiGraph:
        """运行PC算法
        
        参数:
            data: 观测数据矩阵 (n_samples, n_features)
            variables: 变量名称列表
            
        返回:
            有向无环图 (DAG) 表示因果结构
        """
        # 标准化数据
        scaler = StandardScaler()
        data_normalized = scaler.fit_transform(data)
        
        # 重置缓存
        self.cache = {}
        
        # 步骤1: 估计骨架
        start_time = time.time()
        skeleton, sep_set = self._estimate_skeleton(data_normalized, variables)
        skeleton_time = time.time() - start_time
        
        if self.verbose:
            print(f"骨架估计完成, 耗时 {skeleton_time:.2f}秒")
            print(f"骨架边数: {skeleton.number_of_edges()}")
        
        # 步骤2: 定向边
        start_time = time.time()
        dag = self._orient_edges(skeleton, sep_set)
        orientation_time = time.time() - start_time
        
        if self.verbose:
            print(f"边定向完成, 耗时 {orientation_time:.2f}秒")
            print(f"最终DAG边数: {dag.number_of_edges()}")
        
        # 添加节点标签
        for i, var in enumerate(variables):
            dag.nodes[i]["label"] = var
            
        return dag