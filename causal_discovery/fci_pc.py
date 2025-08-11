# causal_discovery/fci_pc.py
import numpy as np
import networkx as nx
from .pc_algorithm import PCAlgorithm
from typing import Set, Dict, Tuple, List, Optional
from joblib import Parallel, delayed

class FCIPCAlgorithm(PCAlgorithm):
    """FCI-PC算法实现，处理潜在混杂变量"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.max_path_length = config.get("max_path_length", 5)
        self.parallel_orient = config.get("parallel_orient", True)
    
    def _estimate_skeleton(self, data: np.ndarray, variables: List[str]) -> nx.Graph:
        """估计骨架图（与PC相同）"""
        return super()._estimate_skeleton(data, variables)
    
    def _orient_edges(self, graph: nx.Graph, sep_set: Dict[Tuple[int, int], Set[int]]) -> nx.DiGraph:
        """定向边（扩展FCI规则）"""
        # 创建部分有向图 (PAG)
        pag = graph.copy()
        
        # 初始定向 (v-结构)
        self._orient_v_structures(pag, sep_set)
        
        # 应用FCI定向规则
        changed = True
        while changed:
            changed = False
            
            # 规则1: 如果 A → B o-o C 且 A 和 C 不相邻，则定向为 B → C
            changed |= self._apply_rule1(pag)
            
            # 规则2: 避免新环
            changed |= self._avoid_cycles(pag)
            
            # 规则3: 如果 A → B → C 且 A o-o C，则定向为 A → C
            changed |= self._apply_rule3(pag)
            
            # 规则4: 如果 A → B ← C 且 A o-o D o-o C 且 A 和 C 不相邻，则定向为 D → B
            changed |= self._apply_rule4(pag)
        
        return pag
    
    def _orient_v_structures(self, pag: nx.Graph, sep_set: Dict[Tuple[int, int], Set[int]]):
        """定向v-结构"""
        for pair in sep_set:
            i, j = pair
            for k in set(pag.nodes) - {i, j}:
                if pag.has_edge(i, k) and pag.has_edge(j, k):
                    if k not in sep_set[(i, j)]:
                        # 定向为 i → k ← j
                        if pag.edges[(i, k)].get('edge_type', None) != 'confounder':
                            pag.add_edge(i, k, edge_type='directed')
                        if pag.edges[(j, k)].get('edge_type', None) != 'confounder':
                            pag.add_edge(j, k, edge_type='directed')
    
    def _apply_rule1(self, pag: nx.Graph) -> bool:
        """应用FCI规则1"""
        changed = False
        for b in pag.nodes:
            # 查找 A → B
            a_candidates = [n for n in pag.predecessors(b) 
                           if pag.edges.get((n, b), {}).get('edge_type') == 'directed']
            
            for a in a_candidates:
                # 查找 B o-o C
                c_candidates = [n for n in pag.neighbors(b) 
                               if n != a and pag.edges.get((b, n), {}).get('edge_type') == 'undirected']
                
                for c in c_candidates:
                    # 检查 A 和 C 是否不相邻
                    if not pag.has_edge(a, c) and not pag.has_edge(c, a):
                        # 定向为 B → C
                        pag.edges[(b, c)]['edge_type'] = 'directed'
                        changed = True
        return changed
    
    def _apply_rule3(self, pag: nx.Graph) -> bool:
        """应用FCI规则3"""
        changed = False
        for c in pag.nodes:
            # 查找 A → B → C
            b_candidates = [n for n in pag.predecessors(c) 
                          if pag.edges.get((n, c), {}).get('edge_type') == 'directed']
            
            for b in b_candidates:
                a_candidates = [n for n in pag.predecessors(b) 
                              if pag.edges.get((n, b), {}).get('edge_type') == 'directed']
                
                for a in a_candidates:
                    # 检查 A o-o C
                    if pag.has_edge(a, c) and pag.edges[(a, c)].get('edge_type') == 'undirected':
                        # 定向为 A → C
                        pag.edges[(a, c)]['edge_type'] = 'directed'
                        changed = True
        return changed
    
    def _apply_rule4(self, pag: nx.Graph) -> bool:
        """应用FCI规则4（检测潜在混杂）"""
        changed = False
        for b in pag.nodes:
            # 查找 A → B ← C
            a_candidates = [n for n in pag.predecessors(b) 
                          if pag.edges.get((n, b), {}).get('edge_type') == 'directed']
            c_candidates = [n for n in pag.predecessors(b) 
                          if n not in a_candidates and pag.edges.get((n, b), {}).get('edge_type') == 'directed']
            
            for a in a_candidates:
                for c in c_candidates:
                    # 查找 A o-o D o-o C
                    d_candidates = set(pag.neighbors(a)) & set(pag.neighbors(c))
                    d_candidates = [d for d in d_candidates 
                                   if pag.edges[(a, d)].get('edge_type') == 'undirected' and 
                                   pag.edges[(c, d)].get('edge_type') == 'undirected']
                    
                    for d in d_candidates:
                        # 定向为 D → B
                        if pag.edges[(d, b)].get('edge_type') == 'undirected':
                            pag.edges[(d, b)]['edge_type'] = 'directed'
                            changed = True
        return changed
    
    def _avoid_cycles(self, pag: nx.Graph) -> bool:
        """避免引入有向环"""
        changed = False
        
        # 尝试检测环
        try:
            # 创建一个只有有向边的子图
            directed_edges = [(u, v) for u, v, data in pag.edges(data=True) 
                             if data.get('edge_type') == 'directed']
            directed_graph = nx.DiGraph(directed_edges)
            
            # 查找环
            cycle = next(nx.simple_cycles(directed_graph), None)
            if cycle:
                # 找到环，移除一条边
                u, v = cycle[0], cycle[1]
                if pag.has_edge(u, v):
                    pag.remove_edge(u, v)
                    changed = True
        except (nx.NetworkXNoCycle, StopIteration):
            pass
        
        return changed
    
    def to_dag(self, pag: nx.Graph) -> nx.DiGraph:
        """将PAG转换为DAG（通过忽略不确定的边）"""
        dag = nx.DiGraph()
        dag.add_nodes_from(pag.nodes(data=True))
        
        for u, v, data in pag.edges(data=True):
            if data.get('edge_type') == 'directed':
                dag.add_edge(u, v)
            elif data.get('edge_type') == 'undirected':
                # 随机定向（或基于其他启发式方法）
                dag.add_edge(u, v)  # 简单实现，实际中应使用更智能的方法
        
        # 确保无环
        while True:
            try:
                nx.find_cycle(dag)
                # 移除一条随机边
                edge = list(dag.edges)[0]
                dag.remove_edge(*edge)
            except nx.NetworkXNoCycle:
                break
        
        return dag