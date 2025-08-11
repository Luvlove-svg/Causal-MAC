import numpy as np
import torch
from segment_tree import SumSegmentTree, MinSegmentTree

class PrioritizedReplayBuffer:
    """优先级经验回放池，支持多智能体经验存储"""
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        """
        Args:
            capacity: 回放池容量
            alpha: 优先级指数 (0-1)
            beta: 重要性采样系数
            beta_increment: beta的增量
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.pos = 0
        self.full = False
        
        # 主存储结构
        self.states = [None] * capacity
        self.actions = [None] * capacity
        self.rewards = [None] * capacity
        self.next_states = [None] * capacity
        self.dones = [None] * capacity
        
        # 优先级存储
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0
        
        # 分段树（高效计算优先级和）
        self.sum_tree = SumSegmentTree(capacity)
        self.min_tree = MinSegmentTree(capacity)
    
    def add(self, states, actions, rewards, next_states, dones):
        """添加经验到回放池"""
        idx = self.pos
        
        # 存储经验
        self.states[idx] = states
        self.actions[idx] = actions
        self.rewards[idx] = rewards
        self.next_states[idx] = next_states
        self.dones[idx] = dones
        
        # 设置初始优先级
        priority = self.max_priority ** self.alpha
        self.sum_tree[idx] = priority
        self.min_tree[idx] = priority
        
        # 更新指针
        self.pos = (self.pos + 1) % self.capacity
        if not self.full and self.pos == 0:
            self.full = True
    
    def sample(self, batch_size):
        """采样一个批次的经验"""
        indices = self._sample_indices(batch_size)
        weights = self._calculate_weights(indices)
        
        # 提取批次数据
        batch = {
            'states': [self.states[i] for i in indices],
            'actions': [self.actions[i] for i in indices],
            'rewards': [self.rewards[i] for i in indices],
            'next_states': [self.next_states[i] for i in indices],
            'dones': [self.dones[i] for i in indices],
            'weights': weights,
            'indices': indices
        }
        
        # 更新beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return batch
    
    def update_priorities(self, indices, priorities):
        """更新样本的优先级"""
        for idx, priority in zip(indices, priorities):
            priority = max(priority, 1e-6)  # 避免零优先级
            
            # 更新优先级
            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha
            self.max_priority = max(self.max_priority, priority)
    
    def _sample_indices(self, batch_size):
        """基于优先级采样索引"""
        indices = []
        total_priority = self.sum_tree.sum()
        segment_size = total_priority / batch_size
        
        for i in range(batch_size):
            min_val = segment_size * i
            max_val = segment_size * (i + 1)
            s = np.random.uniform(min_val, max_val)
            idx = self.sum_tree.find_prefixsum_idx(s)
            indices.append(idx)
        
        return indices
    
    def _calculate_weights(self, indices):
        """计算重要性采样权重"""
        min_priority = self.min_tree.min()
        max_weight = (min_priority * len(self)) ** (-self.beta)
        
        weights = []
        for idx in indices:
            prob = self.sum_tree[idx] / self.sum_tree.sum()
            weight = (prob * len(self)) ** (-self.beta) / max_weight
            weights.append(weight)
        
        return np.array(weights, dtype=np.float32)
    
    def __len__(self):
        return self.capacity if self.full else self.pos

# 辅助数据结构
class SegmentTree:
    """分段树基类"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity)
    
    def _propagate(self, idx, value):
        """从叶子节点向上传播更新"""
        parent = (idx - 1) // 2
        left = 2 * parent + 1
        right = left + 1
        
        self.tree[parent] = self._reduce(self.tree[left], self.tree[right])
        if parent != 0:
            self._propagate(parent, value)
    
    def _reduce(self, a, b):
        """在子类中实现具体操作"""
        raise NotImplementedError
    
    def __setitem__(self, idx, value):
        """设置叶子节点的值"""
        idx += self.capacity - 1
        self.tree[idx] = value
        self._propagate(idx, value)
    
    def __getitem__(self, idx):
        """获取叶子节点的值"""
        return self.tree[self.capacity - 1 + idx]

class SumSegmentTree(SegmentTree):
    """求和分段树"""
    def __init__(self, capacity):
        super().__init__(capacity)
        self.tree = np.zeros(2 * capacity)
    
    def _reduce(self, a, b):
        return a + b
    
    def sum(self, start=0, end=None):
        """计算区间和"""
        if end is None:
            end = self.capacity
        return self._range_query(start, end, 0, 0, self.capacity - 1)
    
    def _range_query(self, start, end, node, node_start, node_end):
        """递归区间查询"""
        if start <= node_start and node_end <= end:
            return self.tree[node]
        
        if end < node_start or node_end < start:
            return 0
        
        mid = (node_start + node_end) // 2
        return (self._range_query(start, end, 2*node+1, node_start, mid) + 
                self._range_query(start, end, 2*node+2, mid+1, node_end))
    
    def find_prefixsum_idx(self, prefixsum):
        """找到满足前缀和的索引"""
        idx = 1  # 从根节点开始
        while idx < self.capacity - 1:
            if self.tree[2*idx+1] > prefixsum:
                idx = 2*idx+1
            else:
                prefixsum -= self.tree[2*idx+1]
                idx = 2*idx+2
        return idx - (self.capacity - 1)

class MinSegmentTree(SegmentTree):
    """最小值分段树"""
    def __init__(self, capacity):
        super().__init__(capacity)
        self.tree = np.full(2 * capacity, float('inf'))
    
    def _reduce(self, a, b):
        return min(a, b)
    
    def min(self, start=0, end=None):
        """查询最小值"""
        if end is None:
            end = self.capacity
        return self._range_query(start, end, 0, 0, self.capacity - 1)
    
    def _range_query(self, start, end, node, node_start, node_end):
        """递归区间查询"""
        if start <= node_start and node_end <= end:
            return self.tree[node]
        
        if end < node_start or node_end < start:
            return float('inf')
        
        mid = (node_start + node_end) // 2
        return min(self._range_query(start, end, 2*node+1, node_start, mid),
                   self._range_query(start, end, 2*node+2, mid+1, node_end))