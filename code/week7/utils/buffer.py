import numpy as np
import random
from collections import deque
import torch

class ReplayBuffer:
    """标准经验回放缓冲区"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)

class SumTree:
    """用于优先经验回放的SumTree数据结构"""
    def __init__(self, capacity):
        self.capacity = capacity  # 叶节点数量
        self.tree = np.zeros(2 * capacity - 1)  # 总节点数量
        self.data = np.zeros(capacity, dtype=object)  # 存储经验数据
        self.data_pointer = 0  # 当前数据指针
        self.size = 0  # 当前存储的经验数量
    
    def add(self, priority, data):
        """添加新经验"""
        # 找到要更新的叶节点索引
        tree_idx = self.data_pointer + self.capacity - 1
        
        # 存储数据
        self.data[self.data_pointer] = data
        
        # 更新优先级
        self.update(tree_idx, priority)
        
        # 更新指针和大小
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def update(self, tree_idx, priority):
        """更新优先级"""
        # 计算变化量
        change = priority - self.tree[tree_idx]
        
        # 更新叶节点
        self.tree[tree_idx] = priority
        
        # 传播变化到根节点
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change
    
    def get_leaf(self, value):
        """根据给定的累积和找到叶节点"""
        parent_idx = 0
        
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            
            # 如果到达叶节点，则返回
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            
            # 根据累积和来选择子树
            if value <= self.tree[left_child_idx]:
                parent_idx = left_child_idx
            else:
                value -= self.tree[left_child_idx]
                parent_idx = right_child_idx
        
        data_idx = leaf_idx - (self.capacity - 1)
        
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]
    
    def total_priority(self):
        """返回根节点值（所有优先级之和）"""
        return self.tree[0]

class PrioritizedReplayBuffer:
    """优先经验回放缓冲区"""
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha  # 控制优先级程度的参数
        self.beta = beta_start  # 用于重要性采样的参数
        self.beta_increment = (1 - beta_start) / beta_frames
        self.max_priority = 1.0  # 用于新经验的最大优先级
    
    def add(self, state, action, reward, next_state, done):
        """添加新经验到缓冲区"""
        experience = (state, action, reward, next_state, done)
        # 新经验使用最大优先级
        self.tree.add(self.max_priority ** self.alpha, experience)
    
    def sample(self, batch_size):
        """从缓冲区采样一批经验"""
        batch = []
        indices = []
        priorities = []
        
        # 计算每段的大小
        segment = self.tree.total_priority() / batch_size
        
        # 增加beta值
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # 从每段中采样
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            value = np.random.uniform(a, b)
            
            idx, priority, data = self.tree.get_leaf(value)
            
            indices.append(idx)
            priorities.append(priority)
            batch.append(data)
        
        # 计算重要性采样权重
        sampling_probabilities = np.array(priorities) / self.tree.total_priority()
        weights = (self.tree.size * sampling_probabilities) ** (-self.beta)
        weights /= weights.max()  # 归一化权重
        
        # 解包经验
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            indices,
            np.array(weights, dtype=np.float32)
        )
    
    def update_priorities(self, indices, priorities):
        """更新经验的优先级"""
        for idx, priority in zip(indices, priorities):
            # 添加小的常数防止优先级为0
            priority = max(priority, 1e-5)
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority ** self.alpha)
    
    def __len__(self):
        """返回缓冲区中经验的数量"""
        return self.tree.size
