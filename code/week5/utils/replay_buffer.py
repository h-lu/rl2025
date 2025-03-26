"""
经验回放缓冲区实现

经验回放 (Experience Replay) 是DQN算法的关键组成部分，
它打破了样本间的时序相关性，提高了学习的稳定性和样本利用率。
"""

import numpy as np
import random
from collections import deque

class ReplayBuffer:
    """
    经验回放缓冲区类
    
    用于存储和采样智能体与环境交互的经验数据(state, action, reward, next_state, done)
    """
    
    def __init__(self, capacity):
        """
        初始化经验回放缓冲区
        
        参数:
            capacity (int): 缓冲区容量，存储的最大经验数量
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """
        添加经验到缓冲区
        
        参数:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否为终止状态
        """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """
        从缓冲区中随机采样一批经验数据
        
        参数:
            batch_size (int): 批次大小
            
        返回:
            tuple: 包含numpy数组形式的states, actions, rewards, next_states, dones
        """
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
            
        # 随机采样
        experiences = random.sample(self.buffer, batch_size)
        
        # 解包经验数据
        states, actions, rewards, next_states, dones = zip(*experiences)
        
        # 转换为numpy数组
        return (
            np.array(states, dtype=np.float32), 
            np.array(actions, dtype=np.int32), 
            np.array(rewards, dtype=np.float32), 
            np.array(next_states, dtype=np.float32), 
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        """
        获取缓冲区中的经验数量
        
        返回:
            int: 缓冲区中的经验数量
        """
        return len(self.buffer)
    
    def clear(self):
        """
        清空缓冲区
        """
        self.buffer.clear()
    
    def is_ready(self, batch_size):
        """
        检查缓冲区是否有足够的经验可供采样
        
        参数:
            batch_size (int): 批次大小
            
        返回:
            bool: 如果缓冲区中的经验数量大于等于batch_size，则返回True
        """
        return len(self.buffer) >= batch_size 