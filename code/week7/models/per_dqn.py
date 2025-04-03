import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from utils.buffer import PrioritizedReplayBuffer
from models.base_dqn import QNetwork

class PERDQN:
    """优先经验回放DQN智能体"""
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=64, target_update=10,
                 alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma  # 折扣因子
        
        # epsilon-greedy策略参数
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # 优先经验回放
        self.buffer = PrioritizedReplayBuffer(buffer_size, alpha, beta_start, beta_frames)
        self.batch_size = batch_size
        
        # Q网络
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        # 初始时目标网络与主网络参数相同
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # 目标网络更新频率
        self.target_update = target_update
        self.update_counter = 0
    
    def select_action(self, state, epsilon=None):
        """使用epsilon-greedy策略选择动作"""
        if epsilon is None:
            epsilon = self.epsilon
        
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        else:
            state = torch.FloatTensor(state)
            with torch.no_grad():
                q_values = self.q_network(state)
            return q_values.argmax().item()
    
    def update(self):
        """更新模型参数"""
        if len(self.buffer) < self.batch_size:
            return 0.0
        
        # 从优先经验回放缓冲区采样
        states, actions, rewards, next_states, dones, indices, weights = self.buffer.sample(self.batch_size)
        
        # 转换为张量
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        weights = torch.FloatTensor(weights)
        
        # 计算当前Q值
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            targets = rewards + self.gamma * next_q_values * (1 - dones)
        
        # 计算TD误差
        td_errors = torch.abs(q_values - targets).detach().cpu().numpy()
        
        # 更新优先级
        self.buffer.update_priorities(indices, td_errors)
        
        # 计算带权重的损失
        loss = (weights * nn.MSELoss(reduction='none')(q_values, targets)).mean()
        
        # 梯度下降
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # 更新目标网络
        self.update_counter += 1
        if self.update_counter % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
