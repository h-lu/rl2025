import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from utils.buffer import ReplayBuffer
from models.base_dqn import QNetwork

class DoubleDQN:
    """Double DQN智能体"""
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=64, target_update=10, hidden_dim=64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma  # 折扣因子
        
        # epsilon-greedy策略参数
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # 经验回放
        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        
        # Q网络
        self.q_network = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_network = QNetwork(state_dim, action_dim, hidden_dim)
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
        
        # 从经验回放缓冲区采样
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        # 转换为张量
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # 计算当前Q值
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN: 使用在线网络选择动作，目标网络评估价值
        with torch.no_grad():
            # 使用在线网络选择下一个状态的最佳动作
            next_q_values_online = self.q_network(next_states)
            best_actions = next_q_values_online.argmax(1, keepdim=True)
            
            # 使用目标网络评估所选动作的Q值
            next_q_values_target = self.target_network(next_states).gather(1, best_actions).squeeze(1)
            
            # 计算目标值
            targets = rewards + self.gamma * next_q_values_target * (1 - dones)
        
        # 计算损失
        loss = nn.MSELoss()(q_values, targets)
        
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
