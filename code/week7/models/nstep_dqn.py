import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from utils.buffer import ReplayBuffer
from models.base_dqn import QNetwork

class NStepReplayBuffer:
    """多步学习的经验回放缓冲区"""
    def __init__(self, capacity, n_steps=3, gamma=0.99):
        self.buffer = deque(maxlen=capacity)
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=n_steps)
    
    def _get_n_step_info(self):
        """计算n步回报和下一个状态"""
        reward, next_state, done = self.n_step_buffer[-1][-3:]
        
        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_s, d = transition[-3:]
            
            reward = r + self.gamma * reward * (1 - d)
            next_state, done = (n_s, d) if d else (next_state, done)
        
        return reward, next_state, done
    
    def add(self, state, action, reward, next_state, done):
        """添加经验到n步缓冲区"""
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        # 当n步缓冲区已满，计算n步回报并存储
        if len(self.n_step_buffer) == self.n_steps:
            reward, next_state, done = self._get_n_step_info()
            state, action, _, _, _ = self.n_step_buffer[0]
            self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """从缓冲区采样一批经验"""
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

class NStepDQN:
    """多步学习DQN智能体"""
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=64, target_update=10, hidden_dim=64, n_steps=3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma  # 折扣因子
        self.n_steps = n_steps  # 多步学习步数
        
        # epsilon-greedy策略参数
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # 经验回放 - 使用多步缓冲区
        self.buffer = NStepReplayBuffer(buffer_size, n_steps=n_steps, gamma=gamma)
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
        
        # 计算目标Q值 - 注意：因为n步回报已经在缓冲区计算时使用了gamma^n，这里gamma=1.0
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            # 多步学习时，gamma已经在计算n步回报时考虑了n次
            targets = rewards + (self.gamma ** self.n_steps) * next_q_values * (1 - dones)
        
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