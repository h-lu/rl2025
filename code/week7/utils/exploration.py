import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

class EpsilonGreedy:
    """ε-greedy探索策略"""
    def __init__(self, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995):
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
    
    def select_action(self, q_values, action_dim):
        """使用ε-greedy策略选择动作"""
        if random.random() < self.epsilon:
            return random.randrange(action_dim)
        else:
            return q_values.argmax().item()
    
    def update(self):
        """更新epsilon值"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

class BoltzmannExploration:
    """Boltzmann（Softmax）探索策略"""
    def __init__(self, temperature_start=1.0, temperature_end=0.1, temperature_decay=0.995):
        self.temperature = temperature_start
        self.temperature_end = temperature_end
        self.temperature_decay = temperature_decay
    
    def select_action(self, q_values, action_dim):
        """使用Boltzmann策略选择动作"""
        # 应用温度参数
        if isinstance(q_values, torch.Tensor):
            probs = F.softmax(q_values / self.temperature, dim=0).detach().numpy()
        else:
            probs = F.softmax(torch.FloatTensor(q_values) / self.temperature, dim=0).numpy()
        
        # 根据概率采样动作
        return np.random.choice(action_dim, p=probs)
    
    def update(self):
        """更新温度值"""
        self.temperature = max(self.temperature_end, self.temperature * self.temperature_decay)

class NoisyLinear(nn.Module):
    """噪声网络中的噪声线性层"""
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # 可学习的参数
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))
        
        # 初始化参数
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """初始化参数"""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
    def _scale_noise(self, size):
        """产生特殊的缩放噪声"""
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())
    
    def reset_noise(self):
        """重置噪声"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def forward(self, x):
        """前向传播"""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)
