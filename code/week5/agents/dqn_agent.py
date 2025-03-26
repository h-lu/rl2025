"""
深度Q网络(DQN)代理实现

该模块实现了完整的DQN代理，包括:
1. 神经网络模型
2. ε-贪婪策略
3. 经验回放
4. 目标网络
5. 学习算法
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.replay_buffer import ReplayBuffer
import config

# 设置随机种子
random.seed(config.SEED)
np.random.seed(config.SEED)
tf.random.set_seed(config.SEED)

def create_dqn_model(state_size, action_size, hidden_size=config.HIDDEN_SIZE):
    """
    创建DQN神经网络模型
    
    参数:
        state_size (int): 状态空间维度
        action_size (int): 动作空间维度
        hidden_size (int): 隐藏层大小
        
    返回:
        keras.Model: 构建好的DQN模型
    """
    model = keras.Sequential([
        layers.Dense(hidden_size, activation='relu', input_shape=(state_size,)),
        layers.Dense(hidden_size, activation='relu'),
        layers.Dense(action_size, activation='linear')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
                 loss='mse')
    return model

class DQNAgent:
    """
    DQN代理类
    
    实现了深度Q网络算法的智能体，用于在强化学习环境中学习和决策。
    """
    
    def __init__(self, state_size, action_size, 
                 gamma=config.GAMMA, 
                 epsilon_start=config.EPSILON_START, 
                 epsilon_end=config.EPSILON_END, 
                 epsilon_decay=config.EPSILON_DECAY, 
                 learning_rate=config.LEARNING_RATE, 
                 buffer_size=config.BUFFER_SIZE, 
                 batch_size=config.BATCH_SIZE, 
                 update_target_every=config.UPDATE_TARGET_EVERY):
        """
        初始化DQN代理
        
        参数:
            state_size (int): 状态空间维度
            action_size (int): 动作空间维度
            gamma (float): 折扣因子
            epsilon_start (float): 初始探索率
            epsilon_end (float): 最小探索率
            epsilon_decay (float): 探索率衰减系数
            learning_rate (float): 学习率
            buffer_size (int): 经验回放缓冲区大小
            batch_size (int): 训练批次大小
            update_target_every (int): 目标网络更新频率
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.learning_rate = learning_rate
        
        # 创建Q网络和目标网络
        self.q_network = create_dqn_model(state_size, action_size)
        self.target_network = create_dqn_model(state_size, action_size)
        self.target_network.set_weights(self.q_network.get_weights())
        
        # 创建经验回放缓冲区
        self.memory = ReplayBuffer(buffer_size)
        
        # 用于记录学习步数
        self.t_step = 0
        
        # 用于记录训练信息
        self.loss_history = []
    
    def step(self, state, action, reward, next_state, done):
        """
        执行学习步骤
        
        存储经验到缓冲区并在适当的时候进行学习
        
        参数:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否为终止状态
        
        返回:
            float or None: 如果执行了学习，返回损失值；否则返回None
        """
        # 将经验添加到缓冲区
        self.memory.add(state, action, reward, next_state, done)
        
        # 更新时间步
        self.t_step += 1
        
        # 如果缓冲区中有足够的样本，执行学习
        loss = None
        if self.memory.is_ready(self.batch_size) and self.t_step % 4 == 0:
            experiences = self.memory.sample(self.batch_size)
            loss = self.learn(experiences)
            
            # 定期更新目标网络
            if self.t_step % self.update_target_every == 0:
                self.target_network.set_weights(self.q_network.get_weights())
        
        return loss
    
    def act(self, state, eval_mode=False):
        """
        根据当前状态选择动作
        
        使用ε-贪婪策略在探索（随机动作）和开发（最优动作）之间进行平衡
        
        参数:
            state: 当前状态
            eval_mode (bool): 是否为评估模式，True时始终选择最优动作
            
        返回:
            int: 选择的动作
        """
        # 将状态转换为批次格式
        state = np.reshape(state, [1, self.state_size])
        
        # 评估模式或使用ε-贪婪策略选择动作
        if eval_mode or random.random() > self.epsilon:
            # 开发：选择Q值最大的动作
            action_values = self.q_network.predict(state, verbose=0)
            return np.argmax(action_values[0])
        else:
            # 探索：随机选择动作
            return random.choice(np.arange(self.action_size))
    
    def learn(self, experiences):
        """
        从经验批次中学习
        
        参数:
            experiences: 包含(states, actions, rewards, next_states, dones)的元组
            
        返回:
            float: 训练损失
        """
        states, actions, rewards, next_states, dones = experiences
        
        # 从目标网络中获取下一个状态的最大Q值
        target_q_values = self.target_network.predict(next_states, verbose=0)
        max_target_q = np.max(target_q_values, axis=1)
        
        # 计算目标Q值
        targets = rewards + (self.gamma * max_target_q * (1 - dones))
        
        # 获取当前预测的Q值并更新目标
        target_f = self.q_network.predict(states, verbose=0)
        
        # 只更新选择的动作对应的Q值
        for i, action in enumerate(actions):
            target_f[i][action] = targets[i]
        
        # 训练Q网络
        history = self.q_network.fit(states, target_f, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        self.loss_history.append(loss)
        
        # 更新ε值，逐渐减少探索
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss
    
    def save(self, filepath):
        """
        保存模型
        
        参数:
            filepath (str): 保存模型的文件路径
        """
        self.q_network.save_weights(filepath)
    
    def load(self, filepath):
        """
        加载模型
        
        参数:
            filepath (str): 加载模型的文件路径
        """
        self.q_network.load_weights(filepath)
        self.target_network.set_weights(self.q_network.get_weights())
    
    def get_epsilon(self):
        """
        获取当前ε值
        
        返回:
            float: 当前ε值
        """
        return self.epsilon
    
    def get_loss_history(self):
        """
        获取训练损失历史
        
        返回:
            list: 损失历史
        """
        return self.loss_history 