"""
Rainbow DQN实现

该文件包含Rainbow DQN的核心组件：
1. Double DQN: 解决Q值过高估计问题
2. Dueling DQN: 分离状态价值和动作优势
3. 优先经验回放: 基于TD误差优先采样重要经验
4. 多步学习: 使用n步回报加速价值传播
5. Noisy Networks: 替代ε-greedy实现自适应探索

注意：此实现不包含Rainbow的第6个组件：分布式RL
"""

import gymnasium as gym
import numpy as np
import random
import math
from collections import deque
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from IPython.display import clear_output
import os
import platform # 导入 platform 模块

# 设置随机种子以获得可重复的结果
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 配置 TensorFlow 使用 GPU 并允许内存增长
print("Configuring TensorFlow for GPU...")
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 设置内存增长，避免一次性占用所有显存
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs detected and configured.")
    except RuntimeError as e:
        # 异常处理
        print(f"Error during GPU configuration: {e}")
else:
    print("No GPU detected, running on CPU.")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # 显式禁用GPU

# 检查是否可用GPU
# print(f"TensorFlow版本: {tf.__version__}") # 已在上方配置时打印
# print(f"使用GPU: {len(tf.config.list_physical_devices('GPU')) > 0}")

###################
# 网络模型定义部分 #
###################

class NoisyDense(layers.Layer):
    """
    噪声网络层实现 (Noisy Networks)
    参考: https://arxiv.org/abs/1706.10295
    """
    def __init__(self, units, activation=None, sigma_init=0.017):
        super(NoisyDense, self).__init__()
        self.units = units
        self.activation = activation
        self.sigma_init = sigma_init
        
    def build(self, input_shape):
        self.input_dim = input_shape[-1]
        
        # 初始化均值权重和偏置（常规参数）
        mu_init = tf.random_uniform_initializer(minval=-1/np.sqrt(self.input_dim), 
                                              maxval=1/np.sqrt(self.input_dim))
        self.weight_mu = self.add_weight(
            name="weight_mu",
            shape=(self.input_dim, self.units),
            initializer=mu_init,
            trainable=True
        )
        self.bias_mu = self.add_weight(
            name="bias_mu",
            shape=(self.units,),
            initializer=mu_init,
            trainable=True
        )
        
        # 初始化标准差权重和偏置（噪声参数）
        sigma_init = tf.constant_initializer(self.sigma_init / np.sqrt(self.input_dim))
        self.weight_sigma = self.add_weight(
            name="weight_sigma",
            shape=(self.input_dim, self.units),
            initializer=sigma_init,
            trainable=True
        )
        self.bias_sigma = self.add_weight(
            name="bias_sigma",
            shape=(self.units,),
            initializer=sigma_init,
            trainable=True
        )
    
    def call(self, inputs, training=True):
        if training:
            # 在训练模式下使用噪声
            # 生成ε噪声
            eps_in = tf.random.normal((self.input_dim,))
            eps_out = tf.random.normal((self.units,))
            
            # 噪声项计算（因子化实现以提高效率）
            weight_eps = tf.tensordot(
                tf.multiply(tf.sign(eps_in), tf.sqrt(tf.abs(eps_in))),
                tf.multiply(tf.sign(eps_out), tf.sqrt(tf.abs(eps_out))),
                axes=0
            )
            bias_eps = tf.multiply(tf.sign(eps_out), tf.sqrt(tf.abs(eps_out)))
            
            # 噪声权重和偏置
            weights = self.weight_mu + self.weight_sigma * weight_eps
            biases = self.bias_mu + self.bias_sigma * bias_eps
        else:
            # 评估模式，使用均值参数
            weights = self.weight_mu
            biases = self.bias_mu
            
        output = tf.matmul(inputs, weights) + biases
        
        if self.activation is not None:
            output = self.activation(output)
            
        return output


def create_dueling_dqn_model(state_size, action_size, hidden_size=128, noisy=False):
    """
    创建Dueling DQN模型（可选Noisy Networks）
    
    参数:
    - state_size: 状态空间维度
    - action_size: 动作空间维度
    - hidden_size: 隐藏层大小
    - noisy: 是否使用噪声网络层
    
    返回:
    - Keras模型
    """
    inputs = layers.Input(shape=(state_size,))
    
    # 特征提取层
    if noisy:
        x = NoisyDense(hidden_size, activation='relu')(inputs)
        features = NoisyDense(hidden_size, activation='relu')(x)
    else:
        x = layers.Dense(hidden_size, activation='relu')(inputs)
        features = layers.Dense(hidden_size, activation='relu')(x)
    
    # 价值流
    if noisy:
        value_stream = NoisyDense(hidden_size//2, activation='relu')(features)
        value = NoisyDense(1)(value_stream)
    else:
        value_stream = layers.Dense(hidden_size//2, activation='relu')(features)
        value = layers.Dense(1)(value_stream)
    
    # 优势流
    if noisy:
        advantage_stream = NoisyDense(hidden_size//2, activation='relu')(features)
        advantage = NoisyDense(action_size)(advantage_stream)
    else:
        advantage_stream = layers.Dense(hidden_size//2, activation='relu')(features)
        advantage = layers.Dense(action_size)(advantage_stream)
    
    # 组合层
    # Q(s,a) = V(s) + (A(s,a) - mean(A(s,:))
    value_tiled = tf.tile(value, [1, action_size])
    advantage_mean = tf.reduce_mean(advantage, axis=1, keepdims=True)
    advantage_centered = advantage - tf.tile(advantage_mean, [1, action_size])
    q_values = value_tiled + advantage_centered
    
    model = keras.Model(inputs=inputs, outputs=q_values)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                 loss='mse')
    
    return model


def create_standard_dqn_model(state_size, action_size, hidden_size=128, noisy=False):
    """
    创建标准DQN模型（可选Noisy Networks）
    
    参数:
    - state_size: 状态空间维度
    - action_size: 动作空间维度
    - hidden_size: 隐藏层大小
    - noisy: 是否使用噪声网络层
    
    返回:
    - Keras模型
    """
    inputs = layers.Input(shape=(state_size,))
    
    if noisy:
        x = NoisyDense(hidden_size, activation='relu')(inputs)
        x = NoisyDense(hidden_size, activation='relu')(x)
        outputs = NoisyDense(action_size)(x)
    else:
        x = layers.Dense(hidden_size, activation='relu')(inputs)
        x = layers.Dense(hidden_size, activation='relu')(x)
        outputs = layers.Dense(action_size)(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                 loss='mse')
    
    return model


#######################
# 经验回放缓冲区部分   #
#######################

class ReplayBuffer:
    """标准经验回放缓冲区"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)
        
        return (np.array(states), 
                np.array(actions), 
                np.array(rewards), 
                np.array(next_states), 
                np.array(dones, dtype=np.float32))
    
    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """优先经验回放缓冲区"""
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = capacity
        self.alpha = alpha  # 优先级重要性因子，控制采样倾向性
        self.beta = beta_start  # 重要性采样权重因子，用于修正采样偏差
        self.beta_increment = (1.0 - beta_start) / beta_frames
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0  # 初始化最大优先级
    
    def add(self, state, action, reward, next_state, done):
        # 新经验的优先级设为最大优先级
        max_priority = self.max_priority
        
        # 添加经验
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        # 更新优先级
        self.priorities[self.position] = max_priority
        
        # 更新位置
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        # 如果缓冲区大小小于批次大小，无法采样
        if len(self.buffer) < batch_size:
            raise ValueError("Buffer contains fewer than batch_size experiences")
            
        # 计算采样概率
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]
            
        # 按优先级的alpha次幂计算采样概率
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # 采样索引
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # 计算重要性采样权重
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # 归一化权重
        
        # 提升beta值，随着训练进行逐渐接近1
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # 获取样本
        samples = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*samples)
        
        return (np.array(states), 
                np.array(actions), 
                np.array(rewards), 
                np.array(next_states), 
                np.array(dones, dtype=np.float32),
                indices,
                np.array(weights, dtype=np.float32))
    
    def update_priorities(self, indices, td_errors):
        """更新优先级（使用TD误差的绝对值）"""
        for idx, td_error in zip(indices, td_errors):
            priority = abs(td_error) + 1e-6  # 添加小常数避免优先级为0
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)


class MultiStepReplayBuffer:
    """多步学习回放缓冲区"""
    def __init__(self, capacity, n_steps=3, gamma=0.99):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=n_steps)
        
    def _get_n_step_info(self):
        """计算n步回报和最终状态"""
        reward = 0
        for i, (_, _, r, _, _) in enumerate(self.n_step_buffer):
            reward += r * (self.gamma ** i)
            
        state, action, _, _, _ = self.n_step_buffer[0]
        _, _, _, next_state, done = self.n_step_buffer[-1]
        
        return state, action, reward, next_state, done
        
    def add(self, state, action, reward, next_state, done):
        # 添加到n步缓冲区
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        # 如果n步缓冲区未满，且当前状态不是终止状态，则不添加到主缓冲区
        if len(self.n_step_buffer) < self.n_steps and not done:
            return
            
        # 计算n步回报和状态
        state, action, reward, next_state, done = self._get_n_step_info()
        
        # 添加到主缓冲区
        self.buffer.append((state, action, reward, next_state, done))
        
        # 如果当前是终止状态，清空n步缓冲区
        if done:
            self.n_step_buffer.clear()
            
    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)
        
        return (np.array(states), 
                np.array(actions), 
                np.array(rewards), 
                np.array(next_states), 
                np.array(dones, dtype=np.float32))
    
    def __len__(self):
        return len(self.buffer)


class PrioritizedMultiStepBuffer:
    """结合优先经验回放和多步学习的缓冲区"""
    def __init__(self, capacity, n_steps=3, alpha=0.6, beta_start=0.4, 
                 beta_frames=100000, gamma=0.99):
        self.capacity = capacity
        self.n_steps = n_steps
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta_start
        self.beta_increment = (1.0 - beta_start) / beta_frames
        
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.n_step_buffer = deque(maxlen=n_steps)
        self.max_priority = 1.0
        
    def _get_n_step_info(self):
        """计算n步回报和最终状态"""
        reward = 0
        for i, (_, _, r, _, _) in enumerate(self.n_step_buffer):
            reward += r * (self.gamma ** i)
            
        state, action, _, _, _ = self.n_step_buffer[0]
        _, _, _, next_state, done = self.n_step_buffer[-1]
        
        return state, action, reward, next_state, done
        
    def add(self, state, action, reward, next_state, done):
        # 添加到n步缓冲区
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        # 如果n步缓冲区未满，且当前状态不是终止状态，则不添加到主缓冲区
        if len(self.n_step_buffer) < self.n_steps and not done:
            return
            
        # 计算n步回报和状态
        state, action, reward, next_state, done = self._get_n_step_info()
        
        # 添加到主缓冲区，使用最大优先级
        max_priority = self.max_priority
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
            
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        
        # 如果当前是终止状态，清空n步缓冲区
        if done:
            self.n_step_buffer.clear()
            
    def sample(self, batch_size):
        # 如果缓冲区大小小于批次大小，无法采样
        if len(self.buffer) < batch_size:
            raise ValueError("Buffer contains fewer than batch_size experiences")
            
        # 计算采样概率
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]
            
        # 按优先级的alpha次幂计算采样概率
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # 采样索引
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # 计算重要性采样权重
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # 归一化权重
        
        # 提升beta值，随着训练进行逐渐接近1
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # 获取样本
        samples = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*samples)
        
        return (np.array(states), 
                np.array(actions), 
                np.array(rewards), 
                np.array(next_states), 
                np.array(dones, dtype=np.float32),
                indices,
                np.array(weights, dtype=np.float32))
    
    def update_priorities(self, indices, td_errors):
        """更新优先级（使用TD误差的绝对值）"""
        for idx, td_error in zip(indices, td_errors):
            priority = abs(td_error) + 1e-6  # 添加小常数避免优先级为0
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)

        
###################
# 智能体实现部分   #
###################

class RainbowDQNAgent:
    """
    Rainbow DQN智能体
    
    实现多种DQN改进:
    - Double DQN: 解决Q值过高估计问题
    - Dueling DQN: 分离状态价值和动作优势
    - 优先经验回放: 基于TD误差优先采样重要经验
    - 多步学习: 使用n步回报加速价值传播
    - Noisy Networks: 替代ε-greedy实现自适应探索
    """
    def __init__(self, state_size, action_size, 
                 # 算法配置
                 use_double_dqn=True,
                 use_dueling_dqn=True,
                 use_prioritized_replay=True,
                 use_n_step=True,
                 use_noisy_nets=True,
                 # 超参数
                 learning_rate=0.001,
                 gamma=0.99,
                 buffer_size=50000,
                 batch_size=64,
                 update_target_every=1000,
                 hidden_size=128,
                 # 如果不使用Noisy Nets，则使用epsilon-greedy
                 epsilon_start=1.0, 
                 epsilon_end=0.01,
                 epsilon_decay=0.995,
                 # 优先经验回放参数
                 alpha=0.6,
                 beta_start=0.4,
                 beta_frames=100000,
                 # 多步学习参数
                 n_steps=3):
        
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        
        # 配置标志
        self.use_double_dqn = use_double_dqn
        self.use_dueling_dqn = use_dueling_dqn
        self.use_prioritized_replay = use_prioritized_replay
        self.use_n_step = use_n_step
        self.use_noisy_nets = use_noisy_nets
        self.n_steps = n_steps
        
        # 如果不使用噪声网络，则配置epsilon-greedy探索
        if not self.use_noisy_nets:
            self.epsilon = epsilon_start
            self.epsilon_end = epsilon_end
            self.epsilon_decay = epsilon_decay
        
        # 创建网络模型
        if self.use_dueling_dqn:
            self.q_network = create_dueling_dqn_model(
                state_size, action_size, hidden_size, self.use_noisy_nets)
            self.target_network = create_dueling_dqn_model(
                state_size, action_size, hidden_size, self.use_noisy_nets)
        else:
            self.q_network = create_standard_dqn_model(
                state_size, action_size, hidden_size, self.use_noisy_nets)
            self.target_network = create_standard_dqn_model(
                state_size, action_size, hidden_size, self.use_noisy_nets)
            
        # 初始同步目标网络
        self.target_network.set_weights(self.q_network.get_weights())
        
        # 创建经验回放缓冲区
        if self.use_prioritized_replay and self.use_n_step:
            self.memory = PrioritizedMultiStepBuffer(
                buffer_size, n_steps, alpha, beta_start, beta_frames, gamma)
        elif self.use_prioritized_replay:
            self.memory = PrioritizedReplayBuffer(
                buffer_size, alpha, beta_start, beta_frames)
        elif self.use_n_step:
            self.memory = MultiStepReplayBuffer(
                buffer_size, n_steps, gamma)
        else:
            self.memory = ReplayBuffer(buffer_size)
            
        # 记录学习步数
        self.t_step = 0
        
        # 记录训练信息
        self.loss_history = []
        
        # 打印配置信息
        print(f"Rainbow DQN配置:")
        print(f"- Double DQN: {self.use_double_dqn}")
        print(f"- Dueling DQN: {self.use_dueling_dqn}")
        print(f"- 优先经验回放: {self.use_prioritized_replay}")
        print(f"- 多步学习 (n={self.n_steps}): {self.use_n_step}")
        print(f"- 噪声网络: {self.use_noisy_nets}")
    
    def step(self, state, action, reward, next_state, done):
        """存储经验并在适当的时候学习"""
        # 添加经验到缓冲区
        self.memory.add(state, action, reward, next_state, done)
        
        # 更新时间步
        self.t_step += 1
        
        # 如果缓冲区中有足够样本且到了学习时间，则进行学习
        loss = None
        if (len(self.memory) > self.batch_size and 
            self.t_step % 4 == 0):  # 每4步学习一次
            loss = self.learn()
            
            # 定期更新目标网络
            if self.t_step % self.update_target_every == 0:
                self.target_network.set_weights(self.q_network.get_weights())
                
        return loss
    
    def act(self, state, eval_mode=False):
        """选择动作（使用噪声网络或epsilon-greedy策略）"""
        # 将状态转换为批次格式
        state = np.reshape(state, [1, self.state_size])
        
        # 如果使用噪声网络
        if self.use_noisy_nets:
            # 评估模式使用没有噪声的网络
            if eval_mode:
                self.q_network.trainable = False  # 禁用训练模式（噪声）
                action_values = self.q_network(state, training=False)
                self.q_network.trainable = True   # 重新启用训练模式
            else:
                # 训练模式使用带噪声的网络
                action_values = self.q_network(state, training=True)
                
            return np.argmax(action_values[0])
        else:
            # 使用epsilon-greedy策略
            if eval_mode or random.random() > self.epsilon:
                action_values = self.q_network.predict(state, verbose=0)
                return np.argmax(action_values[0])
            else:
                return random.choice(np.arange(self.action_size))
    
    def learn(self):
        """从经验中学习"""
        # 从缓冲区中采样经验
        if self.use_prioritized_replay:
            experiences = self.memory.sample(self.batch_size)
            states, actions, rewards, next_states, dones, indices, weights = experiences
        else:
            experiences = self.memory.sample(self.batch_size)
            states, actions, rewards, next_states, dones = experiences
            weights = np.ones_like(rewards)  # 无优先级时权重全为1
            
        # 计算目标Q值（Double DQN或标准DQN）
        if self.use_double_dqn:
            # Double DQN: 使用在线网络选择动作，目标网络评估动作
            next_actions = np.argmax(self.q_network.predict(next_states, verbose=0), axis=1)
            next_q_values = self.target_network.predict(next_states, verbose=0)
            max_next_q = np.array([next_q_values[i, action] for i, action in enumerate(next_actions)])
        else:
            # 标准DQN: 使用目标网络选择和评估动作
            next_q_values = self.target_network.predict(next_states, verbose=0)
            max_next_q = np.max(next_q_values, axis=1)
            
        # 计算目标（考虑n步回报的gamma值）
        if self.use_n_step:
            targets = rewards + (self.gamma ** self.n_steps) * max_next_q * (1 - dones)
        else:
            targets = rewards + self.gamma * max_next_q * (1 - dones)
            
        # 获取当前Q值估计
        current_q = self.q_network.predict(states, verbose=0)
        
        # 如果使用优先经验回放，计算TD误差用于更新优先级
        if self.use_prioritized_replay:
            td_errors = []
            
        # 更新目标Q值
        for i, action in enumerate(actions):
            if self.use_prioritized_replay:
                old_val = current_q[i, action]
                td_error = targets[i] - old_val
                td_errors.append(td_error)
            current_q[i, action] = targets[i]
            
        # 使用样本权重进行加权训练
        history = self.q_network.fit(
            states, current_q, 
            sample_weight=weights,
            epochs=1, 
            verbose=0
        )
        loss = history.history['loss'][0]
        self.loss_history.append(loss)
        
        # 更新优先级
        if self.use_prioritized_replay:
            self.memory.update_priorities(indices, td_errors)
            
        # 如果不使用噪声网络，则更新epsilon
        if not self.use_noisy_nets:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
        return loss
    
    def save(self, filepath):
        """保存模型"""
        self.q_network.save_weights(filepath)
        
    def load(self, filepath):
        """加载模型"""
        self.q_network.load_weights(filepath)
        self.target_network.set_weights(self.q_network.get_weights())
        
    def get_epsilon(self):
        """获取当前epsilon值（如果使用）"""
        if not self.use_noisy_nets:
            return self.epsilon
        else:
            return 0  # 使用噪声网络时不使用epsilon


###################
# 训练和评估函数   #
###################

def set_chinese_font():
    """根据操作系统设置中文字体"""
    system = platform.system()
    try:
        if system == 'Windows':
            plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 黑体
            plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
            print("Font set to SimHei for Windows.")
        elif system == 'Darwin': # macOS
            plt.rcParams['font.sans-serif'] = ['PingFang SC'] # Mac 苹方
            plt.rcParams['axes.unicode_minus'] = False
            print("Font set to PingFang SC for macOS.")
        else: # Linux 或其他
            # 尝试查找常见 Linux 中文字体
            plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'Arial Unicode MS']
            plt.rcParams['axes.unicode_minus'] = False
            print("Attempting to set font to WenQuanYi Zen Hei/SimHei/Arial Unicode MS for Linux/Other.")
    except Exception as e:
        print(f"Error setting Chinese font: {e}. Matplotlib might fallback to default.")


def plot_scores(scores, avg_scores, title="学习曲线"):
    """绘制学习曲线"""
    set_chinese_font() # 在绘图前设置字体
    clear_output(True)
    plt.figure(figsize=(10, 5))
    plt.plot(scores)
    plt.plot(avg_scores)
    plt.title(title)
    plt.xlabel('回合数')
    plt.ylabel('分数')
    plt.legend(['回合分数', '平均分数 (最近100回合)'])
    plt.show()

def train_agent(agent, env_name='CartPole-v1', n_episodes=500, max_t=1000, 
                target_score=195.0, print_every=50, render=False):
    """训练智能体"""
    env = gym.make(env_name)
    scores = []
    avg_scores = []
    
    for i_episode in range(1, n_episodes+1):
        state, _ = env.reset(seed=SEED+i_episode)  # 使用不同的种子
        score = 0
        
        for t in range(max_t):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            
            if done:
                break
        
        scores.append(score)
        avg_score = np.mean(scores[-100:])  # 最近100回合的平均分数
        avg_scores.append(avg_score)
        
        # 输出进度
        message = f'回合 {i_episode}/{n_episodes}, 分数: {score:.2f}, 平均分数: {avg_score:.2f}'
        if not agent.use_noisy_nets:
            message += f', epsilon: {agent.get_epsilon():.4f}'
            
        print(f'\r{message}', end='')
        
        if i_episode % print_every == 0:
            print(f'\n回合 {i_episode}/{n_episodes}, 分数: {score:.2f}, 平均分数: {avg_score:.2f}', end='')
            if not agent.use_noisy_nets:
                print(f', epsilon: {agent.get_epsilon():.4f}')
            else:
                print()
            
            # 确保绘图时字体已设置
            plot_scores(scores, avg_scores, title=f"{env_name} - Rainbow DQN学习曲线")
            
        # 检查是否达到目标
        if avg_score >= target_score and i_episode >= 100:
            print(f'\n环境在{i_episode}回合后解决!')
            save_dir = './models'
            os.makedirs(save_dir, exist_ok=True)
            agent.save(f'{save_dir}/rainbow_dqn_{env_name}_{i_episode}.h5')
            break
    
    return agent, scores, avg_scores

def evaluate_agent(agent, env_name='CartPole-v1', n_episodes=10, render=False):
    """评估训练好的智能体"""
    env = gym.make(env_name, render_mode="human" if render else None)
    scores = []
    
    for i_episode in range(1, n_episodes+1):
        state, _ = env.reset(seed=SEED+i_episode)
        score = 0
        
        while True:
            action = agent.act(state, eval_mode=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = next_state
            score += reward
            
            if done:
                break
        
        scores.append(score)
        print(f'测试回合 {i_episode}/{n_episodes}, 分数: {score:.2f}')
    
    print(f'平均测试分数: {np.mean(scores):.2f}')
    return scores


if __name__ == "__main__":
    # 实验设置
    ENV_NAME = 'CartPole-v1'
    
    # 创建环境以获取状态和动作空间大小
    temp_env = gym.make(ENV_NAME)
    state_size = temp_env.observation_space.shape[0]
    action_size = temp_env.action_space.n
    
    print(f"环境: {ENV_NAME}")
    print(f"状态空间: {state_size}")
    print(f"动作空间: {action_size}")
    
    # 创建Rainbow DQN智能体
    agent = RainbowDQNAgent(
        state_size=state_size,
        action_size=action_size,
        # 算法配置
        use_double_dqn=True,
        use_dueling_dqn=True,
        use_prioritized_replay=True,
        use_n_step=True,
        use_noisy_nets=True,
        # 超参数
        learning_rate=0.001,
        gamma=0.99,
        buffer_size=50000,
        batch_size=64,
        update_target_every=1000,
        hidden_size=128,
        # 如果不使用Noisy Nets，则以下参数会被使用
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        # 优先经验回放参数
        alpha=0.6,
        beta_start=0.4,
        beta_frames=100000,
        # 多步学习参数
        n_steps=3
    )
    
    # 训练智能体
    agent, scores, avg_scores = train_agent(
        agent=agent,
        env_name=ENV_NAME,
        n_episodes=500,
        max_t=1000,
        target_score=195.0
    )
    
    # 绘制最终学习曲线
    set_chinese_font() # 绘图前设置字体
    plt.figure(figsize=(10, 5))
    plt.plot(scores)
    plt.plot(avg_scores)
    plt.title(f'{ENV_NAME} - Rainbow DQN学习曲线')
    plt.xlabel('回合数')
    plt.ylabel('分数')
    plt.legend(['回合分数', '平均分数 (最近100回合)'])
    plt.savefig(f'./rainbow_dqn_{ENV_NAME}_learning_curve.png')
    plt.show()
    
    # 评估训练好的智能体
    eval_scores = evaluate_agent(agent, ENV_NAME, render=True, n_episodes=5) 