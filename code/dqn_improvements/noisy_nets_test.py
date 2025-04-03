"""
噪声网络DQN与标准DQN比较

该脚本比较噪声网络DQN和标准DQN在CartPole环境中的性能差异。
噪声网络通过在网络权重中加入参数化噪声，提供更加自适应的探索机制。
"""

import os
import numpy as np
import gymnasium as gym
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import random
from collections import deque
import matplotlib as mpl
import platform # 导入platform模块用于检测操作系统

#####################################
# 关键参数设置（易于修改）
#####################################
# 训练参数
NUM_EPISODES = 10  # 训练回合数（10轮快速测试）
MAX_STEPS_PER_EPISODE = 200  # 每个回合的最大步数
PRINT_EVERY = 2  # 每隔多少回合打印一次

# DQN参数
BUFFER_SIZE = 10000  # 经验回放缓冲区大小
BATCH_SIZE = 64  # 批量大小
LEARNING_RATE = 0.001  # 学习率
GAMMA = 0.99  # 折扣因子
EPSILON_START = 1.0  # 初始探索率（仅用于标准DQN）
EPSILON_MIN = 0.01  # 最小探索率（仅用于标准DQN）
EPSILON_DECAY = 0.995  # 探索率衰减（仅用于标准DQN）
UPDATE_TARGET_EVERY = 10  # 目标网络更新频率

# 噪声网络参数
SIGMA_INIT = 0.5  # 初始噪声标准差
#####################################

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
        mpl.rcParams['font.family'] = ['sans-serif']  # 使用无衬线字体
    except Exception as e:
        print(f"Error setting Chinese font: {e}. Matplotlib might fallback to default.")

# 设置matplotlib中文字体支持
set_chinese_font()

# 设置随机种子以便结果可复现
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# 设置TensorFlow日志级别（只显示警告和错误）
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class NoisyDense(layers.Layer):
    """噪声网络层，用于替代标准Dense层"""
    def __init__(self, units, activation=None, sigma_init=SIGMA_INIT):
        super(NoisyDense, self).__init__()
        self.units = units
        self.activation = keras.activations.get(activation)
        self.sigma_init = sigma_init
        
    def build(self, input_shape):
        self.p = int(input_shape[-1])  # 输入维度
        
        # 初始化均值权重和偏置
        mu_range = 1.0 / np.sqrt(self.p)
        self.mu_w = self.add_weight(
            name="mu_w", 
            shape=[self.p, self.units],
            initializer=keras.initializers.RandomUniform(-mu_range, mu_range),
            trainable=True
        )
        self.mu_b = self.add_weight(
            name="mu_b", 
            shape=[self.units],
            initializer=keras.initializers.RandomUniform(-mu_range, mu_range),
            trainable=True
        )
        
        # 初始化噪声权重和偏置的标准差
        self.sigma_w = self.add_weight(
            name="sigma_w", 
            shape=[self.p, self.units],
            initializer=keras.initializers.Constant(self.sigma_init / np.sqrt(self.p)),
            trainable=True
        )
        self.sigma_b = self.add_weight(
            name="sigma_b", 
            shape=[self.units],
            initializer=keras.initializers.Constant(self.sigma_init / np.sqrt(self.units)),
            trainable=True
        )
        
        super(NoisyDense, self).build(input_shape)
    
    def call(self, inputs, training=None):
        # 训练时添加噪声，预测时不添加
        if training is None:
            training = tf.keras.backend.learning_phase()
        
        def noisy_path():
            # 生成噪声
            eps_w = self._get_factorized_noise(self.p, self.units)
            eps_b = self._get_factorized_noise(1, self.units)
            
            # 应用噪声
            w = self.mu_w + self.sigma_w * eps_w
            b = self.mu_b + self.sigma_b * eps_b
            return tf.matmul(inputs, w) + b
        
        def deterministic_path():
            # 不添加噪声，仅使用均值参数
            return tf.matmul(inputs, self.mu_w) + self.mu_b
        
        # 根据训练状态选择路径
        output = tf.cond(tf.cast(training, tf.bool), noisy_path, deterministic_path)
        
        if self.activation is not None:
            output = self.activation(output)
        
        return output
    
    def _get_factorized_noise(self, batch_size, units):
        """生成因子化噪声，减少参数数量"""
        f1 = self._f(tf.random.normal([batch_size, 1]))
        f2 = self._f(tf.random.normal([1, units]))
        return tf.matmul(f1, f2)
    
    def _f(self, x):
        """噪声转换函数"""
        return tf.math.sign(x) * tf.sqrt(tf.abs(x))
        
    def compute_output_shape(self, input_shape):
        """计算输出形状"""
        return (input_shape[0], self.units)

class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity=BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def __len__(self):
        return len(self.buffer)

def create_standard_q_network(state_size, action_size):
    """创建标准Q网络"""
    model = keras.Sequential([
        layers.Input(shape=(state_size,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(action_size, activation='linear')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='mse')
    return model

def create_noisy_q_network(state_size, action_size):
    """创建噪声Q网络"""
    inputs = layers.Input(shape=(state_size,))
    x = NoisyDense(64, activation='relu')(inputs)
    x = NoisyDense(64, activation='relu')(x)
    outputs = NoisyDense(action_size, activation='linear')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='mse')
    return model

class DQNAgent:
    """DQN智能体基类"""
    def __init__(self, state_size, action_size, use_noisy_nets=False):
        self.state_size = state_size
        self.action_size = action_size
        self.use_noisy_nets = use_noisy_nets
        
        # 回放缓冲区
        self.memory = ReplayBuffer()
        
        # 创建网络
        if use_noisy_nets:
            self.q_network = create_noisy_q_network(state_size, action_size)
            self.target_network = create_noisy_q_network(state_size, action_size)
            # 噪声网络不需要epsilon-greedy探索
            self.epsilon = 0
            self.epsilon_min = 0
            self.epsilon_decay = 0
        else:
            self.q_network = create_standard_q_network(state_size, action_size)
            self.target_network = create_standard_q_network(state_size, action_size)
            # 标准DQN需要epsilon-greedy探索
            self.epsilon = EPSILON_START
            self.epsilon_min = EPSILON_MIN
            self.epsilon_decay = EPSILON_DECAY
        
        # 复制网络权重
        self.update_target_network()
        
        # 超参数
        self.gamma = GAMMA
        self.batch_size = BATCH_SIZE
        self.train_start = 500  # 开始训练的最小经验数量
        self.update_target_every = UPDATE_TARGET_EVERY
        
        # 训练步数计数
        self.train_step = 0
        
        # 记录每回合奖励
        self.episode_rewards = []
        
        # 记录目标Q值
        self.target_q_values = []
    
    def update_target_network(self):
        """更新目标网络权重"""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def act(self, state):
        """选择动作"""
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        
        state = np.reshape(state, [1, self.state_size])
        # 噪声网络使用训练模式来生成噪声，进行探索
        if self.use_noisy_nets:
            q_values = self.q_network(state, training=True).numpy()[0]
        else:
            q_values = self.q_network.predict(state, verbose=0)[0]
        return np.argmax(q_values)
    
    def step(self, state, action, reward, next_state, done):
        """执行一步并学习"""
        self.memory.add(state, action, reward, next_state, done)
        
        if len(self.memory) > self.train_start:
            self.train_step += 1
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)
            
            # 标准DQN衰减epsilon
            if not self.use_noisy_nets:
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            if self.train_step % self.update_target_every == 0:
                self.update_target_network()
    
    def learn(self, experiences):
        """从经验中学习"""
        states, actions, rewards, next_states, dones = experiences
        
        # 计算目标Q值
        if self.use_noisy_nets:
            # 噪声网络在预测目标Q值时不使用噪声
            next_q_values = self.target_network(next_states, training=False).numpy()
        else:
            next_q_values = self.target_network.predict(next_states, verbose=0)
        
        max_next_q = np.max(next_q_values, axis=1)
        targets = rewards + (1 - dones) * self.gamma * max_next_q
        
        # 记录平均目标Q值
        self.target_q_values.append(np.mean(targets))
        
        # 获取当前预测
        if self.use_noisy_nets:
            q_values = self.q_network(states, training=False).numpy()
        else:
            q_values = self.q_network.predict(states, verbose=0)
        
        # 更新所选动作的Q值
        for i, action in enumerate(actions):
            q_values[i][action] = targets[i]
        
        # 训练网络
        self.q_network.fit(states, q_values, epochs=1, verbose=0)
    
    def record_episode_reward(self, episode_reward):
        """记录回合总奖励"""
        self.episode_rewards.append(episode_reward)
    
    def get_avg_target_q(self):
        """获取平均目标Q值"""
        if len(self.target_q_values) > 0:
            return np.mean(self.target_q_values[-100:])
        return 0

def train_agent(agent, env, n_episodes=NUM_EPISODES, max_steps=MAX_STEPS_PER_EPISODE, print_every=PRINT_EVERY):
    """训练智能体并返回性能数据"""
    scores = []
    avg_scores = []
    target_q_values = []
    
    for episode in range(1, n_episodes+1):
        state, _ = env.reset(seed=SEED+episode)
        score = 0
        
        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            
            if done:
                break
        
        # 记录回合奖励
        agent.record_episode_reward(score)
        
        scores.append(score)
        avg_score = np.mean(scores)  # 计算所有回合的平均分数
        avg_scores.append(avg_score)
        target_q_values.append(agent.get_avg_target_q())
        
        if episode % print_every == 0:
            epsilon_info = f", ε: {agent.epsilon:.4f}" if not agent.use_noisy_nets else ""
            print(f"回合 {episode}/{n_episodes}, 分数: {score:.2f}, 平均分数: {avg_score:.2f}{epsilon_info}, 目标Q值: {agent.get_avg_target_q():.2f}")
    
    print()  # 新行
    return scores, avg_scores, target_q_values

def plot_comparison(std_scores, noisy_scores, std_q, noisy_q):
    """绘制比较图"""
    # 使用英文输入表示算法名称
    std_label = 'Standard DQN (ε-greedy)'
    noisy_label = 'Noisy DQN'
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), dpi=100)
    
    # 绘制分数
    axes[0].plot(std_scores, label=std_label, color='blue', linewidth=2)
    axes[0].plot(noisy_scores, label=noisy_label, color='red', linewidth=2)
    axes[0].set_title('标准DQN与噪声网络DQN的性能比较')
    axes[0].set_xlabel('回合')
    axes[0].set_ylabel('得分')
    axes[0].legend()
    axes[0].grid(True)
    
    # 绘制目标Q值
    axes[1].plot(std_q, label=f'{std_label}的目标Q值', color='blue', linewidth=2)
    axes[1].plot(noisy_q, label=f'{noisy_label}的目标Q值', color='red', linewidth=2)
    axes[1].set_title('目标Q值比较')
    axes[1].set_xlabel('回合')
    axes[1].set_ylabel('平均目标Q值')
    axes[1].legend()
    axes[1].grid(True)
    
    # 绘制行为噪声
    std_action_noise = np.ones(len(std_scores)) * EPSILON_START
    for i in range(1, len(std_scores)):
        std_action_noise[i] = max(EPSILON_MIN, std_action_noise[i-1] * EPSILON_DECAY)
        
    noisy_action_noise = np.ones(len(noisy_scores)) * SIGMA_INIT
    
    axes[2].plot(std_action_noise, label='ε-greedy探索率', color='blue', linewidth=2)
    axes[2].plot(noisy_action_noise, label='噪声网络初始σ', color='red', linewidth=2)
    axes[2].set_title('探索策略比较')
    axes[2].set_xlabel('回合')
    axes[2].set_ylabel('噪声水平')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('noisy_nets_comparison.png')
    print("比较图已保存为 'noisy_nets_comparison.png'")
    
    # 额外生成一个使用英文的图（以防中文显示仍有问题）
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), dpi=100)
    
    # 绘制分数
    axes[0].plot(std_scores, label=std_label, color='blue', linewidth=2)
    axes[0].plot(noisy_scores, label=noisy_label, color='red', linewidth=2)
    axes[0].set_title('Performance Comparison: Standard DQN vs Noisy DQN')
    axes[0].set_xlabel('Episodes')
    axes[0].set_ylabel('Score')
    axes[0].legend()
    axes[0].grid(True)
    
    # 绘制目标Q值
    axes[1].plot(std_q, label=f'{std_label} Target Q-Values', color='blue', linewidth=2)
    axes[1].plot(noisy_q, label=f'{noisy_label} Target Q-Values', color='red', linewidth=2)
    axes[1].set_title('Target Q-Value Comparison')
    axes[1].set_xlabel('Episodes')
    axes[1].set_ylabel('Average Target Q-Value')
    axes[1].legend()
    axes[1].grid(True)
    
    # 绘制行为噪声
    axes[2].plot(std_action_noise, label='ε-greedy exploration rate', color='blue', linewidth=2)
    axes[2].plot(noisy_action_noise, label='Noisy Network initial σ', color='red', linewidth=2)
    axes[2].set_title('Exploration Strategy Comparison')
    axes[2].set_xlabel('Episodes')
    axes[2].set_ylabel('Noise Level')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('noisy_nets_comparison_english.png')
    print("英文版比较图已保存为 'noisy_nets_comparison_english.png'")

def main():
    # 创建环境
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    print(f"环境: CartPole-v1, 状态空间: {state_size}, 动作空间: {action_size}")
    print(f"开始训练标准DQN和噪声网络DQN进行比较，每个算法训练{NUM_EPISODES}回合...")
    
    # 训练标准DQN
    print("\n训练标准DQN（使用ε-greedy探索）...")
    standard_agent = DQNAgent(state_size, action_size, use_noisy_nets=False)
    std_scores, std_avg_scores, std_q = train_agent(standard_agent, env)
    
    # 训练噪声网络DQN
    print("\n训练噪声网络DQN（使用参数化噪声探索）...")
    noisy_agent = DQNAgent(state_size, action_size, use_noisy_nets=True)
    noisy_scores, noisy_avg_scores, noisy_q = train_agent(noisy_agent, env)
    
    # 比较结果
    print("\n===== 结果比较 =====")
    print(f"标准DQN最终平均分数: {std_avg_scores[-1]:.2f}")
    print(f"噪声网络DQN最终平均分数: {noisy_avg_scores[-1]:.2f}")
    print(f"标准DQN平均目标Q值: {np.mean(std_q):.2f}")
    print(f"噪声网络DQN平均目标Q值: {np.mean(noisy_q):.2f}")
    
    # 绘制比较图
    plot_comparison(std_scores, noisy_scores, std_q, noisy_q)
    
    print("\n完成比较！噪声网络DQN的特点是提供自适应的参数化探索，无需手动设置epsilon衰减。")

if __name__ == "__main__":
    main() 