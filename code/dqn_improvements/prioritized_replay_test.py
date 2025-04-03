"""
优先经验回放DQN与标准DQN比较

该脚本比较带优先经验回放的DQN和标准DQN在CartPole环境中的性能差异。
优先经验回放根据TD误差为经验分配优先级，提高采样效率。
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
EPSILON_START = 1.0  # 初始探索率
EPSILON_MIN = 0.01  # 最小探索率
EPSILON_DECAY = 0.995  # 探索率衰减
UPDATE_TARGET_EVERY = 10  # 目标网络更新频率

# 优先经验回放参数
ALPHA = 0.6  # 优先级指数，控制使用TD误差的程度
BETA_START = 0.4  # 初始重要性采样指数
BETA_FRAMES = 10000  # beta退火的帧数
EPSILON_PER = 0.01  # 防止优先级为0
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

class StandardReplayBuffer:
    """标准经验回放缓冲区"""
    def __init__(self, capacity=BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones), None, None
    
    def update_priorities(self, indices, priorities):
        # 标准缓冲区不需要更新优先级
        pass
    
    def __len__(self):
        return len(self.buffer)

class PrioritizedReplayBuffer:
    """优先经验回放缓冲区"""
    def __init__(self, capacity=BUFFER_SIZE, alpha=ALPHA, beta_start=BETA_START, beta_frames=BETA_FRAMES):
        self.capacity = capacity
        self.alpha = alpha  # 确定优先级使用程度的参数
        self.beta = beta_start  # 重要性采样的参数
        self.beta_frames = beta_frames
        self.frame = 1  # 用于beta退火
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.buffer = np.zeros(capacity, dtype=object)
        self.pos = 0
        self.size = 0
    
    def add(self, state, action, reward, next_state, done):
        # 创建经验元组并存储
        experience = (state, action, reward, next_state, done)
        
        # 获取最大优先级（用于新经验）
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        
        # 存储经验和其优先级
        if self.size < self.capacity:
            self.buffer[self.size] = experience
            self.priorities[self.size] = max_priority
            self.size += 1
        else:
            self.buffer[self.pos] = experience
            self.priorities[self.pos] = max_priority
            
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size):
        # 如果缓冲区未满，只从有效部分采样
        if self.size < self.capacity:
            probs = self.priorities[:self.size]
        else:
            probs = self.priorities
        
        # 根据优先级计算概率
        probs = probs ** self.alpha
        probs = probs / probs.sum()
        
        # 根据概率采样索引
        indices = np.random.choice(len(probs), batch_size, p=probs)
        
        # 更新beta（用于重要性采样权重）
        self.beta = min(1.0, self.beta + self.frame * (1.0 - self.beta) / self.beta_frames)
        self.frame += 1
        
        # 计算重要性采样权重
        # 权重 = (N * P(i))^(-β)
        N = len(probs)
        weights = (N * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()  # 归一化
        
        # 获取样本
        samples = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*samples)
        
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones), indices, np.array(weights, dtype=np.float32))
    
    def update_priorities(self, indices, priorities):
        """更新经验的优先级"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + EPSILON_PER  # 加上小常数防止优先级为0
    
    def __len__(self):
        return self.size

def create_q_network(state_size, action_size):
    """创建Q网络"""
    model = keras.Sequential([
        layers.Input(shape=(state_size,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(action_size, activation='linear')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='mse')
    return model

class DQNAgent:
    """基础DQN智能体"""
    def __init__(self, state_size, action_size, use_prioritized=False):
        self.state_size = state_size
        self.action_size = action_size
        self.use_prioritized = use_prioritized
        
        # 选择回放缓冲区类型
        if use_prioritized:
            self.memory = PrioritizedReplayBuffer()
        else:
            self.memory = StandardReplayBuffer()
        
        # Q网络
        self.q_network = create_q_network(state_size, action_size)
        self.target_network = create_q_network(state_size, action_size)
        self.update_target_network()
        
        # 超参数
        self.gamma = GAMMA
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.batch_size = BATCH_SIZE
        self.train_start = 500  # 开始训练的最小经验数量
        self.update_target_every = UPDATE_TARGET_EVERY
        
        # 训练步数计数
        self.train_step = 0
        
        # 记录TD误差
        self.td_errors = []
    
    def update_target_network(self):
        """更新目标网络权重"""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def act(self, state):
        """选择动作"""
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        
        state = np.reshape(state, [1, self.state_size])
        q_values = self.q_network.predict(state, verbose=0)[0]
        return np.argmax(q_values)
    
    def step(self, state, action, reward, next_state, done):
        """执行一步并学习"""
        self.memory.add(state, action, reward, next_state, done)
        
        if len(self.memory) > self.train_start:
            self.train_step += 1
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)
            
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            if self.train_step % self.update_target_every == 0:
                self.update_target_network()
    
    def learn(self, experiences):
        """从经验中学习"""
        states, actions, rewards, next_states, dones, indices, weights = experiences
        
        # 计算目标Q值 - 使用目标网络
        next_q_values = self.target_network.predict(next_states, verbose=0)
        max_next_q = np.max(next_q_values, axis=1)
        targets = rewards + (1 - dones) * self.gamma * max_next_q
        
        # 获取当前预测
        q_values = self.q_network.predict(states, verbose=0)
        
        # 计算TD误差
        td_errors = []
        for i, action in enumerate(actions):
            old_val = q_values[i][action]
            q_values[i][action] = targets[i]
            # 记录TD误差绝对值
            td_error = abs(old_val - targets[i])
            td_errors.append(td_error)
        
        # 如果使用优先经验回放，更新优先级
        if self.use_prioritized and indices is not None:
            self.memory.update_priorities(indices, td_errors)
        
        # 记录平均TD误差
        avg_td_error = np.mean(td_errors)
        self.td_errors.append(avg_td_error)
        
        # 使用自定义权重进行拟合（对于标准DQN，权重全为1）
        if weights is None:
            weights = np.ones_like(rewards)
            
        # 训练网络
        self.q_network.fit(states, q_values, sample_weight=weights, epochs=1, verbose=0)
        
    def get_avg_td_error(self):
        """获取平均TD误差"""
        if len(self.td_errors) > 0:
            return np.mean(self.td_errors[-100:])
        return 0

def train_agent(agent, env, n_episodes=NUM_EPISODES, max_steps=MAX_STEPS_PER_EPISODE, print_every=PRINT_EVERY):
    """训练智能体并返回性能数据"""
    scores = []
    avg_scores = []
    td_errors = []
    
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
        
        scores.append(score)
        avg_score = np.mean(scores)  # 计算所有回合的平均分数
        avg_scores.append(avg_score)
        td_errors.append(agent.get_avg_td_error())
        
        if episode % print_every == 0:
            print(f"回合 {episode}/{n_episodes}, 分数: {score:.2f}, 平均分数: {avg_score:.2f}, ε: {agent.epsilon:.4f}, TD误差: {agent.get_avg_td_error():.4f}")
    
    print()  # 新行
    return scores, avg_scores, td_errors

def plot_comparison(std_scores, per_scores, std_td, per_td):
    """绘制比较图"""
    # 使用英文输入表示算法名称
    std_label = 'Standard DQN'
    per_label = 'Prioritized DQN'
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 12), dpi=100)
    
    # 绘制分数
    axes[0].plot(std_scores, label=std_label, color='blue', linewidth=2)
    axes[0].plot(per_scores, label=per_label, color='red', linewidth=2)
    axes[0].set_title('标准DQN与优先经验回放DQN的性能比较')
    axes[0].set_xlabel('回合')
    axes[0].set_ylabel('得分')
    axes[0].legend()
    axes[0].grid(True)
    
    # 绘制TD误差
    axes[1].plot(std_td, label=f'{std_label}的TD误差', color='blue', linewidth=2)
    axes[1].plot(per_td, label=f'{per_label}的TD误差', color='red', linewidth=2)
    axes[1].set_title('TD误差比较 (优先经验回放可能导致TD误差更快下降)')
    axes[1].set_xlabel('回合')
    axes[1].set_ylabel('平均TD误差')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('prioritized_replay_comparison.png')
    print("比较图已保存为 'prioritized_replay_comparison.png'")
    
    # 额外生成一个使用英文的图（以防中文显示仍有问题）
    fig, axes = plt.subplots(2, 1, figsize=(10, 12), dpi=100)
    
    # 绘制分数
    axes[0].plot(std_scores, label=std_label, color='blue', linewidth=2)
    axes[0].plot(per_scores, label=per_label, color='red', linewidth=2)
    axes[0].set_title('Performance Comparison: Standard DQN vs Prioritized DQN')
    axes[0].set_xlabel('Episodes')
    axes[0].set_ylabel('Score')
    axes[0].legend()
    axes[0].grid(True)
    
    # 绘制TD误差
    axes[1].plot(std_td, label=f'{std_label} TD Error', color='blue', linewidth=2)
    axes[1].plot(per_td, label=f'{per_label} TD Error', color='red', linewidth=2)
    axes[1].set_title('TD Error Comparison (Prioritized Replay may lead to faster TD error reduction)')
    axes[1].set_xlabel('Episodes')
    axes[1].set_ylabel('Average TD Error')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('prioritized_replay_comparison_english.png')
    print("英文版比较图已保存为 'prioritized_replay_comparison_english.png'")

def main():
    # 创建环境
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    print(f"环境: CartPole-v1, 状态空间: {state_size}, 动作空间: {action_size}")
    print(f"开始训练标准DQN和优先经验回放DQN进行比较，每个算法训练{NUM_EPISODES}回合...")
    
    # 训练标准DQN
    print("\n训练标准DQN...")
    standard_agent = DQNAgent(state_size, action_size, use_prioritized=False)
    std_scores, std_avg_scores, std_td = train_agent(standard_agent, env)
    
    # 训练优先经验回放DQN
    print("\n训练优先经验回放DQN...")
    prioritized_agent = DQNAgent(state_size, action_size, use_prioritized=True)
    per_scores, per_avg_scores, per_td = train_agent(prioritized_agent, env)
    
    # 比较结果
    print("\n===== 结果比较 =====")
    print(f"标准DQN最终平均分数: {std_avg_scores[-1]:.2f}")
    print(f"优先经验回放DQN最终平均分数: {per_avg_scores[-1]:.2f}")
    print(f"标准DQN平均TD误差: {np.mean(std_td):.4f}")
    print(f"优先经验回放DQN平均TD误差: {np.mean(per_td):.4f}")
    
    # 绘制比较图
    plot_comparison(std_scores, per_scores, std_td, per_td)
    
    print("\n完成比较！")

if __name__ == "__main__":
    main() 