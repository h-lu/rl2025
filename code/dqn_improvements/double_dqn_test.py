"""
双重DQN与标准DQN比较

该脚本比较双重DQN和标准DQN在CartPole环境中的性能差异。
双重DQN通过分离动作选择和价值评估减少Q值过高估计问题。
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
    """DQN智能体"""
    def __init__(self, state_size, action_size, use_double=False):
        self.state_size = state_size
        self.action_size = action_size
        self.use_double = use_double  # 是否使用双重DQN
        self.memory = ReplayBuffer()
        
        # 创建Q网络和目标网络
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
        
        # 记录每回合奖励和Q值
        self.episode_rewards = []
        self.target_q_values = []
        self.predicted_q_values = []
        self.td_errors = []  # 用于记录TD误差大小
    
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
        states, actions, rewards, next_states, dones = experiences
        
        if self.use_double:
            # 双重DQN：使用当前网络选择动作，使用目标网络评估动作
            next_actions = np.argmax(self.q_network.predict(next_states, verbose=0), axis=1)
            next_q_values = self.target_network.predict(next_states, verbose=0)
            # 对每个样本使用其最大动作的Q值
            max_next_q = np.array([next_q_values[i, next_actions[i]] for i in range(len(next_actions))])
        else:
            # 标准DQN：直接使用目标网络的最大Q值
            next_q_values = self.target_network.predict(next_states, verbose=0)
            max_next_q = np.max(next_q_values, axis=1)
        
        # 计算目标Q值
        targets = rewards + (1 - dones) * self.gamma * max_next_q
        
        # 记录平均目标Q值
        self.target_q_values.append(np.mean(targets))
        
        # 获取当前预测
        q_values = self.q_network.predict(states, verbose=0)
        
        # 计算并记录TD误差
        current_q = np.array([q_values[i, actions[i]] for i in range(len(actions))])
        td_errors = targets - current_q
        self.td_errors.append(np.mean(np.abs(td_errors)))
        
        # 记录平均预测Q值
        self.predicted_q_values.append(np.mean(current_q))
        
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
    
    def get_avg_predicted_q(self):
        """获取平均预测Q值"""
        if len(self.predicted_q_values) > 0:
            return np.mean(self.predicted_q_values[-100:])
        return 0
    
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
    target_q_values = []
    predicted_q_values = []
    
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
        td_errors.append(agent.get_avg_td_error())
        target_q_values.append(agent.get_avg_target_q())
        predicted_q_values.append(agent.get_avg_predicted_q())
        
        if episode % print_every == 0:
            print(f"回合 {episode}/{n_episodes}, 分数: {score:.2f}, 平均分数: {avg_score:.2f}, ε: {agent.epsilon:.4f}, TD误差: {agent.get_avg_td_error():.4f}")
    
    print()  # 新行
    return scores, avg_scores, td_errors, target_q_values, predicted_q_values

def plot_comparison(std_scores, double_scores, std_td, double_td, std_target_q, double_target_q, std_predicted_q, double_predicted_q):
    """绘制比较图"""
    # 使用英文输入表示算法名称
    std_label = 'Standard DQN'
    double_label = 'Double DQN'
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), dpi=100)
    
    # 绘制分数
    axes[0].plot(std_scores, label=std_label, color='blue', linewidth=2)
    axes[0].plot(double_scores, label=double_label, color='red', linewidth=2)
    axes[0].set_title('标准DQN与双重DQN的性能比较')
    axes[0].set_xlabel('回合')
    axes[0].set_ylabel('得分')
    axes[0].legend()
    axes[0].grid(True)
    
    # 绘制TD误差
    axes[1].plot(std_td, label=f'{std_label}的TD误差', color='blue', linewidth=2)
    axes[1].plot(double_td, label=f'{double_label}的TD误差', color='red', linewidth=2)
    axes[1].set_title('TD误差比较 (双重DQN通常有更小的TD误差)')
    axes[1].set_xlabel('回合')
    axes[1].set_ylabel('平均绝对TD误差')
    axes[1].legend()
    axes[1].grid(True)
    
    # 绘制Q值比较
    axes[2].plot(std_target_q, label=f'{std_label}目标Q值', color='blue', linewidth=2)
    axes[2].plot(std_predicted_q, label=f'{std_label}预测Q值', color='blue', linestyle='--', linewidth=2)
    axes[2].plot(double_target_q, label=f'{double_label}目标Q值', color='red', linewidth=2)
    axes[2].plot(double_predicted_q, label=f'{double_label}预测Q值', color='red', linestyle='--', linewidth=2)
    axes[2].set_title('Q值比较 (标准DQN通常有更高的预测Q值)')
    axes[2].set_xlabel('回合')
    axes[2].set_ylabel('平均Q值')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('double_dqn_comparison.png')
    print("比较图已保存为 'double_dqn_comparison.png'")
    
    # 额外生成一个使用英文的图（以防中文显示仍有问题）
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), dpi=100)
    
    # 绘制分数
    axes[0].plot(std_scores, label=std_label, color='blue', linewidth=2)
    axes[0].plot(double_scores, label=double_label, color='red', linewidth=2)
    axes[0].set_title('Performance Comparison: Standard DQN vs Double DQN')
    axes[0].set_xlabel('Episodes')
    axes[0].set_ylabel('Score')
    axes[0].legend()
    axes[0].grid(True)
    
    # 绘制TD误差
    axes[1].plot(std_td, label=f'{std_label} TD Error', color='blue', linewidth=2)
    axes[1].plot(double_td, label=f'{double_label} TD Error', color='red', linewidth=2)
    axes[1].set_title('TD Error Comparison (Double DQN typically has lower TD errors)')
    axes[1].set_xlabel('Episodes')
    axes[1].set_ylabel('Mean Absolute TD Error')
    axes[1].legend()
    axes[1].grid(True)
    
    # 绘制Q值比较
    axes[2].plot(std_target_q, label=f'{std_label} Target Q', color='blue', linewidth=2)
    axes[2].plot(std_predicted_q, label=f'{std_label} Predicted Q', color='blue', linestyle='--', linewidth=2)
    axes[2].plot(double_target_q, label=f'{double_label} Target Q', color='red', linewidth=2)
    axes[2].plot(double_predicted_q, label=f'{double_label} Predicted Q', color='red', linestyle='--', linewidth=2)
    axes[2].set_title('Q-Value Comparison (Standard DQN typically has higher predicted Q-values)')
    axes[2].set_xlabel('Episodes')
    axes[2].set_ylabel('Mean Q-Value')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('double_dqn_comparison_english.png')
    print("英文版比较图已保存为 'double_dqn_comparison_english.png'")

def main():
    # 创建环境
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    print(f"环境: CartPole-v1, 状态空间: {state_size}, 动作空间: {action_size}")
    print(f"开始训练标准DQN和双重DQN进行比较，每个算法训练{NUM_EPISODES}回合...")
    
    # 训练标准DQN
    print("\n训练标准DQN...")
    standard_agent = DQNAgent(state_size, action_size, use_double=False)
    std_scores, std_avg_scores, std_td, std_target_q, std_predicted_q = train_agent(standard_agent, env)
    
    # 训练双重DQN
    print("\n训练双重DQN...")
    double_agent = DQNAgent(state_size, action_size, use_double=True)
    double_scores, double_avg_scores, double_td, double_target_q, double_predicted_q = train_agent(double_agent, env)
    
    # 比较结果
    print("\n===== 结果比较 =====")
    print(f"标准DQN最终平均分数: {std_avg_scores[-1]:.2f}")
    print(f"双重DQN最终平均分数: {double_avg_scores[-1]:.2f}")
    print(f"标准DQN平均TD误差: {np.mean(std_td):.4f}")
    print(f"双重DQN平均TD误差: {np.mean(double_td):.4f}")
    print(f"标准DQN预测Q值与目标Q值差: {np.mean(std_predicted_q) - np.mean(std_target_q):.4f}")
    print(f"双重DQN预测Q值与目标Q值差: {np.mean(double_predicted_q) - np.mean(double_target_q):.4f}")
    
    # 绘制比较图
    plot_comparison(std_scores, double_scores, std_td, double_td, std_target_q, double_target_q, std_predicted_q, double_predicted_q)
    
    print("\n完成比较！")
    print("双重DQN通过分离动作选择和评估，减少了Q值的过高估计问题。")
    print("这可以从更小的TD误差以及预测Q值和目标Q值的更小差距观察到。")

if __name__ == "__main__":
    main() 