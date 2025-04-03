"""
双重DQN与标准DQN比较

该脚本比较双重DQN和标准DQN在CartPole环境中的性能差异。
双重DQN通过将Q值分解为状态价值函数V(s)和优势函数A(s,a)，提高学习效率。
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
import platform # 导入 platform 模块用于检测操作系统

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
    except Exception as e:
        print(f"Error setting Chinese font: {e}. Matplotlib might fallback to default.")

# 设置matplotlib中文字体支持
set_chinese_font()

# 设置随机种子以便结果可复现
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# 删除或注释掉禁用GPU的代码
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示 warning 和 Error

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

def create_dueling_q_network(state_size, action_size):
    """创建双重Q网络"""
    # 输入层
    inputs = layers.Input(shape=(state_size,))
    
    # 共享特征提取层
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.Dense(64, activation='relu')(x)
    
    # 状态价值流
    value_stream = layers.Dense(32, activation='relu')(x)
    value = layers.Dense(1)(value_stream)
    
    # 优势流
    advantage_stream = layers.Dense(32, activation='relu')(x)
    advantage = layers.Dense(action_size)(advantage_stream)
    
    # 组合状态价值和优势以得到Q值
    # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
    
    # 添加 output_shape 函数以解决 Lambda 层的问题
    def get_advantage_mean_shape(tensor_shape):
        return (tensor_shape[0], 1)
    
    advantage_mean = layers.Lambda(
        lambda x: tf.reduce_mean(x, axis=1, keepdims=True),
        output_shape=get_advantage_mean_shape
    )(advantage)
    
    q_values = layers.Add()([value, layers.Subtract()([advantage, advantage_mean])])
    
    model = keras.Model(inputs=inputs, outputs=q_values)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='mse')
    return model

class DQNAgent:
    """DQN智能体基类"""
    def __init__(self, state_size, action_size, use_dueling=False):
        self.state_size = state_size
        self.action_size = action_size
        self.use_dueling = use_dueling
        
        # 创建网络
        if use_dueling:
            self.q_network = create_dueling_q_network(state_size, action_size)
            self.target_network = create_dueling_q_network(state_size, action_size)
        else:
            self.q_network = create_standard_q_network(state_size, action_size)
            self.target_network = create_standard_q_network(state_size, action_size)
        
        # 更新目标网络权重
        self.update_target_network()
        
        # 经验回放缓冲区
        self.memory = ReplayBuffer()
        
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
        
        # 记录每回合奖励
        self.episode_rewards = []
        
        # 记录目标Q值
        self.target_q_values = []
        
        # 记录状态价值和优势值（仅用于双重DQN）
        if use_dueling:
            self.value_estimates = []
            self.advantage_estimates = []
    
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
        
        # 计算目标Q值
        next_q_values = self.target_network.predict(next_states, verbose=0)
        max_next_q = np.max(next_q_values, axis=1)
        targets = rewards + (1 - dones) * self.gamma * max_next_q
        
        # 记录平均目标Q值
        self.target_q_values.append(np.mean(targets))
        
        # 获取当前预测
        q_values = self.q_network.predict(states, verbose=0)
        
        # 记录状态价值和优势值（仅用于双重DQN）
        if self.use_dueling:
            # 为了显示效果，我们创建一个临时模型来提取状态价值
            if hasattr(self, 'value_model') and self.value_model is not None:
                value_estimates = self.value_model.predict(states, verbose=0)
                self.value_estimates.append(np.mean(value_estimates))
                
                # 使用Q值和状态价值计算优势
                advantages = q_values - np.repeat(value_estimates, self.action_size, axis=1)
                self.advantage_estimates.append(np.mean(advantages))
        
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
    
    def get_avg_value_estimate(self):
        """获取平均状态价值估计（仅用于双重DQN）"""
        if self.use_dueling and len(self.value_estimates) > 0:
            return np.mean(self.value_estimates[-100:])
        return 0
    
    def get_avg_advantage_estimate(self):
        """获取平均优势估计（仅用于双重DQN）"""
        if self.use_dueling and len(self.advantage_estimates) > 0:
            return np.mean(self.advantage_estimates[-100:])
        return 0

def create_value_extraction_model(dueling_model):
    """创建一个提取状态价值的模型，用于可视化"""
    # 这里我们简化处理，直接返回一个近似的状态价值
    inputs = layers.Input(shape=(dueling_model.input_shape[1],))
    x = dueling_model.layers[1](inputs)  # 第一个Dense层
    x = dueling_model.layers[2](x)       # 第二个Dense层
    x = layers.Dense(32, activation='relu')(x)
    value = layers.Dense(1)(x)
    
    model = keras.Model(inputs=inputs, outputs=value)
    return model

def train_agent(agent, env, n_episodes=NUM_EPISODES, max_steps=MAX_STEPS_PER_EPISODE, print_every=PRINT_EVERY):
    """训练智能体并返回性能数据"""
    scores = []
    avg_scores = []
    target_q_values = []
    value_estimates = []
    advantage_estimates = []
    
    # 如果是双重DQN，创建一个提取状态价值的模型
    if agent.use_dueling:
        agent.value_model = create_value_extraction_model(agent.q_network)
    
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
        
        if agent.use_dueling:
            value_estimates.append(agent.get_avg_value_estimate())
            advantage_estimates.append(agent.get_avg_advantage_estimate())
        
        if episode % print_every == 0:
            print(f"回合 {episode}/{n_episodes}, 分数: {score:.2f}, 平均分数: {avg_score:.2f}, ε: {agent.epsilon:.4f}, 目标Q值: {agent.get_avg_target_q():.2f}")
    
    print()  # 新行
    
    if agent.use_dueling:
        return scores, avg_scores, target_q_values, value_estimates, advantage_estimates
    else:
        return scores, avg_scores, target_q_values, None, None

def plot_comparison(std_scores, dueling_scores, std_q, dueling_q, value_estimates=None, advantage_estimates=None):
    """绘制比较图"""
    # 设置字体
    set_chinese_font()
    
    # 使用英文输入表示算法名称
    std_label = 'Standard DQN'
    dueling_label = 'Dueling DQN'
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 12), dpi=100)
    
    # 绘制分数
    axes[0].plot(std_scores, label=std_label, color='blue', linewidth=2)
    axes[0].plot(dueling_scores, label=dueling_label, color='red', linewidth=2)
    axes[0].set_title('标准DQN与双重DQN的性能比较')
    axes[0].set_xlabel('回合')
    axes[0].set_ylabel('得分')
    axes[0].legend()
    axes[0].grid(True)
    
    # 绘制目标Q值
    axes[1].plot(std_q, label=f'{std_label}的目标Q值', color='blue', linewidth=2)
    axes[1].plot(dueling_q, label=f'{dueling_label}的目标Q值', color='red', linewidth=2)
    axes[1].set_title('目标Q值比较')
    axes[1].set_xlabel('回合')
    axes[1].set_ylabel('平均目标Q值')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('dueling_dqn_comparison.png')
    print("比较图已保存为 'dueling_dqn_comparison.png'")
    
    # 绘制双重DQN的状态价值和优势图
    if value_estimates is not None and advantage_estimates is not None and len(value_estimates) > 0:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
        ax.plot(value_estimates, label='状态价值 V(s)', color='green', linewidth=2)
        ax.plot(advantage_estimates, label='平均优势 A(s,a)', color='purple', linewidth=2)
        ax.set_title('双重DQN的状态价值和优势分解')
        ax.set_xlabel('回合')
        ax.set_ylabel('值')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig('dueling_dqn_decomposition.png')
        print("双重DQN分解图已保存为 'dueling_dqn_decomposition.png'")
    
    # 额外生成一个使用英文的图（以防中文显示仍有问题）
    fig, axes = plt.subplots(2, 1, figsize=(10, 12), dpi=100)
    
    # 绘制分数
    axes[0].plot(std_scores, label=std_label, color='blue', linewidth=2)
    axes[0].plot(dueling_scores, label=dueling_label, color='red', linewidth=2)
    axes[0].set_title('Performance Comparison: Standard DQN vs Dueling DQN')
    axes[0].set_xlabel('Episodes')
    axes[0].set_ylabel('Score')
    axes[0].legend()
    axes[0].grid(True)
    
    # 绘制目标Q值
    axes[1].plot(std_q, label=f'{std_label} Target Q-Values', color='blue', linewidth=2)
    axes[1].plot(dueling_q, label=f'{dueling_label} Target Q-Values', color='red', linewidth=2)
    axes[1].set_title('Target Q-Value Comparison')
    axes[1].set_xlabel('Episodes')
    axes[1].set_ylabel('Average Target Q-Value')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('dueling_dqn_comparison_english.png')
    print("英文版比较图已保存为 'dueling_dqn_comparison_english.png'")
    
    # 英文版双重DQN的状态价值和优势图
    if value_estimates is not None and advantage_estimates is not None and len(value_estimates) > 0:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
        ax.plot(value_estimates, label='State Value V(s)', color='green', linewidth=2)
        ax.plot(advantage_estimates, label='Mean Advantage A(s,a)', color='purple', linewidth=2)
        ax.set_title('Dueling DQN Value and Advantage Decomposition')
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Values')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig('dueling_dqn_decomposition_english.png')
        print("英文版双重DQN分解图已保存为 'dueling_dqn_decomposition_english.png'")

def main():
    # 创建环境
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    print(f"环境: CartPole-v1, 状态空间: {state_size}, 动作空间: {action_size}")
    print(f"开始训练标准DQN和双重DQN进行比较，每个算法训练{NUM_EPISODES}回合...")
    
    # 训练标准DQN
    print("\n训练标准DQN...")
    standard_agent = DQNAgent(state_size, action_size, use_dueling=False)
    std_scores, std_avg_scores, std_q, _, _ = train_agent(standard_agent, env)
    
    # 训练双重DQN
    print("\n训练双重DQN...")
    dueling_agent = DQNAgent(state_size, action_size, use_dueling=True)
    dueling_scores, dueling_avg_scores, dueling_q, value_estimates, advantage_estimates = train_agent(dueling_agent, env)
    
    # 比较结果
    print("\n===== 结果比较 =====")
    print(f"标准DQN最终平均分数: {std_avg_scores[-1]:.2f}")
    print(f"双重DQN最终平均分数: {dueling_avg_scores[-1]:.2f}")
    print(f"标准DQN平均目标Q值: {np.mean(std_q):.2f}")
    print(f"双重DQN平均目标Q值: {np.mean(dueling_q):.2f}")
    
    # 绘制比较图
    plot_comparison(std_scores, dueling_scores, std_q, dueling_q, value_estimates, advantage_estimates)
    
    print("\n完成比较！")
    print("双重DQN的特点是将Q值分解为状态价值V(s)和优势函数A(s,a)，对于价值接近的动作有更好的泛化能力。")

if __name__ == "__main__":
    main() 