"""
多步学习DQN与标准DQN比较

该脚本比较多步学习DQN和标准DQN在CartPole环境中的性能差异。
多步学习通过使用n步回报加速价值传播，提高学习效率。
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

# 多步学习参数
N_STEPS = 3  # 多步学习的步数
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

class StandardReplayBuffer:
    """标准经验回放缓冲区"""
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

class NStepReplayBuffer:
    """多步学习经验回放缓冲区"""
    def __init__(self, capacity=BUFFER_SIZE, n_steps=N_STEPS, gamma=GAMMA):
        self.buffer = deque(maxlen=capacity)
        self.n_step_buffer = deque(maxlen=n_steps)
        self.n_steps = n_steps
        self.gamma = gamma
    
    def add(self, state, action, reward, next_state, done):
        # 添加到n步缓冲区
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        # 如果n步缓冲区未满且当前转换非终止状态，不存储到主缓冲区
        if len(self.n_step_buffer) < self.n_steps and not done:
            return
        
        # 计算n步回报并存储
        n_step_state, n_step_action, n_step_reward, n_step_next_state, n_step_done = self._get_n_step_info()
        self.buffer.append((n_step_state, n_step_action, n_step_reward, n_step_next_state, n_step_done))
        
        # 如果当前是终止状态，清空n步缓冲区
        if done:
            self.n_step_buffer.clear()
    
    def _get_n_step_info(self):
        """计算n步回报"""
        # 获取第一个状态和动作
        first_transition = self.n_step_buffer[0]
        state = first_transition[0]
        action = first_transition[1]
        
        # 计算累积奖励
        reward = 0
        for i, transition in enumerate(self.n_step_buffer):
            r = transition[2]
            reward += (self.gamma ** i) * r
            if transition[4]:  # 如果有done=True，停止累积
                break
                
        # 获取最后一个状态
        last_transition = self.n_step_buffer[-1]
        next_state = last_transition[3]
        done = last_transition[4]
        
        return state, action, reward, next_state, done
    
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
    """基础DQN智能体"""
    def __init__(self, state_size, action_size, use_n_step=False, n_steps=N_STEPS):
        self.state_size = state_size
        self.action_size = action_size
        self.use_n_step = use_n_step
        self.n_steps = n_steps
        
        # 超参数 - 将这部分移到前面，确保 gamma 优先初始化
        self.gamma = GAMMA
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.batch_size = BATCH_SIZE
        self.train_start = 500  # 开始训练的最小经验数量
        self.update_target_every = UPDATE_TARGET_EVERY
        
        # 选择回放缓冲区类型
        if use_n_step:
            self.memory = NStepReplayBuffer(n_steps=n_steps)
            self.gamma_n = self.gamma ** n_steps  # n步折扣因子
        else:
            self.memory = StandardReplayBuffer()
            self.gamma_n = self.gamma  # 单步折扣因子
        
        # Q网络
        self.q_network = create_q_network(state_size, action_size)
        self.target_network = create_q_network(state_size, action_size)
        self.update_target_network()
        
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
        targets = rewards + (1 - dones) * self.gamma_n * max_next_q
        
        # 记录平均目标Q值
        self.target_q_values.append(np.mean(targets))
        
        # 获取当前预测
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
            print(f"回合 {episode}/{n_episodes}, 分数: {score:.2f}, 平均分数: {avg_score:.2f}, ε: {agent.epsilon:.4f}, 目标Q值: {agent.get_avg_target_q():.2f}")
    
    print()  # 新行
    return scores, avg_scores, target_q_values

def plot_comparison(std_scores, n_step_scores, std_q, n_step_q):
    """绘制比较图"""
    # 设置字体
    set_chinese_font()
    
    # 使用英文输入表示算法名称
    std_label = 'Standard DQN'
    n_step_label = f'{N_STEPS}-Step DQN'
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 12), dpi=100)
    
    # 绘制分数
    axes[0].plot(std_scores, label=std_label, color='blue', linewidth=2)
    axes[0].plot(n_step_scores, label=n_step_label, color='red', linewidth=2)
    axes[0].set_title('标准DQN与多步学习DQN的性能比较')
    axes[0].set_xlabel('回合')
    axes[0].set_ylabel('得分')
    axes[0].legend()
    axes[0].grid(True)
    
    # 绘制目标Q值
    axes[1].plot(std_q, label=f'{std_label}的目标Q值', color='blue', linewidth=2)
    axes[1].plot(n_step_q, label=f'{n_step_label}的目标Q值', color='red', linewidth=2)
    axes[1].set_title('目标Q值比较 (多步学习通常有更大的目标Q值)')
    axes[1].set_xlabel('回合')
    axes[1].set_ylabel('平均目标Q值')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('n_step_learning_comparison.png')
    print("比较图已保存为 'n_step_learning_comparison.png'")
    
    # 额外生成一个使用英文的图（以防中文显示仍有问题）
    fig, axes = plt.subplots(2, 1, figsize=(10, 12), dpi=100)
    
    # 绘制分数
    axes[0].plot(std_scores, label=std_label, color='blue', linewidth=2)
    axes[0].plot(n_step_scores, label=n_step_label, color='red', linewidth=2)
    axes[0].set_title(f'Performance Comparison: Standard DQN vs {N_STEPS}-Step DQN')
    axes[0].set_xlabel('Episodes')
    axes[0].set_ylabel('Score')
    axes[0].legend()
    axes[0].grid(True)
    
    # 绘制目标Q值
    axes[1].plot(std_q, label=f'{std_label} Target Q-Values', color='blue', linewidth=2)
    axes[1].plot(n_step_q, label=f'{n_step_label} Target Q-Values', color='red', linewidth=2)
    axes[1].set_title('Target Q-Value Comparison (N-step learning typically has larger target Q-values)')
    axes[1].set_xlabel('Episodes')
    axes[1].set_ylabel('Average Target Q-Value')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('n_step_learning_comparison_english.png')
    print("英文版比较图已保存为 'n_step_learning_comparison_english.png'")

def main():
    # 创建环境
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    print(f"环境: CartPole-v1, 状态空间: {state_size}, 动作空间: {action_size}")
    print(f"开始训练标准DQN和{N_STEPS}步学习DQN进行比较，每个算法训练{NUM_EPISODES}回合...")
    
    # 训练标准DQN
    print("\n训练标准DQN...")
    standard_agent = DQNAgent(state_size, action_size, use_n_step=False)
    std_scores, std_avg_scores, std_q = train_agent(standard_agent, env)
    
    # 训练多步学习DQN
    print(f"\n训练{N_STEPS}步学习DQN...")
    n_step_agent = DQNAgent(state_size, action_size, use_n_step=True, n_steps=N_STEPS)
    n_step_scores, n_step_avg_scores, n_step_q = train_agent(n_step_agent, env)
    
    # 比较结果
    print("\n===== 结果比较 =====")
    print(f"标准DQN最终平均分数: {std_avg_scores[-1]:.2f}")
    print(f"{N_STEPS}步学习DQN最终平均分数: {n_step_avg_scores[-1]:.2f}")
    print(f"标准DQN平均目标Q值: {np.mean(std_q):.2f}")
    print(f"{N_STEPS}步学习DQN平均目标Q值: {np.mean(n_step_q):.2f}")
    
    # 绘制比较图
    plot_comparison(std_scores, n_step_scores, std_q, n_step_q)
    
    print("\n完成比较！")

if __name__ == "__main__":
    main() 