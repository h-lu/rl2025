"""
基础DQN实现（微型版 - Blackjack环境）

这是一个为Blackjack-v1环境优化的轻量级DQN实现。
该版本采用了小型网络架构和有限的回放缓冲区，便于在笔记本电脑上快速运行。
"""

import gymnasium as gym
import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import platform
import time
import os

# 设置随机种子以获得可重复的结果
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 禁用GPU以确保在所有学生机器上一致运行
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  

class ReplayBuffer:
    """简单的经验回放缓冲区"""
    def __init__(self, capacity=2000):
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

def create_q_network(input_dim, action_size):
    """创建一个简单的Q网络，针对Blackjack环境优化"""
    model = keras.Sequential([
        layers.Dense(16, activation='relu', input_shape=(input_dim,)),
        layers.Dense(16, activation='relu'),
        layers.Dense(action_size)
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse')
    return model

class DQNAgent:
    """基本的DQN智能体"""
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Q网络 - 主网络和目标网络
        self.q_network = create_q_network(state_size, action_size)
        self.target_network = create_q_network(state_size, action_size)
        self.update_target_network()  # 初始化目标网络权重
        
        # 经验回放
        self.memory = ReplayBuffer()
        
        # 学习超参数
        self.gamma = 0.99  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.1  # 最小探索率（在Blackjack中保持较高的探索）
        self.epsilon_decay = 0.995
        self.batch_size = 32
        self.train_start = 500
        self.update_target_every = 500
        
        # 跟踪训练步数
        self.train_step = 0
    
    def update_target_network(self):
        """更新目标网络权重为主网络权重"""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def act(self, state, eval_mode=False):
        """根据当前状态选择动作"""
        # epsilon-greedy策略
        if not eval_mode and np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        
        state = np.reshape(state, [1, self.state_size])
        q_values = self.q_network.predict(state, verbose=0)[0]
        return np.argmax(q_values)
    
    def step(self, state, action, reward, next_state, done):
        """在环境中执行一步并学习"""
        # 将经验添加到缓冲区
        self.memory.add(state, action, reward, next_state, done)
        current_buffer_size = len(self.memory)
        
        # 如果缓冲区足够大，开始学习
        if current_buffer_size > self.train_start:
            self.train_step += 1
            
            # 从回放缓冲区中采样
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)
            
            # 更新epsilon值
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # 定期更新目标网络
            if self.train_step % self.update_target_every == 0:
                self.update_target_network()
                print(f"目标网络已更新。当前epsilon: {self.epsilon:.4f}")
    
    def learn(self, experiences):
        """从经验中更新值函数"""
        states, actions, rewards, next_states, dones = experiences
        
        # 计算下一个状态的最大Q值
        next_q_values = self.target_network.predict(next_states, verbose=0)
        max_next_q = np.max(next_q_values, axis=1)
        
        # 计算目标Q值
        targets = rewards + (1 - dones) * self.gamma * max_next_q
        
        # 获取当前的Q值预测
        q_values = self.q_network.predict(states, verbose=0)
        
        # 仅更新所选动作的Q值
        for i, action in enumerate(actions):
            q_values[i][action] = targets[i]
        
        # 训练网络
        self.q_network.fit(states, q_values, epochs=1, verbose=0)
    
    def save(self, filepath):
        """保存模型"""
        if not filepath.endswith('.weights.h5'):
            filepath = filepath.replace('.h5', '.weights.h5')
        self.q_network.save_weights(filepath)
        
    def load(self, filepath):
        """加载模型"""
        if not filepath.endswith('.weights.h5'):
            filepath = filepath.replace('.h5', '.weights.h5')
        self.q_network.load_weights(filepath)
        self.update_target_network()

def preprocess_state(state):
    """
    预处理Blackjack环境状态
    将(玩家点数, 庄家牌点, 是否有可用A)转换为扁平向量
    """
    player_score, dealer_score, usable_ace = state
    return np.array([player_score, dealer_score, int(usable_ace)])

def set_chinese_font():
    """根据操作系统设置中文字体"""
    system = platform.system()
    try:
        if system == 'Windows':
            plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 黑体
            plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
        elif system == 'Darwin': # macOS
            plt.rcParams['font.sans-serif'] = ['PingFang SC'] # Mac 苹方
            plt.rcParams['axes.unicode_minus'] = False
        else: # Linux 或其他
            plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'Arial Unicode MS']
            plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        print(f"设置中文字体时出错: {e}")

def plot_scores(scores, avg_scores, title="学习曲线", save_path=None):
    """绘制学习曲线"""
    set_chinese_font()  # 设置中文字体
    plt.figure(figsize=(10, 5))
    plt.plot(scores, alpha=0.3, color='blue', label='分数')
    plt.plot(avg_scores, color='red', label='平均分数 (100回合)')
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    plt.xlabel('回合')
    plt.ylabel('分数')
    plt.title(title)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_winrate(winrates, title="获胜率", save_path=None):
    """绘制获胜率曲线"""
    set_chinese_font()  # 设置中文字体
    plt.figure(figsize=(10, 5))
    plt.plot(winrates, color='green', label='获胜率')
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='随机策略')
    plt.xlabel('评估次数')
    plt.ylabel('获胜率')
    plt.title(title)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def train_agent(agent, env_name='Blackjack-v1', n_episodes=20000, eval_freq=1000):
    """训练智能体"""
    start_time = time.time()  # 记录开始时间
    
    print("开始准备训练环境...")
    env = gym.make(env_name)
    scores = []
    avg_scores = []
    winrates = []
    
    print(f"开始填充回放缓冲区(目标: {agent.train_start}个样本)...")
    current_buffer_size = 0
    
    # 训练循环
    for i_episode in range(1, n_episodes+1):
        state, _ = env.reset(seed=SEED+i_episode)
        state = preprocess_state(state)
        done = False
        score = 0
        steps = 0
        
        # 单局游戏循环
        while not done:
            # 选择动作
            action = agent.act(state)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = preprocess_state(next_state)
            done = terminated or truncated
            
            # 将经验添加到缓冲区
            agent.memory.add(state, action, reward, next_state, done)
            current_buffer_size = len(agent.memory)
            
            # 如果缓冲区已经足够大，开始学习
            if current_buffer_size > agent.train_start and hasattr(agent, 'learn_step'):
                agent.learn_step += 1
            
            # 学习
            agent.step(state, action, reward, next_state, done)
            
            # 更新状态和分数
            state = next_state
            score += reward
            steps += 1
            
            if done:
                break
        
        # 记录分数
        scores.append(score)
        
        # 计算滑动平均
        window_size = min(100, len(scores))
        avg_score = np.mean(scores[-window_size:])
        avg_scores.append(avg_score)
        
        # 更频繁地打印进度
        if i_episode == 1 or i_episode % 100 == 0 or i_episode == n_episodes:
            elapsed_time = time.time() - start_time
            buffer_status = f"缓冲区: {current_buffer_size}/{agent.train_start}"
            if hasattr(agent, 'epsilon'):
                print(f'回合 {i_episode}/{n_episodes}\t分数: {score}\t平均: {avg_score:.3f}\t探索率: {agent.epsilon:.4f}\t{buffer_status}\t用时: {elapsed_time:.1f}秒')
            else:
                print(f'回合 {i_episode}/{n_episodes}\t分数: {score}\t平均: {avg_score:.3f}\t{buffer_status}\t用时: {elapsed_time:.1f}秒')
            
        # 定期评估
        if i_episode % eval_freq == 0:
            print(f"\n开始第{i_episode//eval_freq}次评估 ({eval_freq}回合训练后)...")
            win_rate = evaluate_agent(agent, env_name, n_episodes=1000, verbose=False)
            winrates.append(win_rate)
            print(f'评估完成: 获胜率: {win_rate:.3f}\n')
    
    # 记录结束时间和总训练时间
    total_time = time.time() - start_time
    print(f"\n训练完成! 总用时: {total_time:.2f}秒")
    
    return scores, avg_scores, winrates

def evaluate_agent(agent, env_name='Blackjack-v1', n_episodes=1000, verbose=True):
    """评估智能体"""
    env = gym.make(env_name)
    wins = 0
    draws = 0
    losses = 0
    
    for i in range(n_episodes):
        state, _ = env.reset(seed=SEED+i+10000)
        state = preprocess_state(state)
        done = False
        
        while not done:
            action = agent.act(state, eval_mode=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = preprocess_state(next_state)
            done = terminated or truncated
            state = next_state
        
        if reward > 0:
            wins += 1
        elif reward == 0:
            draws += 1
        else:
            losses += 1
    
    win_rate = wins / n_episodes
    draw_rate = draws / n_episodes
    loss_rate = losses / n_episodes
    
    if verbose:
        print(f'评估 {n_episodes} 回合:')
        print(f'获胜率: {win_rate:.3f}')
        print(f'平局率: {draw_rate:.3f}')
        print(f'失败率: {loss_rate:.3f}')
    
    return win_rate

if __name__ == "__main__":
    # 环境名称
    env_name = 'Blackjack-v1'
    
    # 创建环境
    env = gym.make(env_name)
    # Blackjack状态是玩家点数(0-31)、庄家牌点(1-10)和是否有可用A
    # 我们使用向量表示，所以是3个状态值
    state_size = 3  
    # Blackjack动作是要牌(1)或停牌(0)
    action_size = 2  
    
    print(f"状态向量大小: {state_size}, 动作空间大小: {action_size}")
    
    # 创建智能体
    agent = DQNAgent(state_size, action_size)
    
    # 训练智能体
    print("开始训练DQN智能体...")
    scores, avg_scores, winrates = train_agent(agent, env_name)
    
    # 绘制并保存学习曲线
    plot_scores(scores, avg_scores, 
                "DQN在Blackjack环境中的学习曲线", 
                "dqn_blackjack_learning_curve.png")
    
    # 绘制并保存获胜率曲线
    plot_winrate(winrates, 
                "DQN在Blackjack环境中的获胜率", 
                "dqn_blackjack_winrate.png")
    
    # 最终评估
    print("\n最终评估:")
    evaluate_agent(agent, env_name)
    
    # 保存模型
    agent.save("dqn_blackjack_model.h5") 