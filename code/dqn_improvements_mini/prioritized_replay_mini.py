"""
优先经验回放（Prioritized Experience Replay）实现（微型版 - Blackjack环境）

优先经验回放是DQN的一个重要改进，它根据经验的TD误差来为样本分配优先级，
使得具有较高TD误差的样本被更频繁地采样，从而提高学习效率。
同时，通过重要性采样权重来修正引入的偏差。

参考文献: https://arxiv.org/abs/1511.05952
"""

import gymnasium as gym
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import platform
import time
import os
from collections import deque

# 设置随机种子以获得可重复的结果
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 禁用GPU以确保在所有学生机器上一致运行
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  

class SumTree:
    """
    SumTree数据结构用于高效采样优先级样本
    
    SumTree是一个二叉树，叶节点保存样本的优先级值，内部节点保存子树中所有优先级的和。
    这种结构可以高效地进行基于优先级的采样。
    """
    def __init__(self, capacity):
        self.capacity = capacity  # 回放缓冲区容量（叶子节点数量）
        self.tree = np.zeros(2 * capacity - 1)  # 总共需要2*capacity-1个节点
        self.data = np.zeros(capacity, dtype=object)  # 数据存储
        self.data_pointer = 0  # 指向当前要替换的数据位置
        self.size = 0  # 当前存储的样本数量
    
    def add(self, priority, data):
        """添加新样本"""
        # 找到插入数据的位置
        tree_index = self.data_pointer + self.capacity - 1
        
        # 更新数据
        self.data[self.data_pointer] = data
        
        # 更新优先级
        self.update(tree_index, priority)
        
        # 更新指针
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        
        # 更新大小
        if self.size < self.capacity:
            self.size += 1
    
    def update(self, tree_index, priority):
        """更新优先级"""
        # 计算变化量
        change = priority - self.tree[tree_index]
        
        # 更新叶子节点
        self.tree[tree_index] = priority
        
        # 向上传播变化
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change
    
    def get_leaf(self, v):
        """
        获取叶子节点
        
        参数:
        - v: 在[0, total_priority]范围内的一个值
        
        返回:
        - leaf_index: 叶子节点索引
        - priority: 叶子节点的优先级
        - data: 叶子节点存储的样本数据
        """
        parent_index = 0
        
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            
            # 如果到达叶子节点，则停止
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            
            # 否则，向下遍历树
            if v <= self.tree[left_child_index]:
                parent_index = left_child_index
            else:
                v -= self.tree[left_child_index]
                parent_index = right_child_index
        
        data_index = leaf_index - self.capacity + 1
        
        return leaf_index, self.tree[leaf_index], self.data[data_index]
    
    @property
    def total_priority(self):
        """获取总优先级"""
        return self.tree[0]

class PrioritizedReplayBuffer:
    """
    优先经验回放缓冲区
    
    使用SumTree实现样本存储和基于优先级的抽样。
    """
    def __init__(self, capacity=2000, alpha=0.6, beta_start=0.4, beta_frames=20000):
        self.tree = SumTree(capacity)
        self.alpha = alpha  # 控制多大程度上依赖TD误差，alpha=0意味着均匀采样
        self.beta = beta_start  # 重要性采样权重系数，随着训练逐渐增加到1
        self.beta_increment = (1.0 - beta_start) / beta_frames
        self.epsilon = 1e-5  # 小常数，避免优先级为0
        self.max_priority = 1.0  # 初始最大优先级
    
    def add(self, state, action, reward, next_state, done):
        """添加新样本到回放缓冲区"""
        # 将样本封装为元组
        experience = (state, action, reward, next_state, done)
        
        # 新样本使用最大优先级（确保至少被采样一次）
        priority = (self.max_priority + self.epsilon) ** self.alpha
        
        # 添加到SumTree
        self.tree.add(priority, experience)
    
    def sample(self, batch_size):
        """
        根据优先级采样batch_size个样本
        
        返回:
        - batch: 包含样本的元组(states, actions, rewards, next_states, dones)
        - indices: 样本在SumTree中的索引，用于后续更新
        - weights: 重要性采样权重，用于修正偏差
        """
        batch = []
        indices = []
        weights = np.zeros(batch_size, dtype=np.float32)
        
        # 计算优先级区间
        segment = self.tree.total_priority / batch_size
        
        # 增加beta以减少重要性采样权重的影响
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # 计算最小概率（对应于最小优先级，用于归一化权重）
        min_prob = np.min(self.tree.tree[-self.tree.capacity:] / self.tree.total_priority + self.epsilon)
        
        for i in range(batch_size):
            # 在区间内随机选择一个值
            a = segment * i
            b = segment * (i + 1)
            v = np.random.uniform(a, b)
            
            # 获取对应的叶子节点
            index, priority, data = self.tree.get_leaf(v)
            
            # 计算采样概率
            sampling_prob = priority / self.tree.total_priority
            
            # 计算权重
            weights[i] = (sampling_prob * self.tree.size) ** (-self.beta)
            
            indices.append(index)
            batch.append(data)
        
        # 归一化权重
        weights /= weights.max()
        
        # 解包样本批次
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (np.array(states), 
                np.array(actions), 
                np.array(rewards), 
                np.array(next_states), 
                np.array(dones, dtype=np.float32)), indices, weights
    
    def update_priorities(self, indices, priorities):
        """更新样本优先级"""
        for idx, priority in zip(indices, priorities):
            # 应用alpha参数来控制重要性
            priority = (priority + self.epsilon) ** self.alpha
            
            # 更新最大优先级
            self.max_priority = max(self.max_priority, priority)
            
            # 更新树中的优先级
            self.tree.update(idx, priority)
    
    def __len__(self):
        return self.tree.size

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

class PrioritizedDQNAgent:
    """使用优先经验回放的DQN智能体"""
    def __init__(self, state_size, action_size, use_double_dqn=True):
        self.state_size = state_size
        self.action_size = action_size
        self.use_double_dqn = use_double_dqn
        
        # Q网络 - 主网络和目标网络
        self.q_network = create_q_network(state_size, action_size)
        self.target_network = create_q_network(state_size, action_size)
        self.update_target_network()  # 初始化目标网络权重
        
        # 优先经验回放
        self.memory = PrioritizedReplayBuffer()
        
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
        # 添加到回放缓冲区
        self.memory.add(state, action, reward, next_state, done)
        
        # 如果缓冲区足够大，开始学习
        if len(self.memory) > self.train_start:
            self.train_step += 1
            
            # 从回放缓冲区中采样
            experiences, indices, weights = self.memory.sample(self.batch_size)
            self.learn(experiences, indices, weights)
            
            # 更新epsilon值
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # 定期更新目标网络
            if self.train_step % self.update_target_every == 0:
                self.update_target_network()
                print(f"目标网络已更新。当前epsilon: {self.epsilon:.4f}")
    
    def learn(self, experiences, indices, weights):
        """从经验中更新值函数"""
        states, actions, rewards, next_states, dones = experiences
        
        if self.use_double_dqn:
            # Double DQN: 使用主网络选择动作，目标网络评估动作
            next_actions = np.argmax(self.q_network.predict(next_states, verbose=0), axis=1)
            next_q_values = self.target_network.predict(next_states, verbose=0)
            max_next_q = np.array([next_q_values[i, action] for i, action in enumerate(next_actions)])
        else:
            # 标准DQN: 使用目标网络选择和评估动作
            next_q_values = self.target_network.predict(next_states, verbose=0)
            max_next_q = np.max(next_q_values, axis=1)
            
        # 计算目标Q值
        targets = rewards + (1 - dones) * self.gamma * max_next_q
        
        # 获取当前的Q值预测，用于计算TD误差
        q_values = self.q_network.predict(states, verbose=0)
        
        # 存储TD误差，用于更新优先级
        td_errors = np.zeros(len(indices))
        
        # 创建目标批次
        target_batch = q_values.copy()
        
        # 为每个样本更新目标值并计算TD误差
        for i, action in enumerate(actions):
            # 计算TD误差
            td_errors[i] = abs(targets[i] - q_values[i, action])
            
            # 应用重要性采样权重
            target_batch[i, action] = targets[i]
        
        # 训练网络
        self.q_network.fit(states, target_batch, 
                           sample_weight=weights,
                           epochs=1, verbose=0)
        
        # 更新优先级
        self.memory.update_priorities(indices, td_errors)
    
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

class StandardDQNAgent:
    """标准DQN智能体（用于比较）"""
    def __init__(self, state_size, action_size, use_double_dqn=True):
        self.state_size = state_size
        self.action_size = action_size
        self.use_double_dqn = use_double_dqn
        
        # Q网络 - 主网络和目标网络
        self.q_network = create_q_network(state_size, action_size)
        self.target_network = create_q_network(state_size, action_size)
        self.update_target_network()  # 初始化目标网络权重
        
        # 标准经验回放
        self.memory = deque(maxlen=2000)
        
        # 学习超参数
        self.gamma = 0.99  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.1  # 最小探索率
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
        # 添加到回放缓冲区
        self.memory.add(state, action, reward, next_state, done)
        
        # 如果缓冲区足够大，开始学习
        if len(self.memory) > self.train_start:
            self.train_step += 1
            
            # 从回放缓冲区中采样
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            experiences = (np.array(states), 
                         np.array(actions), 
                         np.array(rewards), 
                         np.array(next_states), 
                         np.array(dones, dtype=np.float32))
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
        
        if self.use_double_dqn:
            # Double DQN
            next_actions = np.argmax(self.q_network.predict(next_states, verbose=0), axis=1)
            next_q_values = self.target_network.predict(next_states, verbose=0)
            max_next_q = np.array([next_q_values[i, action] for i, action in enumerate(next_actions)])
        else:
            # 标准DQN
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
    current_buffer_size = len(agent.memory)
    
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
            
            # 学习
            agent.step(state, action, reward, next_state, done)
            current_buffer_size = len(agent.memory)
            
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
            print(f'回合 {i_episode}/{n_episodes}\t分数: {score}\t平均: {avg_score:.3f}\t探索率: {agent.epsilon:.4f}\t{buffer_status}\t用时: {elapsed_time:.1f}秒')
            
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

def compare_standard_vs_prioritized_dqn(env_name='Blackjack-v1', n_episodes=20000, eval_freq=1000):
    """比较标准DQN和使用优先经验回放的DQN的性能"""
    # 创建环境
    env = gym.make(env_name)
    state_size = 3  # [玩家点数, 庄家牌点, 是否有可用A]
    action_size = 2  # [停牌, 要牌]
    
    # 创建两种智能体，两者都使用Double DQN以减少对比变量
    standard_agent = StandardDQNAgent(state_size, action_size, use_double_dqn=True)
    prioritized_agent = PrioritizedDQNAgent(state_size, action_size, use_double_dqn=True)
    
    # 训练标准DQN
    print("\n开始训练标准DQN智能体...")
    standard_scores, standard_avg_scores, standard_winrates = train_agent(standard_agent, env_name, n_episodes, eval_freq)
    
    print("\n开始训练使用优先经验回放的DQN智能体...")
    prioritized_scores, prioritized_avg_scores, prioritized_winrates = train_agent(prioritized_agent, env_name, n_episodes, eval_freq)
    
    # 绘制比较学习曲线图
    set_chinese_font()
    plt.figure(figsize=(12, 6))
    
    # 平滑处理的分数
    plt.plot(standard_avg_scores, label='标准DQN平均分数')
    plt.plot(prioritized_avg_scores, label='优先经验回放DQN平均分数')
    
    plt.xlabel('回合')
    plt.ylabel('平均分数')
    plt.title('标准DQN vs. 优先经验回放DQN 学习曲线比较')
    plt.legend()
    plt.tight_layout()
    plt.savefig("standard_vs_prioritized_dqn_learning_curve.png")
    plt.show()
    
    # 绘制比较获胜率图
    plt.figure(figsize=(12, 6))
    
    x_eval = [(i+1)*eval_freq for i in range(len(standard_winrates))]
    
    plt.plot(x_eval, standard_winrates, label='标准DQN获胜率', marker='o')
    plt.plot(x_eval, prioritized_winrates, label='优先经验回放DQN获胜率', marker='s')
    
    plt.xlabel('训练回合')
    plt.ylabel('获胜率')
    plt.title('标准DQN vs. 优先经验回放DQN 获胜率比较')
    plt.legend()
    plt.tight_layout()
    plt.savefig("standard_vs_prioritized_dqn_winrate.png")
    plt.show()
    
    # 最终评估
    print("\n最终评估标准DQN:")
    standard_eval = evaluate_agent(standard_agent, env_name)
    
    print("\n最终评估优先经验回放DQN:")
    prioritized_eval = evaluate_agent(prioritized_agent, env_name)
    
    print(f"\n标准DQN最终获胜率: {standard_eval:.3f}")
    print(f"优先经验回放DQN最终获胜率: {prioritized_eval:.3f}")
    
    # 保存模型
    standard_agent.save("standard_dqn_blackjack_model.h5")
    prioritized_agent.save("prioritized_dqn_blackjack_model.h5")
    
    return (standard_scores, standard_avg_scores, standard_winrates,
            prioritized_scores, prioritized_avg_scores, prioritized_winrates)

if __name__ == "__main__":
    # 环境名称
    env_name = 'Blackjack-v1'
    
    # 使用对比模式
    compare_standard_vs_prioritized_dqn(env_name)
    
    """
    # 或者只运行优先经验回放DQN
    
    # 创建环境
    env = gym.make(env_name)
    state_size = 3  # [玩家点数, 庄家牌点, 是否有可用A]
    action_size = 2  # [停牌, 要牌]
    
    print(f"状态向量大小: {state_size}, 动作空间大小: {action_size}")
    
    # 创建智能体
    agent = PrioritizedDQNAgent(state_size, action_size)
    
    # 训练智能体
    print("开始训练优先经验回放DQN智能体...")
    scores, avg_scores, winrates = train_agent(agent, env_name)
    
    # 绘制并保存学习曲线
    plot_scores(scores, avg_scores, 
                "优先经验回放DQN在Blackjack环境中的学习曲线", 
                "prioritized_dqn_blackjack_learning_curve.png")
    
    # 绘制并保存获胜率曲线
    plot_winrate(winrates, 
                "优先经验回放DQN在Blackjack环境中的获胜率", 
                "prioritized_dqn_blackjack_winrate.png")
    
    # 最终评估
    print("\n最终评估:")
    evaluate_agent(agent, env_name)
    
    # 保存模型
    agent.save("prioritized_dqn_blackjack_model.h5")
    """ 