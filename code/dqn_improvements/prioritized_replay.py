"""
优先经验回放 (Prioritized Experience Replay) 实现

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
from IPython.display import clear_output
import os
from collections import deque
import platform # 导入 platform 模块

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

# 设置随机种子以获得可重复的结果
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

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
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.tree = SumTree(capacity)
        self.alpha = alpha  # 控制多大程度上依赖TD误差，alpha=0意味着均匀采样
        self.beta = beta_start  # 重要性采样权重系数，随着训练逐渐增加到1
        self.beta_increment = (1.0 - beta_start) / beta_frames
        self.epsilon = 1e-6  # 小常数，避免优先级为0
        self.max_priority = 1.0  # 初始最大优先级
    
    def add(self, state, action, reward, next_state, done):
        """添加新样本到回放缓冲区"""
        # 将样本封装为元组
        experience = (state, action, reward, next_state, done)
        
        # 新样本使用最大优先级（确保至少被采样一次）
        priority = self.max_priority ** self.alpha
        
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
        
        # 为每个样本计算优先级区间
        segment = self.tree.total_priority / batch_size
        
        # 增加beta以减少重要性采样权重的影响
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # 计算最小优先级（用于归一化权重）
        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
        
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
                np.array(dones, dtype=np.float32),
                indices,
                weights)
    
    def update_priorities(self, indices, errors):
        """
        更新样本优先级
        
        参数:
        - indices: 样本在SumTree中的索引
        - errors: TD误差绝对值
        """
        for idx, error in zip(indices, errors):
            # 添加小常数避免优先级为0
            priority = (error + self.epsilon) ** self.alpha
            
            # 更新最大优先级
            self.max_priority = max(self.max_priority, priority)
            
            # 更新SumTree
            self.tree.update(idx, priority)
    
    def __len__(self):
        """返回当前存储的样本数量"""
        return self.tree.size


class StandardReplayBuffer:
    """标准经验回放缓冲区（用于比较）"""
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


def create_q_network(state_size, action_size, hidden_size=64):
    """创建Q网络"""
    model = keras.Sequential([
        layers.Dense(hidden_size, activation='relu', input_shape=(state_size,)),
        layers.Dense(hidden_size, activation='relu'),
        layers.Dense(action_size)
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                 loss='mse')
    return model


class PrioritizedReplayDQNAgent:
    """带优先经验回放的DQN智能体"""
    def __init__(self, state_size, action_size, hidden_size=64, use_double_dqn=True):
        self.state_size = state_size
        self.action_size = action_size
        self.use_double_dqn = use_double_dqn
        
        # Q网络 - 主网络和目标网络
        self.q_network = create_q_network(state_size, action_size, hidden_size)
        self.target_network = create_q_network(state_size, action_size, hidden_size)
        self.update_target_network()  # 初始化目标网络权重
        
        # 优先经验回放
        self.memory = PrioritizedReplayBuffer(capacity=10000)
        
        # 学习超参数
        self.gamma = 0.99  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.train_start = 1000
        self.update_target_every = 1000
        
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
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)
            
            # 更新epsilon值
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # 定期更新目标网络
            if self.train_step % self.update_target_every == 0:
                self.update_target_network()
                print(f"\n目标网络已更新。当前epsilon: {self.epsilon:.4f}")
    
    def learn(self, experiences):
        """从经验中更新值函数"""
        states, actions, rewards, next_states, dones, indices, weights = experiences
        
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
        
        # 获取当前的Q值预测
        q_values = self.q_network.predict(states, verbose=0)
        
        # 计算TD误差用于更新优先级
        td_errors = []
        
        # 更新目标Q值并记录TD误差
        for i, action in enumerate(actions):
            old_val = q_values[i][action]
            q_values[i][action] = targets[i]
            td_errors.append(abs(old_val - targets[i]))
        
        # 使用重要性采样权重进行加权训练
        self.q_network.fit(states, q_values, 
                           sample_weight=weights,
                           epochs=1, verbose=0)
        
        # 更新样本优先级
        self.memory.update_priorities(indices, td_errors)
    
    def save(self, filepath):
        """保存模型"""
        self.q_network.save_weights(filepath)
        
    def load(self, filepath):
        """加载模型"""
        self.q_network.load_weights(filepath)
        self.update_target_network()

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

class StandardDQNAgent(PrioritizedReplayDQNAgent):
    """标准DQN智能体（用于比较）"""
    def __init__(self, state_size, action_size, hidden_size=64, use_double_dqn=True):
        super().__init__(state_size, action_size, hidden_size, use_double_dqn)
        # 替换为标准回放缓冲区
        self.memory = StandardReplayBuffer(capacity=10000)
    
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
        
        # 更新目标Q值
        for i, action in enumerate(actions):
            q_values[i][action] = targets[i]
        
        # 训练网络
        self.q_network.fit(states, q_values, epochs=1, verbose=0)


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
                target_score=195.0, print_every=20):
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
        print(f'\r回合 {i_episode}/{n_episodes}, 分数: {score:.2f}, 平均分数: {avg_score:.2f}, epsilon: {agent.epsilon:.4f}', end='')
        
        if i_episode % print_every == 0:
            print(f'\n回合 {i_episode}/{n_episodes}, 分数: {score:.2f}, 平均分数: {avg_score:.2f}, epsilon: {agent.epsilon:.4f}')
            # 确保绘图时字体已设置
            if isinstance(agent.memory, PrioritizedReplayBuffer):
                 plot_scores(scores, avg_scores, title=f"{env_name} - Prioritized Replay DQN学习曲线 (beta: {agent.memory.beta:.4f})")
            else:
                 plot_scores(scores, avg_scores, title=f"{env_name} - Standard DQN学习曲线 (epsilon: {agent.epsilon:.4f})")
            
        # 检查是否达到目标
        if avg_score >= target_score and i_episode >= 100:
            print(f'\n环境在{i_episode}回合后解决!')
            save_dir = './models'
            os.makedirs(save_dir, exist_ok=True)
            agent.save(f'{save_dir}/{agent.__class__.__name__.lower()}_{env_name}_{i_episode}.h5')
            break
    
    return scores, avg_scores

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

def compare_standard_vs_prioritized_replay(env_name='CartPole-v1', n_episodes=300):
    """比较标准DQN和带优先经验回放的DQN性能"""
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # 创建智能体
    standard_agent = StandardDQNAgent(state_size, action_size, use_double_dqn=True)
    prioritized_agent = PrioritizedReplayDQNAgent(state_size, action_size, use_double_dqn=True)
    
    # 训练标准DQN
    print("训练标准DQN...")
    standard_scores, standard_avg_scores = train_agent(
        standard_agent, env_name, n_episodes=n_episodes, print_every=100)
    
    # 训练带优先经验回放的DQN
    print("\n训练带优先经验回放的DQN...")
    prioritized_scores, prioritized_avg_scores = train_agent(
        prioritized_agent, env_name, n_episodes=n_episodes, print_every=100)
    
    # 绘制比较图
    set_chinese_font() # 绘图前设置字体
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(standard_scores)
    plt.plot(prioritized_scores)
    plt.title(f'{env_name} - 回合分数比较')
    plt.xlabel('回合数')
    plt.ylabel('分数')
    plt.legend(['标准DQN', '优先经验回放DQN'])
    
    plt.subplot(1, 2, 2)
    plt.plot(standard_avg_scores)
    plt.plot(prioritized_avg_scores)
    plt.title(f'{env_name} - 平均分数比较 (最近100回合)')
    plt.xlabel('回合数')
    plt.ylabel('平均分数')
    plt.legend(['标准DQN', '优先经验回放DQN'])
    
    plt.tight_layout()
    plt.savefig(f'./standard_vs_prioritized_replay_{env_name}.png')
    plt.show()
    
    return standard_scores, standard_avg_scores, prioritized_scores, prioritized_avg_scores


if __name__ == "__main__":
    # 环境设置
    ENV_NAME = 'CartPole-v1'
    
    # 创建环境以获取状态和动作空间大小
    temp_env = gym.make(ENV_NAME)
    state_size = temp_env.observation_space.shape[0]
    action_size = temp_env.action_space.n
    
    print(f"环境: {ENV_NAME}")
    print(f"状态空间: {state_size}")
    print(f"动作空间: {action_size}")
    
    # 创建带优先经验回放的DQN智能体
    agent = PrioritizedReplayDQNAgent(state_size, action_size, hidden_size=64, use_double_dqn=True)
    
    # 训练智能体
    scores, avg_scores = train_agent(
        agent, 
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
    plt.title(f'{ENV_NAME} - 优先经验回放DQN学习曲线')
    plt.xlabel('回合数')
    plt.ylabel('分数')
    plt.legend(['回合分数', '平均分数 (最近100回合)'])
    plt.savefig(f'./prioritized_replay_dqn_{ENV_NAME}_learning_curve.png')
    plt.show()
    
    # # 进行性能比较实验
    # print("\n开始比较标准DQN和带优先经验回放的DQN性能...")
    # compare_standard_vs_prioritized_replay(ENV_NAME, n_episodes=300)
    
    # 评估训练好的智能体
    eval_scores = evaluate_agent(agent, ENV_NAME, render=True, n_episodes=5) 