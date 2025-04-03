"""
噪声网络 (Noisy Networks) 实现

噪声网络是一种通过向网络层的权重和偏置添加参数化噪声来实现探索的方法。
这种方法消除了传统的epsilon-greedy探索策略的需求，使得探索行为可以根据环境状态自适应地调整。

参考文献: https://arxiv.org/abs/1706.10295
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

class NoisyDense(layers.Layer):
    """
    噪声网络层实现
    
    用参数化的噪声代替传统Dense层的权重和偏置，以实现自适应探索。
    """
    def __init__(self, units, activation=None, sigma_init=0.017):
        super(NoisyDense, self).__init__()
        self.units = units
        self.activation = activation
        self.sigma_init = sigma_init
        
    def build(self, input_shape):
        self.input_dim = input_shape[-1]
        
        # 初始化权重的均值参数
        mu_init = tf.random_uniform_initializer(minval=-1/np.sqrt(self.input_dim), 
                                             maxval=1/np.sqrt(self.input_dim))
        self.weight_mu = self.add_weight(
            name="weight_mu",
            shape=(self.input_dim, self.units),
            initializer=mu_init,
            trainable=True
        )
        
        # 初始化偏置的均值参数
        self.bias_mu = self.add_weight(
            name="bias_mu",
            shape=(self.units,),
            initializer=mu_init,
            trainable=True
        )
        
        # 初始化权重的噪声尺度参数
        sigma_init = tf.constant_initializer(self.sigma_init / np.sqrt(self.input_dim))
        self.weight_sigma = self.add_weight(
            name="weight_sigma",
            shape=(self.input_dim, self.units),
            initializer=sigma_init,
            trainable=True
        )
        
        # 初始化偏置的噪声尺度参数
        self.bias_sigma = self.add_weight(
            name="bias_sigma",
            shape=(self.units,),
            initializer=sigma_init,
            trainable=True
        )
    
    def call(self, inputs, training=True):
        """
        前向传播
        
        训练模式下添加噪声，评估模式下使用均值参数
        """
        if training:
            # 训练模式：使用参数化噪声
            # 生成噪声
            eps_in = tf.random.normal((self.input_dim,))
            eps_out = tf.random.normal((self.units,))
            
            # 使用因子化噪声技巧
            # f(x) = sgn(x) * sqrt(|x|)
            f_eps_in = tf.multiply(tf.sign(eps_in), tf.sqrt(tf.abs(eps_in)))
            f_eps_out = tf.multiply(tf.sign(eps_out), tf.sqrt(tf.abs(eps_out)))
            
            # 计算噪声权重
            epsilon_weight = tf.tensordot(f_eps_in, f_eps_out, axes=0)
            weights = self.weight_mu + self.weight_sigma * epsilon_weight
            
            # 计算噪声偏置
            epsilon_bias = f_eps_out
            biases = self.bias_mu + self.bias_sigma * epsilon_bias
        else:
            # 评估模式：仅使用均值参数
            weights = self.weight_mu
            biases = self.bias_mu
            
        output = tf.matmul(inputs, weights) + biases
        
        if self.activation is not None:
            output = self.activation(output)
            
        return output

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

def create_noisy_dqn_model(state_size, action_size, hidden_size=64):
    """创建带噪声网络的DQN模型"""
    inputs = layers.Input(shape=(state_size,))
    
    # 使用噪声层替代标准Dense层
    x = NoisyDense(hidden_size, activation='relu')(inputs)
    x = NoisyDense(hidden_size, activation='relu')(x)
    outputs = NoisyDense(action_size)(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                 loss='mse')
    return model

def create_standard_dqn_model(state_size, action_size, hidden_size=64):
    """创建标准DQN模型（用于比较）"""
    model = keras.Sequential([
        layers.Dense(hidden_size, activation='relu', input_shape=(state_size,)),
        layers.Dense(hidden_size, activation='relu'),
        layers.Dense(action_size)
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                 loss='mse')
    return model

class NoisyNetDQNAgent:
    """带噪声网络的DQN智能体"""
    def __init__(self, state_size, action_size, hidden_size=64, use_double_dqn=True):
        self.state_size = state_size
        self.action_size = action_size
        self.use_double_dqn = use_double_dqn
        
        # 创建主网络和目标网络
        self.q_network = create_noisy_dqn_model(state_size, action_size, hidden_size)
        self.target_network = create_noisy_dqn_model(state_size, action_size, hidden_size)
        self.update_target_network()  # 初始化目标网络权重
        
        # 经验回放
        self.memory = ReplayBuffer(capacity=10000)
        
        # 学习超参数
        self.gamma = 0.99  # 折扣因子
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
        state = np.reshape(state, [1, self.state_size])
        
        # 在评估模式下关闭噪声
        # 注意：这里需要明确设置training=False来禁用噪声
        if eval_mode:
            # 临时禁用训练模式以不产生噪声
            self.q_network.trainable = False
            q_values = self.q_network(state, training=False).numpy()[0]
            self.q_network.trainable = True
        else:
            # 训练模式下使用噪声实现探索
            q_values = self.q_network(state, training=True).numpy()[0]
            
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
            
            # 定期更新目标网络
            if self.train_step % self.update_target_every == 0:
                self.update_target_network()
                print(f"\n目标网络已更新。训练步数: {self.train_step}")
    
    def learn(self, experiences):
        """从经验中更新值函数"""
        states, actions, rewards, next_states, dones = experiences
        
        if self.use_double_dqn:
            # Double DQN: 使用主网络选择动作，目标网络评估动作
            # 注意：主网络仍使用噪声（进行探索）
            next_q_values_main = self.q_network(next_states, training=True).numpy()
            next_actions = np.argmax(next_q_values_main, axis=1)
            
            # 目标网络评估不使用噪声
            self.target_network.trainable = False
            next_q_values_target = self.target_network(next_states, training=False).numpy()
            self.target_network.trainable = True
            
            max_next_q = np.array([next_q_values_target[i, action] for i, action in enumerate(next_actions)])
        else:
            # 标准DQN: 使用目标网络选择和评估动作
            self.target_network.trainable = False
            next_q_values = self.target_network(next_states, training=False).numpy()
            self.target_network.trainable = True
            max_next_q = np.max(next_q_values, axis=1)
            
        # 计算目标Q值
        targets = rewards + (1 - dones) * self.gamma * max_next_q
        
        # 获取当前的Q值预测
        q_values = self.q_network(states, training=False).numpy()
        
        # 更新目标Q值
        for i, action in enumerate(actions):
            q_values[i][action] = targets[i]
        
        # 训练网络
        self.q_network.fit(states, q_values, epochs=1, verbose=0)
    
    def save(self, filepath):
        """保存模型"""
        self.q_network.save_weights(filepath)
        
    def load(self, filepath):
        """加载模型"""
        self.q_network.load_weights(filepath)
        self.update_target_network()

class StandardDQNAgent:
    """标准DQN智能体（带epsilon-greedy探索，用于比较）"""
    def __init__(self, state_size, action_size, hidden_size=64, use_double_dqn=True):
        self.state_size = state_size
        self.action_size = action_size
        self.use_double_dqn = use_double_dqn
        
        # Q网络 - 主网络和目标网络
        self.q_network = create_standard_dqn_model(state_size, action_size, hidden_size)
        self.target_network = create_standard_dqn_model(state_size, action_size, hidden_size)
        self.update_target_network()  # 初始化目标网络权重
        
        # 经验回放
        self.memory = ReplayBuffer(capacity=10000)
        
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
        
        # 获取当前的Q值预测
        q_values = self.q_network.predict(states, verbose=0)
        
        # 更新目标Q值
        for i, action in enumerate(actions):
            q_values[i][action] = targets[i]
        
        # 训练网络
        self.q_network.fit(states, q_values, epochs=1, verbose=0)
    
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
        message = f'回合 {i_episode}/{n_episodes}, 分数: {score:.2f}, 平均分数: {avg_score:.2f}'
        if hasattr(agent, 'epsilon'):
            message += f', epsilon: {agent.epsilon:.4f}'
            
        print(f'\r{message}', end='')
        
        if i_episode % print_every == 0:
            print(f'\n回合 {i_episode}/{n_episodes}, 分数: {score:.2f}, 平均分数: {avg_score:.2f}')
            # 确保绘图时字体已设置
            if isinstance(agent, NoisyNetDQNAgent):
                plot_scores(scores, avg_scores, title=f"{env_name} - NoisyNet DQN学习曲线")
            else:
                plot_scores(scores, avg_scores, title=f"{env_name} - Epsilon-Greedy DQN学习曲线 (epsilon: {agent.epsilon:.4f})")
            
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

def compare_agents(env_name='CartPole-v1', n_episodes=300):
    """比较标准DQN(ε-greedy)和带噪声网络的DQN性能"""
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # 创建智能体
    standard_agent = StandardDQNAgent(state_size, action_size, use_double_dqn=True)
    noisy_agent = NoisyNetDQNAgent(state_size, action_size, use_double_dqn=True)
    
    # 训练标准DQN (epsilon-greedy)
    print("训练标准DQN (epsilon-greedy探索)...")
    standard_scores, standard_avg_scores = train_agent(
        standard_agent, env_name, n_episodes=n_episodes, print_every=100)
    
    # 训练带噪声网络的DQN
    print("\n训练带噪声网络的DQN...")
    noisy_scores, noisy_avg_scores = train_agent(
        noisy_agent, env_name, n_episodes=n_episodes, print_every=100)
    
    # 绘制比较图
    set_chinese_font() # 绘图前设置字体
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(standard_scores)
    plt.plot(noisy_scores)
    plt.title(f'{env_name} - 回合分数比较')
    plt.xlabel('回合数')
    plt.ylabel('分数')
    plt.legend(['标准DQN (ε-greedy)', '噪声网络DQN'])
    
    plt.subplot(1, 2, 2)
    plt.plot(standard_avg_scores)
    plt.plot(noisy_avg_scores)
    plt.title(f'{env_name} - 平均分数比较 (最近100回合)')
    plt.xlabel('回合数')
    plt.ylabel('平均分数')
    plt.legend(['标准DQN (ε-greedy)', '噪声网络DQN'])
    
    plt.tight_layout()
    plt.savefig(f'./standard_vs_noisy_nets_{env_name}.png')
    plt.show()
    
    return standard_scores, standard_avg_scores, noisy_scores, noisy_avg_scores


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
    
    # 创建带噪声网络的DQN智能体
    agent = NoisyNetDQNAgent(state_size, action_size, hidden_size=64, use_double_dqn=True)
    
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
    plt.title(f'{ENV_NAME} - 噪声网络DQN学习曲线')
    plt.xlabel('回合数')
    plt.ylabel('分数')
    plt.legend(['回合分数', '平均分数 (最近100回合)'])
    plt.savefig(f'./noisy_nets_dqn_{ENV_NAME}_learning_curve.png')
    plt.show()
    
    # # 进行性能比较实验
    # print("\n开始比较标准DQN和带噪声网络的DQN性能...")
    # compare_agents(ENV_NAME, n_episodes=300)
    
    # 评估训练好的智能体
    eval_scores = evaluate_agent(agent, ENV_NAME, render=True, n_episodes=5) 