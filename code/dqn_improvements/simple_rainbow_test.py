# 简化版Rainbow DQN测试脚本
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Layer, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import gymnasium as gym
import random
from collections import deque
import matplotlib.pyplot as plt
import os
import pickle
import platform # 导入 platform 模块以检测操作系统

# 配置 TensorFlow 使用 GPU 并允许内存增长
print("Configuring TensorFlow for GPU...")
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    try:
        # 设置内存增长，避免一次性占用所有显存
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"找到 {len(physical_devices)} 个 GPU 设备并配置完成。")
    except RuntimeError as e:
        print(f"GPU 配置错误: {e}")
else:
    print("未检测到 GPU，将使用 CPU 运行。")

# 设置随机种子以获得可重复的结果
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

def set_chinese_font():
    """根据操作系统设置中文字体"""
    system = platform.system()
    try:
        if system == 'Windows':
            plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 黑体
            plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
            print("字体设置为 SimHei (Windows)。")
        elif system == 'Darwin': # macOS
            plt.rcParams['font.sans-serif'] = ['PingFang SC'] # Mac 苹方
            plt.rcParams['axes.unicode_minus'] = False
            print("字体设置为 PingFang SC (macOS)。")
        else: # Linux 或其他
            # 尝试查找常见 Linux 中文字体
            plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'Arial Unicode MS']
            plt.rcParams['axes.unicode_minus'] = False
            print("尝试设置字体为 WenQuanYi Zen Hei/SimHei/Arial Unicode MS (Linux/其他系统)。")
    except Exception as e:
        print(f"设置中文字体时出错: {e}。Matplotlib 可能会回退到默认字体。")

# 简化的Noisy Networks实现
class NoisyDense(Layer):
    def __init__(self, units, activation=None, **kwargs):
        super(NoisyDense, self).__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)
        
    def build(self, input_shape):
        self.input_dim = input_shape[-1]
        
        # 初始化均值权重参数
        self.weight_mu = self.add_weight(
            shape=(self.input_dim, self.units),
            initializer=keras.initializers.RandomUniform(-0.01, 0.01),
            name='weight_mu')
        self.bias_mu = self.add_weight(
            shape=(self.units,),
            initializer=keras.initializers.Zeros(),
            name='bias_mu')
        
        # 初始化噪声权重参数
        self.weight_sigma = self.add_weight(
            shape=(self.input_dim, self.units),
            initializer=keras.initializers.Constant(0.017),
            name='weight_sigma')
        self.bias_sigma = self.add_weight(
            shape=(self.units,),
            initializer=keras.initializers.Constant(0.017),
            name='bias_sigma')
        
        super(NoisyDense, self).build(input_shape)
    
    def call(self, inputs):
        # 简化版本：总是使用噪声
        # 生成噪声
        eps_i = tf.random.normal((self.input_dim, 1))
        eps_j = tf.random.normal((1, self.units))
        # 计算f(eps_i)和f(eps_j)，这里f(x) = sign(x) * sqrt(|x|)
        f_eps_i = tf.sign(eps_i) * tf.sqrt(tf.abs(eps_i))
        f_eps_j = tf.sign(eps_j) * tf.sqrt(tf.abs(eps_j))
        # 生成epsilon权重和偏置的噪声
        eps_w = f_eps_i * f_eps_j
        eps_b = tf.squeeze(f_eps_j)
        
        # 带噪声的前向传播
        weights = self.weight_mu + self.weight_sigma * eps_w
        bias = self.bias_mu + self.bias_sigma * eps_b
        output = tf.matmul(inputs, weights) + bias
        
        if self.activation is not None:
            output = self.activation(output)
            
        return output
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

# 定义标准DQN的网络结构
def create_q_network(state_size, action_size):
    inputs = Input(shape=(state_size,))
    x = Dense(64, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(action_size, activation='linear')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

# 优先经验回放缓冲区
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=0.01):
        self.capacity = capacity
        self.alpha = alpha  # 优先级的指数
        self.beta = beta    # 重要性采样的指数
        self.beta_increment = beta_increment  # beta的增量，随着训练逐渐接近1
        self.epsilon = epsilon  # 防止优先级为0
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
    
    def add(self, state, action, reward, next_state, done):
        max_priority = np.max(self.priorities) if self.size > 0 else 1.0
        
        if self.size < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.size += 1
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        if self.size < batch_size:
            indices = range(self.size)
        else:
            # 根据优先级计算采样概率
            priorities = self.priorities[:self.size]
            probabilities = (priorities + self.epsilon) ** self.alpha
            probabilities /= np.sum(probabilities)
            
            # 随机采样
            indices = np.random.choice(self.size, batch_size, p=probabilities)
        
        # 获取样本和权重
        samples = [self.buffer[idx] for idx in indices]
        
        # 计算重要性采样权重
        weights = []
        total = self.size
        probabilities = (self.priorities[indices] + self.epsilon) ** self.alpha
        probabilities /= np.sum(self.priorities[:self.size] ** self.alpha)
        
        max_weight = (total * np.min(probabilities)) ** (-self.beta)
        
        for prob in probabilities:
            weight = (total * prob) ** (-self.beta)
            weights.append(weight / max_weight)
        
        # 增加beta以接近1
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        states = np.array([sample[0] for sample in samples])
        actions = np.array([sample[1] for sample in samples])
        rewards = np.array([sample[2] for sample in samples])
        next_states = np.array([sample[3] for sample in samples])
        dones = np.array([sample[4] for sample in samples]).astype(np.float32)
        
        return states, actions, rewards, next_states, dones, indices, np.array(weights)
    
    def update_priorities(self, indices, new_priorities):
        for idx, priority in zip(indices, new_priorities):
            self.priorities[idx] = priority + self.epsilon

# 双重网络(Dueling Network)结构
def create_dueling_q_network(state_size, action_size):
    inputs = Input(shape=(state_size,))
    x = Dense(64, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    
    # 价值流
    value_stream = Dense(32, activation='relu')(x)
    value = Dense(1, activation='linear')(value_stream)
    
    # 优势流
    advantage_stream = Dense(32, activation='relu')(x)
    advantage = Dense(action_size, activation='linear')(advantage_stream)
    
    # 合并价值和优势以获得Q值
    outputs = value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# 带噪声的双重网络结构
def create_noisy_dueling_q_network(state_size, action_size):
    inputs = Input(shape=(state_size,))
    x = NoisyDense(64, activation='relu')(inputs)
    x = NoisyDense(64, activation='relu')(x)
    
    # 价值流
    value_stream = NoisyDense(32, activation='relu')(x)
    value = NoisyDense(1, activation='linear')(value_stream)
    
    # 优势流
    advantage_stream = NoisyDense(32, activation='relu')(x)
    advantage = NoisyDense(action_size, activation='linear')(advantage_stream)
    
    # 使用 Lambda 层处理 advantage_mean 和合并操作，避免直接使用 tf 操作
    # 这样可以确保所有操作都在 Keras 层内部完成
    def combine_value_and_advantage(inputs):
        value, advantage = inputs
        advantage_mean = keras.backend.mean(advantage, axis=1, keepdims=True)
        return value + (advantage - advantage_mean)
    
    # 显式指定 output_shape 参数
    def compute_output_shape(input_shapes):
        return input_shapes[1]  # 返回 advantage 的形状，即 (batch_size, action_size)
    
    outputs = Lambda(combine_value_and_advantage, 
                    output_shape=compute_output_shape)([value, advantage])
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# 标准DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99    # 折扣因子
        self.epsilon = 1.0   # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.update_target_freq = 10  # 目标网络更新频率
        self.batch_size = 64
        
        # 创建主网络和目标网络
        self.q_network = create_q_network(state_size, action_size)
        self.target_q_network = create_q_network(state_size, action_size)
        self.q_network.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        
        # 同步两个网络的权重
        self.update_target_network()
    
    def update_target_network(self):
        self.target_q_network.set_weights(self.q_network.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, evaluate=False):
        # ε-贪婪策略
        if not evaluate and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = np.reshape(state, [1, self.state_size])
        q_values = self.q_network.predict(state, verbose=0)[0]
        return np.argmax(q_values)
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0
        
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([tup[0] for tup in minibatch])
        actions = np.array([tup[1] for tup in minibatch])
        rewards = np.array([tup[2] for tup in minibatch])
        next_states = np.array([tup[3] for tup in minibatch])
        dones = np.array([tup[4] for tup in minibatch]).astype(np.float32)
        
        # 计算目标Q值
        target = rewards + (1 - dones) * self.gamma * np.amax(self.target_q_network.predict(next_states, verbose=0), axis=1)
        
        # 获取当前Q值估计，并更新选定动作的目标
        target_f = self.q_network.predict(states, verbose=0)
        for i, action in enumerate(actions):
            target_f[i][action] = target[i]
        
        # 训练网络
        history = self.q_network.fit(states, target_f, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        
        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss

# Rainbow DQN Agent
class RainbowDQNAgent:
    """Rainbow DQN 智能体"""
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Rainbow配置
        self.memory = PrioritizedReplayBuffer(capacity=10000)
        self.gamma = 0.99    # 折扣因子
        self.learning_rate = 0.001
        self.update_target_freq = 10  # 目标网络更新频率
        self.batch_size = 64
        self.use_noisy_nets = True  # Rainbow使用噪声网络
        
        # 创建带噪声的双重网络
        self.q_network = create_noisy_dueling_q_network(state_size, action_size)
        self.target_q_network = create_noisy_dueling_q_network(state_size, action_size)
        self.q_network.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        
        # 同步两个网络的权重
        self.update_target_network()
    
    def update_target_network(self):
        self.target_q_network.set_weights(self.q_network.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        # 添加经验到优先回放缓冲区
        self.memory.add(state, action, reward, next_state, done)
    
    def act(self, state, evaluate=False):
        # Rainbow使用噪声网络，不需要ε-贪婪
        # 在评估模式下，我们仍然使用噪声，但在实际应用中可能需要修改噪声参数或完全禁用
        state = np.reshape(state, [1, self.state_size])
        q_values = self.q_network.predict(state, verbose=0)[0]
        return np.argmax(q_values)
    
    def replay(self):
        """从优先回放缓冲区抽样并训练网络"""
        # 如果缓冲区样本不足，则跳过训练
        if len(self.memory.buffer) < self.batch_size:
            return 0
        
        try:
            # 从优先回放缓冲区采样
            states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size)
            
            # Double DQN: 先从在线网络获取最佳动作
            next_actions = np.argmax(self.q_network.predict(next_states, verbose=0), axis=1)
            
            # 然后从目标网络获取这些动作的Q值
            next_q_values = self.target_q_network.predict(next_states, verbose=0)
            max_next_q = np.array([next_q_values[i, action] for i, action in enumerate(next_actions)])
            
            # 计算TD目标
            targets = rewards + (1 - dones) * self.gamma * max_next_q
            
            # 获取当前预测
            target_f = self.q_network.predict(states, verbose=0)
            
            # 计算TD误差并更新优先级
            td_errors = np.abs(targets - np.array([target_f[i, action] for i, action in enumerate(actions)]))
            self.memory.update_priorities(indices, td_errors)
            
            # 更新目标值
            for i, action in enumerate(actions):
                target_f[i][action] = targets[i]
            
            # 训练网络
            self.q_network.fit(states, target_f, sample_weight=weights, epochs=1, verbose=0)
            
            return np.mean(td_errors)
        except Exception as e:
            print(f"在 replay 方法中出错: {e}")
            return 0

# 训练和测试函数
def train_dqn(env, agent, episodes=1000, max_steps=500, show_progress=False):
    """训练 DQN 智能体并返回训练过程中的分数"""
    scores = []
    
    for e in range(1, episodes + 1):
        state, _ = env.reset(seed=SEED+e)  # 使用不同的种子
        state = np.reshape(state, [1, agent.state_size])[0]
        score = 0
        
        for step in range(max_steps):
            # 选择动作
            action = agent.act(state)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = np.reshape(next_state, [1, agent.state_size])[0]
            
            # 存储经验
            agent.remember(state, action, reward, next_state, done)
            
            # 更新状态
            state = next_state
            score += reward
            
            # 进行经验回放
            agent.replay()
            
            # 如果游戏结束，跳出循环
            if done:
                break
        
        # 更新目标网络（如果需要）
        if e % agent.update_target_freq == 0:
            agent.update_target_network()
        
        # 衰减探索率（如果有）
        if hasattr(agent, 'epsilon'):
            agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        
        # 记录分数
        scores.append(score)
        
        # 显示进度
        if show_progress and e % 2 == 0:  # 每2回合显示一次
            epsilon = agent.epsilon if hasattr(agent, 'epsilon') else None
            if epsilon is not None:
                print(f"回合 {e}/{episodes}, 分数: {score:.2f}, 平均分数: {np.mean(scores):.2f}, ε: {epsilon:.4f}")
            else:
                print(f"回合 {e}/{episodes}, 分数: {score:.2f}, 平均分数: {np.mean(scores):.2f}")
    
    return scores

def evaluate_agent(env, agent, episodes=10, max_steps=500):
    scores = []
    
    for e in range(episodes):
        state, _ = env.reset()
        state = np.reshape(state, [1, agent.state_size])[0]
        score = 0
        
        for step in range(max_steps):
            action = agent.act(state, evaluate=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = np.reshape(next_state, [1, agent.state_size])[0]
            
            state = next_state
            score += reward
            
            if done:
                break
        
        scores.append(score)
    
    return np.mean(scores)

def plot_comparison(dqn_scores, rainbow_scores, smoothing_window=3):
    # 设置中文字体
    set_chinese_font()
    
    # 平滑处理分数
    def smooth_scores(scores, window=3):
        if len(scores) < window:
            return scores
        smoothed = []
        for i in range(len(scores)):
            start = max(0, i - window + 1)
            smoothed.append(np.mean(scores[start:i+1]))
        return smoothed
    
    dqn_smooth = smooth_scores(dqn_scores, smoothing_window)
    rainbow_smooth = smooth_scores(rainbow_scores, smoothing_window)
    
    plt.figure(figsize=(12, 6))
    plt.plot(dqn_scores, alpha=0.3, color='blue')
    plt.plot(dqn_smooth, linewidth=2, color='blue', label='标准 DQN')
    plt.plot(rainbow_scores, alpha=0.3, color='red')
    plt.plot(rainbow_smooth, linewidth=2, color='red', label='Rainbow DQN')
    plt.xlabel('回合')
    plt.ylabel('得分')
    plt.title('标准 DQN vs Rainbow DQN 性能比较')
    plt.legend()
    plt.grid(True)
    plt.savefig('dqn_vs_rainbow.png')
    plt.show()

def save_results(dqn_scores, rainbow_scores, dqn_eval_score, rainbow_eval_score):
    # 创建结果目录（如果不存在）
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # 保存训练分数
    with open('results/dqn_vs_rainbow_scores.pkl', 'wb') as f:
        pickle.dump({
            'dqn_scores': dqn_scores,
            'rainbow_scores': rainbow_scores,
            'dqn_eval_score': dqn_eval_score,
            'rainbow_eval_score': rainbow_eval_score
        }, f)
    
    # 保存结果摘要
    with open('results/dqn_vs_rainbow_summary.txt', 'w') as f:
        f.write(f"标准DQN 平均训练分数: {np.mean(dqn_scores):.2f}\n")
        f.write(f"Rainbow DQN 平均训练分数: {np.mean(rainbow_scores):.2f}\n")
        f.write(f"标准DQN 评估分数: {dqn_eval_score:.2f}\n")
        f.write(f"Rainbow DQN 评估分数: {rainbow_eval_score:.2f}\n")
        f.write(f"Rainbow vs 标准DQN 提升: {(rainbow_eval_score/dqn_eval_score - 1)*100:.2f}%\n")

def main():
    # 创建环境
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    print(f"环境: CartPole-v1, 状态空间: {state_size}, 动作空间: {action_size}")
    print(f"开始训练标准DQN和Rainbow DQN进行比较，每个算法训练10回合...")
    
    # 训练标准DQN
    print("\n训练标准DQN...")
    dqn_agent = DQNAgent(state_size, action_size)
    dqn_scores = train_dqn(env, dqn_agent, episodes=10, show_progress=True)
    dqn_eval_score = evaluate_agent(env, dqn_agent)
    
    # 训练Rainbow DQN
    print("\n训练Rainbow DQN...")
    rainbow_agent = RainbowDQNAgent(state_size, action_size)
    rainbow_scores = train_dqn(env, rainbow_agent, episodes=10, show_progress=True)
    rainbow_eval_score = evaluate_agent(env, rainbow_agent)
    
    # 比较结果
    print("\n===== 结果比较 =====")
    print(f"标准DQN最终评估分数: {dqn_eval_score:.2f}")
    print(f"Rainbow DQN最终评估分数: {rainbow_eval_score:.2f}")
    
    # 绘制比较图
    plot_comparison(dqn_scores, rainbow_scores)
    
    # 保存结果
    save_results(dqn_scores, rainbow_scores, dqn_eval_score, rainbow_eval_score)
    
    print("\n完成比较！")

if __name__ == "__main__":
    main() 