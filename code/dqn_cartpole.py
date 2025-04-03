import gymnasium as gym
import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from IPython.display import clear_output

# 设置随机种子以获得可重复的结果
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 检查是否可用GPU
print(f"TensorFlow版本: {tf.__version__}")
print(f"使用GPU: {len(tf.config.list_physical_devices('GPU')) > 0}")

# 定义DQN网络 (使用Keras)
def create_dqn_model(state_size, action_size, hidden_size=64):
    model = keras.Sequential([
        layers.Dense(hidden_size, activation='relu', input_shape=(state_size,)),
        layers.Dense(hidden_size, activation='relu'),
        layers.Dense(action_size, activation='linear')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                 loss='mse')
    return model

# 定义经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)
        
        # 转换为NumPy数组而不是PyTorch张量
        return (np.array(states), 
                np.array(actions), 
                np.array(rewards), 
                np.array(next_states), 
                np.array(dones, dtype=np.float32))
    
    def __len__(self):
        return len(self.buffer)

# 定义DQN代理
class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, epsilon_start=1.0, 
                 epsilon_end=0.01, epsilon_decay=0.995, buffer_size=10000, batch_size=64, 
                 update_target_every=10):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon_start  # 探索率
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.learning_rate = learning_rate
        
        # Q网络和目标网络 (使用Keras)
        self.q_network = create_dqn_model(state_size, action_size)
        self.target_network = create_dqn_model(state_size, action_size)
        self.target_network.set_weights(self.q_network.get_weights())
        
        # 经验回放缓冲区
        self.memory = ReplayBuffer(buffer_size)
        
        # 记录学习步数
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # 存储经验到缓冲区
        self.memory.add(state, action, reward, next_state, done)
        
        # 递增时间步
        self.t_step += 1
        
        # 当缓冲区中有足够的样本时，从中学习
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)
            
            # 定期更新目标网络
            if self.t_step % self.update_target_every == 0:
                self.target_network.set_weights(self.q_network.get_weights())
    
    def act(self, state, eval_mode=False):
        # 将状态转换为批次格式
        state = np.reshape(state, [1, self.state_size])
        
        # 评估模式或使用epsilon-greedy策略选择动作
        if eval_mode or random.random() > self.epsilon:
            action_values = self.q_network.predict(state, verbose=0)
            return np.argmax(action_values[0])
        else:
            return random.choice(np.arange(self.action_size))
    
    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        
        # 从目标网络中获取下一个状态的最大Q值
        target_q_values = self.target_network.predict(next_states, verbose=0)
        max_target_q = np.max(target_q_values, axis=1)
        
        # 计算目标Q值
        targets = rewards + (self.gamma * max_target_q * (1 - dones))
        
        # 获取当前预测的Q值并更新目标
        target_f = self.q_network.predict(states, verbose=0)
        for i, action in enumerate(actions):
            target_f[i][action] = targets[i]
        
        # 训练Q网络
        self.q_network.fit(states, target_f, epochs=1, verbose=0)
        
        # 更新epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, filename):
        self.q_network.save_weights(filename)
    
    def load(self, filename):
        self.q_network.load_weights(filename)
        self.target_network.set_weights(self.q_network.get_weights())

# 绘制学习过程
def plot_scores(scores, avg_scores):
    clear_output(True)
    plt.figure(figsize=(10, 5))
    plt.plot(scores)
    plt.plot(avg_scores)
    plt.title('DQN学习曲线')
    plt.xlabel('回合数')
    plt.ylabel('分数')
    plt.legend(['回合分数', '平均分数 (最近100回合)'])
    plt.show()

# 训练DQN代理
def train_dqn(env_name='CartPole-v1', n_episodes=1000, max_t=1000, eps_start=1.0, 
              eps_end=0.01, eps_decay=0.995, target_score=195.0, render=False):
    env = gym.make(env_name)
    print(f"状态空间: {env.observation_space.shape[0]}")
    print(f"动作空间: {env.action_space.n}")
    
    agent = DQNAgent(state_size=env.observation_space.shape[0], 
                     action_size=env.action_space.n, 
                     epsilon_start=eps_start, 
                     epsilon_end=eps_end, 
                     epsilon_decay=eps_decay)
    
    scores = []
    avg_scores = []
    
    for i_episode in range(1, n_episodes+1):
        state, _ = env.reset(seed=SEED)
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
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)
        
        print(f'\r回合 {i_episode}/{n_episodes}, 分数: {score:.2f}, 平均分数: {avg_score:.2f}, epsilon: {agent.epsilon:.4f}', end='')
        
        if i_episode % 50 == 0:
            print(f'\r回合 {i_episode}/{n_episodes}, 分数: {score:.2f}, 平均分数: {avg_score:.2f}, epsilon: {agent.epsilon:.4f}')
            plot_scores(scores, avg_scores)
            
        if avg_score >= target_score and i_episode >= 100:
            print(f'\n环境在{i_episode}回合后解决!')
            agent.save(f'dqn_checkpoint_{env_name}_{i_episode}.h5')
            break
    
    return agent, scores, avg_scores

# 评估训练好的代理
def evaluate_agent(agent, env_name='CartPole-v1', n_episodes=10, render=False):
    env = gym.make(env_name, render_mode="human" if render else None)
    scores = []
    
    for i_episode in range(1, n_episodes+1):
        state, _ = env.reset(seed=SEED)
        score = 0
        
        while True:
            if render:
                env.render()
            
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

if __name__ == "__main__":
    # 训练DQN代理
    agent, scores, avg_scores = train_dqn(env_name='CartPole-v1', 
                                         n_episodes=500, 
                                         eps_decay=0.995,
                                         target_score=195.0)
    
    # 绘制最终学习曲线
    plt.figure(figsize=(10, 5))
    plt.plot(scores)
    plt.plot(avg_scores)
    plt.title('DQN学习曲线')
    plt.xlabel('回合数')
    plt.ylabel('分数')
    plt.legend(['回合分数', '平均分数 (最近100回合)'])
    plt.savefig('dqn_learning_curve.png')
    plt.show()
    
    # 评估训练好的代理
    eval_scores = evaluate_agent(agent, render=True, n_episodes=5) 