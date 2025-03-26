"""
Q-Learning算法优化技巧演示
展示不同探索策略、Q-Table初始化和奖励函数设计的效果
"""

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import time
from treasure_maze_env import TreasureMazeEnv

class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, 
                 exploration_strategy="epsilon_greedy", epsilon=0.1, 
                 epsilon_decay=0.995, epsilon_min=0.01,
                 q_init_strategy="zeros", q_init_value=0):
        """
        Q-Learning智能体
        
        参数:
        - env: 环境
        - learning_rate: 学习率
        - discount_factor: 折扣因子
        - exploration_strategy: 探索策略 ("epsilon_greedy", "epsilon_decay", "softmax")
        - epsilon: 探索概率
        - epsilon_decay: epsilon衰减率
        - epsilon_min: epsilon最小值
        - q_init_strategy: Q表初始化策略 ("zeros", "random", "optimistic")
        - q_init_value: 初始化值（用于optimistic初始化）
        """
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_strategy = exploration_strategy
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # 获取状态空间和动作空间大小
        self.state_size = (env.size, env.size)
        self.action_size = env.action_space.n
        
        # 初始化Q表
        self.q_init_strategy = q_init_strategy
        self.q_init_value = q_init_value
        self._initialize_q_table()
        
        # 记录训练过程
        self.rewards_history = []
        self.steps_history = []
        self.epsilon_history = []
    
    def _initialize_q_table(self):
        """根据不同策略初始化Q表"""
        if self.q_init_strategy == "zeros":
            # 零值初始化
            self.q_table = np.zeros(self.state_size + (self.action_size,))
        elif self.q_init_strategy == "random":
            # 随机值初始化
            self.q_table = np.random.uniform(low=0, high=1, size=self.state_size + (self.action_size,))
        elif self.q_init_strategy == "optimistic":
            # 乐观初始化
            self.q_table = np.ones(self.state_size + (self.action_size,)) * self.q_init_value
        else:
            raise ValueError(f"不支持的Q表初始化策略: {self.q_init_strategy}")
    
    def choose_action(self, state):
        """根据当前状态和探索策略选择动作"""
        state_tuple = tuple(state)
        
        if self.exploration_strategy == "epsilon_greedy":
            # ε-greedy策略
            if np.random.random() < self.epsilon:
                return self.env.action_space.sample()  # 探索
            else:
                return np.argmax(self.q_table[state_tuple])  # 利用
        
        elif self.exploration_strategy == "epsilon_decay":
            # ε-greedy退火策略
            if np.random.random() < self.epsilon:
                return self.env.action_space.sample()  # 探索
            else:
                return np.argmax(self.q_table[state_tuple])  # 利用
        
        elif self.exploration_strategy == "softmax":
            # Softmax策略
            q_values = self.q_table[state_tuple]
            # 避免溢出
            q_values = q_values - np.max(q_values)
            exp_values = np.exp(q_values / 0.5)  # 温度参数为0.5
            probabilities = exp_values / np.sum(exp_values)
            return np.random.choice(self.action_size, p=probabilities)
        
        else:
            raise ValueError(f"不支持的探索策略: {self.exploration_strategy}")
    
    def learn(self, state, action, reward, next_state, done):
        """Q-Learning更新规则"""
        state_tuple = tuple(state)
        next_state_tuple = tuple(next_state)
        
        # 当前状态-动作对的Q值
        current_q = self.q_table[state_tuple][action]
        
        # 下一状态的最大Q值
        max_next_q = np.max(self.q_table[next_state_tuple]) if not done else 0
        
        # 计算TD目标
        td_target = reward + self.discount_factor * max_next_q
        
        # 更新Q值
        self.q_table[state_tuple][action] += self.learning_rate * (td_target - current_q)
        
        # 如果使用epsilon衰减策略，更新epsilon
        if self.exploration_strategy == "epsilon_decay":
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def train(self, episodes, max_steps=None, render=False):
        """训练智能体"""
        for episode in range(episodes):
            try:
                # 重置环境
                state, _ = self.env.reset()
                total_reward = 0
                steps = 0
                done = False
                truncated = False
                
                while not (done or truncated):
                    try:
                        # 选择动作
                        action = self.choose_action(state)
                        
                        # 执行动作
                        next_state, reward, done, truncated, _ = self.env.step(action)
                        
                        # 学习
                        self.learn(state, action, reward, next_state, done)
                        
                        # 更新状态
                        state = next_state
                        total_reward += reward
                        steps += 1
                        
                        # 渲染
                        if render:
                            try:
                                self.env.render()
                                time.sleep(0.1)
                            except Exception as e:
                                print(f"渲染警告: {e}")
                                # 渲染错误不应该中断训练
                                render = False
                        
                        # 检查是否达到最大步数
                        if max_steps and steps >= max_steps:
                            break
                    
                    except Exception as e:
                        print(f"步骤执行错误 (episode {episode+1}, step {steps+1}): {e}")
                        # 尝试继续下一个episode
                        break
                
                # 记录训练过程
                self.rewards_history.append(total_reward)
                self.steps_history.append(steps)
                self.epsilon_history.append(self.epsilon)
                
                # 每10个episode打印一次进度
                if (episode + 1) % 10 == 0:
                    avg_reward = np.mean(self.rewards_history[-10:])
                    avg_steps = np.mean(self.steps_history[-10:])
                    print(f"Episode: {episode+1}/{episodes}, Avg Reward: {avg_reward:.2f}, Avg Steps: {avg_steps:.2f}, Epsilon: {self.epsilon:.4f}")
            
            except Exception as e:
                print(f"Episode {episode+1} 执行错误: {e}")
                # 记录部分结果以保持列表长度一致
                if len(self.rewards_history) < episode + 1:
                    self.rewards_history.append(0)
                if len(self.steps_history) < episode + 1:
                    self.steps_history.append(0)
                if len(self.epsilon_history) < episode + 1:
                    self.epsilon_history.append(self.epsilon)
                continue
    
    def plot_training_results(self):
        """绘制训练结果"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
        
        # 绘制奖励曲线
        ax1.plot(self.rewards_history)
        ax1.set_title('奖励曲线')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('总奖励')
        
        # 绘制步数曲线
        ax2.plot(self.steps_history)
        ax2.set_title('步数曲线')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('总步数')
        
        # 绘制epsilon曲线
        ax3.plot(self.epsilon_history)
        ax3.set_title('Epsilon曲线')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Epsilon值')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_policy(self):
        """可视化学习到的策略"""
        policy = np.zeros(self.state_size, dtype=int)
        value = np.zeros(self.state_size)
        
        for i in range(self.state_size[0]):
            for j in range(self.state_size[1]):
                policy[i, j] = np.argmax(self.q_table[i, j])
                value[i, j] = np.max(self.q_table[i, j])
        
        # 创建策略图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # 绘制策略
        cmap = plt.cm.get_cmap('viridis', 4)
        im1 = ax1.imshow(policy, cmap=cmap)
        ax1.set_title('策略 (0=上, 1=右, 2=下, 3=左)')
        
        # 添加颜色条
        cbar1 = plt.colorbar(im1, ax=ax1, ticks=[0, 1, 2, 3])
        cbar1.set_ticklabels(['上', '右', '下', '左'])
        
        # 绘制值函数
        im2 = ax2.imshow(value, cmap='hot')
        ax2.set_title('值函数')
        plt.colorbar(im2, ax=ax2)
        
        plt.tight_layout()
        plt.show()

def compare_exploration_strategies(env_class, episodes=100, trials=3):
    """比较不同探索策略的效果"""
    strategies = ["epsilon_greedy", "epsilon_decay", "softmax"]
    results = {strategy: {"rewards": [], "steps": []} for strategy in strategies}
    
    for strategy in strategies:
        print(f"\n测试探索策略: {strategy}")
        
        for trial in range(trials):
            print(f"试验 {trial+1}/{trials}")
            
            # 创建环境和智能体
            env = env_class(render_mode=None)
            agent = QLearningAgent(
                env=env,
                exploration_strategy=strategy,
                epsilon=0.3 if strategy != "softmax" else 0.1,
                epsilon_decay=0.99
            )
            
            # 训练智能体
            agent.train(episodes=episodes)
            
            # 记录结果
            results[strategy]["rewards"].append(agent.rewards_history)
            results[strategy]["steps"].append(agent.steps_history)
            
            # 关闭环境
            env.close()
    
    # 绘制比较结果
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 绘制奖励曲线
    for strategy in strategies:
        mean_rewards = np.mean(results[strategy]["rewards"], axis=0)
        ax1.plot(mean_rewards, label=strategy)
    
    ax1.set_title('不同探索策略的奖励比较')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('平均奖励')
    ax1.legend()
    
    # 绘制步数曲线
    for strategy in strategies:
        mean_steps = np.mean(results[strategy]["steps"], axis=0)
        ax2.plot(mean_steps, label=strategy)
    
    ax2.set_title('不同探索策略的步数比较')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('平均步数')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def compare_q_initializations(env_class, episodes=100, trials=3):
    """比较不同Q表初始化策略的效果"""
    init_strategies = ["zeros", "random", "optimistic"]
    results = {strategy: {"rewards": [], "steps": []} for strategy in init_strategies}
    
    for strategy in init_strategies:
        print(f"\n测试Q表初始化策略: {strategy}")
        
        for trial in range(trials):
            print(f"试验 {trial+1}/{trials}")
            
            # 创建环境和智能体
            env = env_class(render_mode=None)
            agent = QLearningAgent(
                env=env,
                q_init_strategy=strategy,
                q_init_value=5.0 if strategy == "optimistic" else 0,
                exploration_strategy="epsilon_greedy",
                epsilon=0.1
            )
            
            # 训练智能体
            agent.train(episodes=episodes)
            
            # 记录结果
            results[strategy]["rewards"].append(agent.rewards_history)
            results[strategy]["steps"].append(agent.steps_history)
            
            # 关闭环境
            env.close()
    
    # 绘制比较结果
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 绘制奖励曲线
    for strategy in init_strategies:
        mean_rewards = np.mean(results[strategy]["rewards"], axis=0)
        ax1.plot(mean_rewards, label=strategy)
    
    ax1.set_title('不同Q表初始化策略的奖励比较')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('平均奖励')
    ax1.legend()
    
    # 绘制步数曲线
    for strategy in init_strategies:
        mean_steps = np.mean(results[strategy]["steps"], axis=0)
        ax2.plot(mean_steps, label=strategy)
    
    ax2.set_title('不同Q表初始化策略的步数比较')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('平均步数')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def compare_reward_functions(episodes=100, trials=3):
    """比较不同奖励函数的效果"""
    reward_types = ["sparse", "dense"]
    results = {reward_type: {"rewards": [], "steps": []} for reward_type in reward_types}
    
    for reward_type in reward_types:
        print(f"\n测试奖励函数: {reward_type}")
        
        for trial in range(trials):
            print(f"试验 {trial+1}/{trials}")
            
            # 创建环境和智能体
            env = TreasureMazeEnv(render_mode=None, dense_reward=(reward_type == "dense"))
            agent = QLearningAgent(
                env=env,
                exploration_strategy="epsilon_decay",
                epsilon=0.3,
                epsilon_decay=0.99
            )
            
            # 训练智能体
            agent.train(episodes=episodes)
            
            # 记录结果
            results[reward_type]["rewards"].append(agent.rewards_history)
            results[reward_type]["steps"].append(agent.steps_history)
            
            # 关闭环境
            env.close()
    
    # 绘制比较结果
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 绘制奖励曲线
    for reward_type in reward_types:
        mean_rewards = np.mean(results[reward_type]["rewards"], axis=0)
        ax1.plot(mean_rewards, label=reward_type)
    
    ax1.set_title('不同奖励函数的奖励比较')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('平均奖励')
    ax1.legend()
    
    # 绘制步数曲线
    for reward_type in reward_types:
        mean_steps = np.mean(results[reward_type]["steps"], axis=0)
        ax2.plot(mean_steps, label=reward_type)
    
    ax2.set_title('不同奖励函数的步数比较')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('平均步数')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def demo_single_agent(render=True):
    """演示单个智能体的学习过程"""
    # 创建环境和智能体
    env = TreasureMazeEnv(render_mode="human" if render else None, size=7, dense_reward=True)
    agent = QLearningAgent(
        env=env,
        learning_rate=0.1,
        discount_factor=0.99,
        exploration_strategy="epsilon_decay",
        epsilon=0.3,
        epsilon_decay=0.99,
        epsilon_min=0.01,
        q_init_strategy="optimistic",
        q_init_value=5.0
    )
    
    # 训练智能体
    print("开始训练...")
    agent.train(episodes=100, render=render)
    
    # 绘制训练结果
    agent.plot_training_results()
    
    # 可视化策略
    agent.visualize_policy()
    
    # 关闭环境
    env.close()

if __name__ == "__main__":
    # 演示单个智能体
    demo_single_agent(render=True)
    
    # 比较不同探索策略
    # compare_exploration_strategies(TreasureMazeEnv, episodes=100, trials=3)
    
    # 比较不同Q表初始化策略
    # compare_q_initializations(TreasureMazeEnv, episodes=100, trials=3)
    
    # 比较不同奖励函数
    # compare_reward_functions(episodes=100, trials=3) 