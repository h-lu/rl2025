"""
复杂迷宫环境中Q-Learning优化技巧演示
展示不同优化技巧在复杂环境中的效果差异
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import traceback
import sys
from complex_maze_env import ComplexMazeEnv
from q_learning_demo import QLearningAgent

class ComplexQLearningAgent(QLearningAgent):
    """扩展QLearningAgent以支持复杂环境的观察空间"""
    
    def choose_action(self, state):
        """根据当前状态和探索策略选择动作"""
        # 处理战争迷雾情况下的字典观察
        if isinstance(state, dict):
            # 使用位置作为状态索引
            state_tuple = tuple(state['position'])
        else:
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
        # 处理战争迷雾情况下的字典观察
        if isinstance(state, dict):
            state_tuple = tuple(state['position'])
            next_state_tuple = tuple(next_state['position'])
        else:
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

def compare_exploration_strategies_complex(episodes=100, trials=3):
    """比较不同探索策略在复杂环境中的效果"""
    strategies = ["epsilon_greedy", "epsilon_decay", "softmax"]
    results = {strategy: {"rewards": [], "steps": []} for strategy in strategies}
    
    for strategy in strategies:
        print(f"\n测试探索策略: {strategy}")
        
        for trial in range(trials):
            print(f"试验 {trial+1}/{trials}")
            
            # 创建环境和智能体
            env = ComplexMazeEnv(render_mode=None, size=10, 
                                 moving_traps=True, time_penalty=True, fog_of_war=False)
            agent = ComplexQLearningAgent(
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
    
    ax1.set_title('Complex Maze: Different Exploration Strategies (Rewards)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Reward')
    ax1.legend()
    
    # 绘制步数曲线
    for strategy in strategies:
        mean_steps = np.mean(results[strategy]["steps"], axis=0)
        ax2.plot(mean_steps, label=strategy)
    
    ax2.set_title('Complex Maze: Different Exploration Strategies (Steps)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Steps')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def compare_q_initializations_complex(episodes=100, trials=3):
    """比较不同Q表初始化策略在复杂环境中的效果"""
    init_strategies = ["zeros", "random", "optimistic"]
    results = {strategy: {"rewards": [], "steps": []} for strategy in init_strategies}
    
    for strategy in init_strategies:
        print(f"\n测试Q表初始化策略: {strategy}")
        
        for trial in range(trials):
            print(f"试验 {trial+1}/{trials}")
            
            # 创建环境和智能体
            env = ComplexMazeEnv(render_mode=None, size=10, 
                                 moving_traps=True, time_penalty=True, fog_of_war=False)
            agent = ComplexQLearningAgent(
                env=env,
                q_init_strategy=strategy,
                q_init_value=10.0 if strategy == "optimistic" else 0,
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
    
    ax1.set_title('Complex Maze: Different Q-Table Initializations (Rewards)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Reward')
    ax1.legend()
    
    # 绘制步数曲线
    for strategy in init_strategies:
        mean_steps = np.mean(results[strategy]["steps"], axis=0)
        ax2.plot(mean_steps, label=strategy)
    
    ax2.set_title('Complex Maze: Different Q-Table Initializations (Steps)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Steps')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def compare_reward_functions_complex(episodes=100, trials=3):
    """比较不同奖励函数在复杂环境中的效果"""
    reward_types = ["sparse", "dense"]
    results = {reward_type: {"rewards": [], "steps": []} for reward_type in reward_types}
    
    for reward_type in reward_types:
        print(f"\n测试奖励函数: {reward_type}")
        
        for trial in range(trials):
            print(f"试验 {trial+1}/{trials}")
            
            # 创建环境和智能体
            env = ComplexMazeEnv(render_mode=None, size=10, 
                                 dense_reward=(reward_type == "dense"),
                                 moving_traps=True, time_penalty=True, fog_of_war=False)
            agent = ComplexQLearningAgent(
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
    
    ax1.set_title('Complex Maze: Different Reward Functions (Rewards)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Reward')
    ax1.legend()
    
    # 绘制步数曲线
    for reward_type in reward_types:
        mean_steps = np.mean(results[reward_type]["steps"], axis=0)
        ax2.plot(mean_steps, label=reward_type)
    
    ax2.set_title('Complex Maze: Different Reward Functions (Steps)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Steps')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def demo_complex_agent(render=True):
    """演示单个智能体在复杂环境中的学习过程"""
    # 创建环境和智能体
    env = ComplexMazeEnv(render_mode="human" if render else None, size=10, 
                         dense_reward=True, moving_traps=True, 
                         time_penalty=True, fog_of_war=False, animation_speed=5.0, render_every=10)
    agent = ComplexQLearningAgent(
        env=env,
        learning_rate=0.1,
        discount_factor=0.99,
        exploration_strategy="epsilon_decay",
        epsilon=0.3,
        epsilon_decay=0.99,
        epsilon_min=0.01,
        q_init_strategy="optimistic",
        q_init_value=10.0
    )
    
    # 训练智能体
    print("开始训练...")
    agent.train(episodes=100, render=render)
    
    # 绘制训练结果
    agent.plot_training_results()
    
    # 关闭环境
    env.close()

def main():
    print("复杂迷宫环境中Q-Learning优化技巧演示")
    print("=" * 50)
    print("请选择要演示的内容：")
    print("1. 单个智能体学习过程")
    print("2. 比较不同探索策略")
    print("3. 比较不同Q表初始化策略")
    print("4. 比较不同奖励函数")
    print("0. 退出")
    
    choice = input("请输入选项（0-4）：")
    
    try:
        if choice == "1":
            print("\n演示单个智能体学习过程...")
            try:
                demo_complex_agent(render=True)
            except Exception as e:
                print(f"训练过程中发生错误: {e}")
                traceback.print_exc()
        
        elif choice == "2":
            print("\n比较不同探索策略...")
            try:
                compare_exploration_strategies_complex(episodes=50, trials=2)
            except Exception as e:
                print(f"比较探索策略时发生错误: {e}")
                traceback.print_exc()
        
        elif choice == "3":
            print("\n比较不同Q表初始化策略...")
            try:
                compare_q_initializations_complex(episodes=50, trials=2)
            except Exception as e:
                print(f"比较Q表初始化策略时发生错误: {e}")
                traceback.print_exc()
        
        elif choice == "4":
            print("\n比较不同奖励函数...")
            try:
                compare_reward_functions_complex(episodes=50, trials=2)
            except Exception as e:
                print(f"比较奖励函数时发生错误: {e}")
                traceback.print_exc()
        
        elif choice == "0":
            print("退出程序")
            return
        
        else:
            print("无效选项，请重新运行程序")
    
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序运行时发生错误: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"程序运行时发生严重错误: {e}")
        traceback.print_exc()
        sys.exit(1) 