import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque
import time

from utils.exploration import EpsilonGreedy, BoltzmannExploration, NoisyLinear
from environment.blackjack_env import BlackjackWrapper
from utils.visualization import render_blackjack_state

# 设置页面
st.set_page_config(page_title="探索策略 - DQN改进", layout="wide")

# 标题和介绍
st.title("探索的艺术：不同探索策略的比较")
st.markdown("""
探索 (Exploration) 与利用 (Exploitation) 的平衡是强化学习中的经典问题。本页面展示了三种不同的探索策略：
- **ε-greedy**: 以概率 ε 随机选择动作，以概率 1-ε 选择Q值最大的动作
- **Boltzmann探索**: 根据Q值的相对大小确定动作选择概率
- **噪声网络**: 直接在网络权重中引入可学习的噪声
""")

# 基本Q网络
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)

# 噪声网络
class NoisyQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(NoisyQNetwork, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        self.noisy1 = NoisyLinear(hidden_dim, hidden_dim)
        self.noisy2 = NoisyLinear(hidden_dim, action_dim)
    
    def forward(self, x):
        feature = self.feature(x)
        hidden = F.relu(self.noisy1(feature))
        output = self.noisy2(hidden)
        return output
    
    def reset_noise(self):
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()

# 探索策略可视化
st.header("探索策略可视化")

# 创建示例Q值
q_values = np.array([0.2, 0.5, 0.1, 0.8, 0.3])  # 示例Q值
actions = np.arange(len(q_values))

# 选择探索策略
exploration_type = st.selectbox(
    "选择探索策略",
    ["ε-greedy", "Boltzmann探索", "噪声网络"]
)

# 参数设置
col1, col2 = st.columns(2)

with col1:
    if exploration_type == "ε-greedy":
        epsilon = st.slider("探索率 (ε)", 0.0, 1.0, 0.3, 0.01)
        
        # 创建ε-greedy策略
        epsilon_greedy = EpsilonGreedy(epsilon_start=epsilon)
        
        # 可视化动作选择概率
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # 计算选择每个动作的概率
        probs = np.ones(len(q_values)) * epsilon / len(q_values)
        probs[np.argmax(q_values)] += (1 - epsilon)
        
        ax.bar(actions, probs, color='skyblue')
        ax.set_xlabel('动作')
        ax.set_ylabel('选择概率')
        ax.set_title(f'ε-greedy策略下的动作选择概率 (ε={epsilon})')
        ax.set_xticks(actions)
        
        st.pyplot(fig)
        
        st.markdown(f"""
        ### ε-greedy策略:
        - 以概率 {epsilon} 随机选择一个动作
        - 以概率 {1-epsilon} 选择Q值最大的动作 (当前是动作 {np.argmax(q_values)})
        - 优点: 简单、易实现，有理论保证
        - 缺点: 完全随机的探索，不考虑动作间的价值差异
        """)
        
    elif exploration_type == "Boltzmann探索":
        temperature = st.slider("温度 (τ)", 0.1, 5.0, 1.0, 0.1)
        
        # 创建Boltzmann探索策略
        boltzmann = BoltzmannExploration(temperature_start=temperature)
        
        # 可视化动作选择概率
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # 计算Boltzmann分布
        q_values_tensor = torch.FloatTensor(q_values)
        probs = F.softmax(q_values_tensor / temperature, dim=0).numpy()
        
        ax.bar(actions, probs, color='salmon')
        ax.set_xlabel('动作')
        ax.set_ylabel('选择概率')
        ax.set_title(f'Boltzmann策略下的动作选择概率 (τ={temperature})')
        ax.set_xticks(actions)
        
        st.pyplot(fig)
        
        st.markdown(f"""
        ### Boltzmann探索:
        - 基于Q值的相对大小确定动作选择概率
        - 温度参数 τ 控制探索程度: τ越大，动作选择越随机；τ越小，更倾向于选择高Q值动作
        - 当前各动作被选择的概率: {', '.join([f'动作{i}: {p:.2f}' for i, p in enumerate(probs)])}
        - 优点: 考虑了动作间的价值差异，探索更有方向性
        - 缺点: 需要额外的温度参数调节，且Q值差异很大时可能退化为贪婪策略
        """)
        
    elif exploration_type == "噪声网络":
        std_init = st.slider("噪声初始标准差", 0.01, 1.0, 0.5, 0.01)
        
        # 创建噪声示例
        state_dim = 3
        action_dim = 2
        noisy_net = NoisyQNetwork(state_dim, action_dim, hidden_dim=64)
        
        # 多次前向传播，观察输出变化
        state = torch.FloatTensor(np.array([15, 5, 0]))  # 示例状态
        
        outputs = []
        for _ in range(10):
            noisy_net.reset_noise()
            outputs.append(noisy_net(state).detach().numpy())
        
        outputs = np.array(outputs)
        
        # 可视化多次前向传播的结果
        fig, ax = plt.subplots(figsize=(10, 5))
        
        ax.boxplot([outputs[:, 0], outputs[:, 1]])
        ax.scatter(np.ones(len(outputs)) * 1, outputs[:, 0], color='blue', alpha=0.6)
        ax.scatter(np.ones(len(outputs)) * 2, outputs[:, 1], color='red', alpha=0.6)
        
        ax.set_xlabel('动作')
        ax.set_ylabel('Q值')
        ax.set_title('噪声网络多次前向传播的Q值分布')
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['停牌', '要牌'])
        
        st.pyplot(fig)
        
        st.markdown(f"""
        ### 噪声网络:
        - 直接在网络权重中引入可学习的参数化噪声
        - 每次前向传播，噪声都会导致Q值估计略有不同
        - 噪声随着学习过程自适应调整，无需手动设计探索策略
        - 优点: 探索是状态相关的，且程度是自适应的；测试时可以关闭噪声
        - 缺点: 实现较复杂，增加了网络参数数量
        """)

# 策略对比演示
st.header("探索策略在21点环境中的表现")

# 创建环境
env = BlackjackWrapper(gym.make('Blackjack-v1'))

# 简单的神经网络智能体
class SimpleAgent:
    def __init__(self, state_dim, action_dim, exploration_type='epsilon-greedy'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.exploration_type = exploration_type
        
        # 创建网络
        if exploration_type == 'noisy':
            self.q_network = NoisyQNetwork(state_dim, action_dim)
        else:
            self.q_network = QNetwork(state_dim, action_dim)
        
        # 创建探索策略
        if exploration_type == 'epsilon-greedy':
            self.exploration = EpsilonGreedy(epsilon_start=0.5, epsilon_decay=0.99)
        elif exploration_type == 'boltzmann':
            self.exploration = BoltzmannExploration(temperature_start=1.0, temperature_decay=0.99)
    
    def select_action(self, state):
        state_tensor = torch.FloatTensor(state)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        
        if self.exploration_type == 'epsilon-greedy':
            action = self.exploration.select_action(q_values, self.action_dim)
        elif self.exploration_type == 'boltzmann':
            action = self.exploration.select_action(q_values, self.action_dim)
        elif self.exploration_type == 'noisy':
            # 重置噪声
            self.q_network.reset_noise()
            # 贪婪选择
            action = q_values.argmax().item()
        
        return action

# 运行一局游戏
if st.button("使用不同探索策略玩一局21点"):
    # 创建智能体
    agents = {
        "ε-greedy": SimpleAgent(3, env.action_space.n, 'epsilon-greedy'),
        "Boltzmann探索": SimpleAgent(3, env.action_space.n, 'boltzmann'),
        "噪声网络": SimpleAgent(3, env.action_space.n, 'noisy')
    }
    
    # 创建列
    cols = st.columns(3)
    
    for i, (name, agent) in enumerate(agents.items()):
        with cols[i]:
            st.subheader(name)
            
            # 重置环境
            state, _ = env.reset()
            render_blackjack_state(state)
            
            # 游戏循环
            done = False
            while not done:
                # 选择动作
                action = agent.select_action(state)
                
                # 执行动作
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # 显示动作和下一个状态
                st.write(f"动作: {'要牌' if action == 1 else '停牌'}")
                st.write(f"新状态: 玩家点数={next_state[0]}, 庄家明牌={next_state[1]}, 可用A={'是' if next_state[2] else '否'}")
                
                # 游戏结束
                if done:
                    if reward > 0:
                        st.success(f"游戏结束，获胜! 奖励: {reward}")
                    elif reward < 0:
                        st.error(f"游戏结束，失败! 奖励: {reward}")
                    else:
                        st.info(f"游戏结束，平局! 奖励: {reward}")
                
                # 更新状态
                state = next_state
                
                # 更新探索参数
                if hasattr(agent, 'exploration') and hasattr(agent.exploration, 'update'):
                    agent.exploration.update()
                
                # 稍微延迟以便观察
                time.sleep(0.5)

# 探索策略的实际效果
st.header("不同探索策略的特点总结")

st.markdown("""
| 策略 | 原理 | 优点 | 缺点 | 适用场景 |
|------|------|------|------|----------|
| ε-greedy | 以ε概率随机探索，1-ε概率利用 | 简单易实现，收敛有保证 | 完全随机的探索，不考虑动作价值差异 | 通用场景，特别是问题简单、动作空间较小时 |
| Boltzmann探索 | 根据Q值相对大小确定选择概率 | 探索有方向性，考虑动作间价值差异 | 需要温度参数调整，Q值差异大时可能退化为贪婪策略 | 动作价值有明显差异、希望探索"看起来有潜力"的动作时 |
| 噪声网络 | 网络权重中加入可学习噪声 | 状态相关探索，自适应程度，测试时可关闭 | 实现复杂，增加参数量 | 需要高效探索的复杂环境，如Atari游戏 |
""")

st.info("""
**选择探索策略的建议**:
1. 对于入门和简单问题，ε-greedy是不错的选择
2. 如果发现常规探索效率不高，可以尝试Boltzmann探索
3. 对于复杂环境和大规模问题，噪声网络通常能带来更好的性能
4. 实际应用中，可以通过实验比较不同探索策略在特定问题上的效果
""")

# 返回主页按钮
st.markdown("[返回主页](./)")
