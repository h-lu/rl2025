import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

from models.base_dqn import DQN
from models.nstep_dqn import NStepDQN
from environment.blackjack_env import BlackjackWrapper
from utils.training import train_agent, evaluate_agent
from utils.visualization import plot_training_results, visualize_q_values

# 设置页面
st.set_page_config(page_title="多步学习 DQN - DQN改进", layout="wide")

# 标题和介绍
st.title("多步学习 DQN: 加速信息传播的改进")

st.markdown("""
## 多步学习DQN的原理
标准DQN使用单步TD目标来更新Q值，即只考虑一步后的奖励和下一个状态的最大Q值：

$$ y_t = r_t + \gamma \max_{a'} Q(s_{t+1}, a'; \theta^-) $$

而多步学习DQN使用n步回报代替单步TD目标，即考虑未来n步的累积折扣奖励和n步后状态的最大Q值：

$$ y_t^{(n)} = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + ... + \gamma^{n-1} r_{t+n-1} + \gamma^n \max_{a'} Q(s_{t+n}, a'; \theta^-) $$

### 多步学习DQN的优势
1. **更快的信息传播**: 奖励信号可以更快地传播到较早的状态，减少了对不准确Q值估计的依赖
2. **加速学习**: 特别是在奖励稀疏的环境中，多步学习可以显著加速学习过程
3. **更有效的探索**: 使用多步目标可以更快地发现有价值的路径
""")

# 可视化多步学习的效果
st.subheader("多步学习vs单步学习可视化")

# 创建示例数据
example_states = ["起始状态", "中间状态1", "中间状态2", "中间状态3", "终止状态"]
rewards = [0, 0, 0, 1]  # 只有在最后一步有奖励
gamma = 0.9

# 计算不同步长的目标值
single_step_targets = []
three_step_targets = []

# 单步目标
for i in range(len(example_states)-1):
    if i == len(example_states)-2:
        # 终止状态，无下一状态的Q值
        target = rewards[i]
    else:
        # 使用下一状态的最大Q值（假设为0.5）
        max_q = 0.5 if i < len(example_states)-3 else 0
        target = rewards[i] + gamma * max_q
    single_step_targets.append(target)

# 三步目标
for i in range(len(example_states)-3):
    r1 = rewards[i]
    r2 = rewards[i+1] * gamma
    r3 = rewards[i+2] * (gamma**2)
    
    # 使用三步后状态的最大Q值
    max_q = 0.5 if i+3 < len(example_states)-1 else 0
    future_q = max_q * (gamma**3)
    
    target = r1 + r2 + r3 + future_q
    three_step_targets.append(target)

# 补充三步回报
three_step_targets.append(rewards[-3] + gamma * rewards[-2] + gamma**2 * rewards[-1])
three_step_targets.append(rewards[-2] + gamma * rewards[-1])

# 绘制对比图
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(example_states)-1)
width = 0.35

ax.bar(x - width/2, single_step_targets, width, label='单步目标')
ax.bar(x + width/2, three_step_targets, width, label='三步目标')

ax.set_ylabel('目标值')
ax.set_xlabel('时间步')
ax.set_title('单步学习 vs 多步学习目标值对比')
ax.set_xticks(x)
ax.set_xticklabels([f'{s} → {example_states[i+1]}' for i, s in enumerate(example_states[:-1])])
ax.legend()

st.pyplot(fig)

# 解释多步学习
st.markdown("""
在上图中，我们可以看到多步学习和单步学习目标值的区别：

- **单步目标**只考虑下一步的奖励和下一个状态的价值
- **三步目标**考虑未来三步的累积折扣奖励和第三步后状态的价值

注意在有稀疏奖励的情况下（例如只有在最后一步获得奖励），多步学习能更快地将奖励信号传播到较早的状态，
使得智能体更早地学习到正确的行为。
""")

# 可视化信息传播
st.subheader("奖励信号传播可视化")

# 创建一个简单的MDP示例
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 单步TD学习的信息流
    
    在单步TD学习中，每个状态的价值仅从下一个状态的估计价值学习：
    
    1. 初始时，所有状态价值为0
    2. 终止状态获得+1奖励
    3. 价值逐渐向前传播，每次迭代只影响相邻状态
    
    这导致远离奖励的状态需要多次迭代才能获得准确的价值估计。
    """)
    
    # 绘制单步学习的信息流
    iterations = 5
    states = 5
    values = np.zeros((iterations, states))
    values[:, -1] = 1  # 终止状态奖励为1
    
    # 模拟单步TD学习的价值传播
    for i in range(1, iterations):
        for s in range(states-1):
            values[i, s] = 0.1 * gamma * values[i-1, s+1] + 0.9 * values[i-1, s]
    
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    im = ax1.imshow(values, cmap='viridis')
    ax1.set_xlabel('状态')
    ax1.set_ylabel('迭代次数')
    ax1.set_xticks(np.arange(states))
    ax1.set_yticks(np.arange(iterations))
    ax1.set_xticklabels([f'S{i}' for i in range(states)])
    ax1.set_yticklabels([f'迭代 {i}' for i in range(iterations)])
    fig1.colorbar(im, ax=ax1, label='状态价值')
    
    for i in range(iterations):
        for j in range(states):
            ax1.text(j, i, f'{values[i, j]:.2f}', ha='center', va='center', color='white' if values[i, j] > 0.5 else 'black')
    
    st.pyplot(fig1)

with col2:
    st.markdown("""
    ### 多步学习的信息流
    
    在多步学习中，每个状态的价值可以直接从多步后的状态和累积奖励学习：
    
    1. 初始时，所有状态价值为0
    2. 终止状态获得+1奖励
    3. 价值快速向前传播，多步连接允许远距离信息流动
    
    这使得即使远离奖励的状态也能在较少的迭代中获得更准确的价值估计。
    """)
    
    # 绘制多步学习的信息流
    n_step = 3
    iterations = 5
    states = 5
    values = np.zeros((iterations, states))
    values[:, -1] = 1  # 终止状态奖励为1
    
    # 模拟多步TD学习的价值传播
    for i in range(1, iterations):
        for s in range(states-1):
            # 计算每个状态可达的最远状态（受n_step限制）
            max_reach = min(s + n_step, states-1)
            # 多步连接使价值传播更快
            for r in range(s+1, max_reach+1):
                step_dist = r - s
                values[i, s] += 0.1 * (gamma ** step_dist) * values[i-1, r]
            values[i, s] += 0.9 * values[i-1, s]  # 保留部分旧价值
    
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    im = ax2.imshow(values, cmap='viridis')
    ax2.set_xlabel('状态')
    ax2.set_ylabel('迭代次数')
    ax2.set_xticks(np.arange(states))
    ax2.set_yticks(np.arange(iterations))
    ax2.set_xticklabels([f'S{i}' for i in range(states)])
    ax2.set_yticklabels([f'迭代 {i}' for i in range(iterations)])
    fig2.colorbar(im, ax=ax2, label='状态价值')
    
    for i in range(iterations):
        for j in range(states):
            ax2.text(j, i, f'{values[i, j]:.2f}', ha='center', va='center', color='white' if values[i, j] > 0.5 else 'black')
    
    st.pyplot(fig2)

# 多步学习DQN在21点环境中的实验
st.header("多步学习DQN在21点环境中的实验")

# 参数设置
col1, col2 = st.columns(2)

with col1:
    num_episodes = st.slider("训练回合数", 500, 5000, 1000, 100)
    
    # 添加网络参数设置
    hidden_dim = st.slider("隐藏层大小", 16, 256, 64, 16)
    learning_rate = st.slider("学习率", 1e-5, 1e-2, 1e-3, format="%.5f")
    gamma = st.slider("折扣因子", 0.8, 0.999, 0.99, 0.001)
    
    # 多步参数
    n_steps = st.slider("步数", 1, 10, 3, 1, help="使用多少步来计算TD目标")
    
    with st.expander("更多参数"):
        epsilon_start = st.slider("初始探索率", 0.5, 1.0, 1.0, 0.01)
        epsilon_end = st.slider("最终探索率", 0.01, 0.5, 0.1, 0.01)
        epsilon_decay = st.slider("探索率衰减", 0.9, 0.999, 0.995, 0.001)
        buffer_size = st.slider("经验缓冲区大小", 1000, 100000, 10000, 1000)
        batch_size = st.slider("批次大小", 16, 256, 64, 8)
        target_update = st.slider("目标网络更新频率", 1, 100, 10, 1)
    
    compare = st.checkbox("与标准DQN对比", value=True)

with col2:
    st.info("""
    **实验设置**
    
    我们将在21点环境中训练多步学习DQN智能体，并可选择与标准DQN进行对比。通过比较它们的学习曲线和最终性能，
    我们可以观察多步学习是否能够加速学习过程，改善性能。
    
    多步学习的关键参数是**步数**，它决定了一个经验中包含多少步的未来奖励。
    步数越大，信息传播得越远，但也可能引入更多的噪声和方差。
    
    您可以调整各种超参数来观察它们对学习过程的影响：
    - **步数**: 用于计算目标值的步数
    - **隐藏层大小**: 更大的隐藏层可以提高模型表达能力，但可能需要更多数据训练
    - **学习率**: 控制模型参数更新的步长，太大可能不稳定，太小可能收敛慢
    - **折扣因子**: 控制对未来奖励的重视程度
    """)

# 运行实验按钮
if st.button("开始实验"):
    with st.spinner("实验进行中，请稍候..."):
        # 创建环境
        env = BlackjackWrapper(gym.make('Blackjack-v1'))
        
        # 创建多步学习DQN智能体
        nstep_dqn = NStepDQN(
            state_dim=3,  # 黑杰克状态维度：玩家点数、庄家明牌、是否有可用A
            action_dim=env.action_space.n,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            buffer_size=buffer_size,
            batch_size=batch_size,
            target_update=target_update,
            hidden_dim=hidden_dim,
            n_steps=n_steps  # 设置多步学习的步数
        )
        
        # 训练多步学习DQN
        rewards_nstep, losses_nstep, epsilons_nstep = train_agent(env, nstep_dqn, num_episodes=num_episodes)
        
        # 评估多步学习DQN
        avg_reward_nstep, success_rate_nstep = evaluate_agent(env, nstep_dqn, num_episodes=100)
        
        # 如果需要与标准DQN对比
        if compare:
            # 创建标准DQN智能体
            standard_dqn = DQN(
                state_dim=3,  # 黑杰克状态维度：玩家点数、庄家明牌、是否有可用A
                action_dim=env.action_space.n,
                learning_rate=learning_rate,
                gamma=gamma,
                epsilon_start=epsilon_start,
                epsilon_end=epsilon_end,
                epsilon_decay=epsilon_decay,
                buffer_size=buffer_size,
                batch_size=batch_size,
                target_update=target_update,
                hidden_dim=hidden_dim
            )
            
            # 训练标准DQN
            rewards_standard, losses_standard, epsilons_standard = train_agent(env, standard_dqn, num_episodes=num_episodes)
            
            # 评估标准DQN
            avg_reward_standard, success_rate_standard = evaluate_agent(env, standard_dqn, num_episodes=100)
            
            # 绘制对比图
            fig, axes = plt.subplots(3, 1, figsize=(12, 15))
            
            # 奖励对比
            axes[0].plot(rewards_standard, alpha=0.3, color='blue', label='标准DQN原始奖励')
            axes[0].plot(rewards_nstep, alpha=0.3, color='purple', label=f'{n_steps}步DQN原始奖励')
            
            # 计算移动平均
            window = 50
            rewards_standard_smoothed = np.convolve(rewards_standard, np.ones(window)/window, mode='valid')
            rewards_nstep_smoothed = np.convolve(rewards_nstep, np.ones(window)/window, mode='valid')
            
            axes[0].plot(range(len(rewards_standard_smoothed)), rewards_standard_smoothed, color='blue', label=f'标准DQN ({window}回合平均)')
            axes[0].plot(range(len(rewards_nstep_smoothed)), rewards_nstep_smoothed, color='purple', label=f'{n_steps}步DQN ({window}回合平均)')
            
            axes[0].set_xlabel('回合')
            axes[0].set_ylabel('奖励')
            axes[0].set_title('标准DQN vs 多步学习DQN: 奖励对比')
            axes[0].legend()
            axes[0].grid(True, linestyle='--', alpha=0.6)
            
            # 损失对比
            axes[1].plot(losses_standard, alpha=0.3, color='blue', label='标准DQN原始损失')
            axes[1].plot(losses_nstep, alpha=0.3, color='purple', label=f'{n_steps}步DQN原始损失')
            
            # 计算移动平均
            losses_standard_smoothed = np.convolve(losses_standard, np.ones(window)/window, mode='valid')
            losses_nstep_smoothed = np.convolve(losses_nstep, np.ones(window)/window, mode='valid')
            
            axes[1].plot(range(len(losses_standard_smoothed)), losses_standard_smoothed, color='blue', label=f'标准DQN ({window}回合平均)')
            axes[1].plot(range(len(losses_nstep_smoothed)), losses_nstep_smoothed, color='purple', label=f'{n_steps}步DQN ({window}回合平均)')
            
            axes[1].set_xlabel('更新次数')
            axes[1].set_ylabel('损失')
            axes[1].set_title('标准DQN vs 多步学习DQN: 损失对比')
            axes[1].legend()
            axes[1].grid(True, linestyle='--', alpha=0.6)
            
            # 学习速度对比（累积奖励）
            cum_rewards_standard = np.cumsum(rewards_standard)
            cum_rewards_nstep = np.cumsum(rewards_nstep)
            
            axes[2].plot(cum_rewards_standard, color='blue', label='标准DQN累积奖励')
            axes[2].plot(cum_rewards_nstep, color='purple', label=f'{n_steps}步DQN累积奖励')
            
            axes[2].set_xlabel('回合')
            axes[2].set_ylabel('累积奖励')
            axes[2].set_title('标准DQN vs 多步学习DQN: 学习速度对比')
            axes[2].legend()
            axes[2].grid(True, linestyle='--', alpha=0.6)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # 显示评估结果
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("标准DQN平均奖励", f"{avg_reward_standard:.3f}")
                st.metric("标准DQN成功率", f"{success_rate_standard*100:.1f}%")
                st.metric("标准DQN最终100回合平均奖励", f"{np.mean(rewards_standard[-100:]):.3f}")
            
            with col2:
                st.metric(f"{n_steps}步DQN平均奖励", f"{avg_reward_nstep:.3f}")
                st.metric(f"{n_steps}步DQN成功率", f"{success_rate_nstep*100:.1f}%")
                st.metric(f"{n_steps}步DQN最终100回合平均奖励", f"{np.mean(rewards_nstep[-100:]):.3f}")
            
            # 对比结论
            reward_diff = avg_reward_nstep - avg_reward_standard
            success_rate_diff = success_rate_nstep - success_rate_standard
            
            if reward_diff > 0 and success_rate_diff > 0:
                st.success(f"""
                **实验结论:** {n_steps}步DQN在平均奖励(+{reward_diff:.3f})和成功率(+{success_rate_diff*100:.1f}%)上均优于标准DQN。
                
                这表明在21点环境中，多步学习通过更有效地传播奖励信息，能够提高学习效率和最终性能。
                特别是对于序列决策问题，多步学习可以更快地将奖励信息传播到早期状态。
                """)
            elif reward_diff > 0:
                st.info(f"""
                **实验结论:** {n_steps}步DQN在平均奖励上优于标准DQN(+{reward_diff:.3f})，但在成功率上差异不大。
                
                这可能表明多步学习在21点环境中能够学到更好的价值估计，但对策略的实际影响有限。
                试试增加步数或调整其他参数，可能会获得更明显的改进。
                """)
            else:
                st.warning(f"""
                **实验结论:** 在这次实验中，标准DQN表现似乎优于{n_steps}步DQN。
                
                这可能是由于多种原因：
                1. 步数选择不当，较大的步数可能引入过多噪声和方差
                2. 在21点这种回合较短的环境中，多步学习的优势可能不明显
                3. 训练回合数不够，两者差异尚未充分体现
                
                尝试减小步数（如2步），或在更复杂的环境中测试多步学习的效果。
                """)
            
            # 可视化学习到的策略
            st.subheader("学习到的策略对比")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**标准DQN策略**")
                visualize_q_values(standard_dqn, render_type='policy')
            
            with col2:
                st.write(f"**{n_steps}步DQN策略**")
                visualize_q_values(nstep_dqn, render_type='policy')
            
        else:
            # 只显示多步学习DQN的结果
            plot_training_results(rewards_nstep, losses_nstep, epsilons_nstep)
            
            st.success(f"""
            **实验结果:**
            - 平均奖励: {avg_reward_nstep:.3f}
            - 成功率: {success_rate_nstep*100:.1f}%
            - 最终100回合平均奖励: {np.mean(rewards_nstep[-100:]):.3f}
            """)
            
            # 可视化学习到的策略
            st.subheader(f"{n_steps}步DQN学习到的策略")
            visualize_q_values(nstep_dqn, render_type='policy')

# 总结
st.header("多步学习DQN总结")

st.markdown("""
### 核心优势
- **加速奖励信息传播**: 多步目标使得奖励信号能够更快地传播到较早的状态
- **减少对不准确估计的依赖**: 通过使用更多的实际奖励和更少的估计值，可以减少初期不准确Q值估计的影响
- **提高样本效率**: 从相同的经验中提取更多信息，提高样本利用效率

### 权衡考量
- **步数选择**: 步数n需要权衡偏差和方差
  - n较小: 偏差大，方差小，类似于标准DQN
  - n较大: 偏差小，方差大，学习可能更不稳定
- **环境特性**: 在不同环境中表现各异
  - 在奖励稀疏的环境中效果更好
  - 在回合较长的环境中优势更明显

### 代码实现关键点
多步学习DQN的实现主要集中在经验回放缓冲区的设计上。与标准DQN不同，多步学习需要特殊的缓冲区来计算n步回报：

def _get_n_step_info(self):
    \"\"\"计算n步回报和下一个状态\"\"\"
    reward, next_state, done = self.n_step_buffer[-1][-3:]
    
    for transition in reversed(list(self.n_step_buffer)[:-1]):
        r, n_s, d = transition[-3:]
        
        reward = r + self.gamma * reward * (1 - d)
        next_state, done = (n_s, d) if d else (next_state, done)
    
    return reward, next_state, done

在计算目标Q值时，需要考虑已经应用了n次gamma：

# 多步学习时，gamma已经在计算n步回报时考虑了多次
targets = rewards + (self.gamma ** self.n_steps) * next_q_values * (1 - dones)
""")

# 返回主页按钮
st.markdown("[返回主页](./)") 