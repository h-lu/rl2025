import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

from models.base_dqn import DQN, QNetwork
from models.double_dqn import DoubleDQN
from environment.blackjack_env import BlackjackWrapper
from utils.training import train_agent, evaluate_agent
from utils.visualization import plot_training_results, visualize_q_values

# 设置页面
st.set_page_config(page_title="Double DQN - DQN改进", layout="wide")

# 标题和介绍
st.title("Double DQN: 解决Q值过高估计问题")

st.markdown("""
## Double DQN的原理
标准DQN在计算目标Q值时，使用同一个网络（目标网络）既选择最大Q值对应的动作，又评估该动作的Q值：

$$ y_t = r_t + \gamma \max_{a'} Q(s_{t+1}, a'; \theta^-) $$

这容易导致对Q值的**过高估计 (overestimation)**，尤其是在学习初期或存在噪声时，某个被随机高估的Q值更容易被`max`操作选中，导致策略学习产生偏差。

### Double DQN的解决方案
Double DQN巧妙地将**动作选择**和**动作评估**解耦：
- 使用**当前网络**选择下一个状态的最优动作: $a^* = \argmax_{a'} Q(s_{t+1}, a'; \theta)$
- 使用**目标网络**评估这个选定动作的Q值: $Q(s_{t+1}, a^*; \theta^-)$

Double DQN的目标变为：

$$ y_t = r_t + \gamma Q(s_{t+1}, \argmax_{a'} Q(s_{t+1}, a'; \theta); \theta^-) $$

这种方式可以减少过高估计，因为两个网络的估计误差相互抵消而不是累积。
""")

# 可视化Double DQN的工作原理
st.subheader("Double DQN工作原理可视化")

# 创建示例数据
state_dim = 3
action_dim = 4
q_network = QNetwork(state_dim, action_dim)
target_network = QNetwork(state_dim, action_dim)

# 随机初始化参数，使两个网络不同
q_network_state_dict = q_network.state_dict()
target_network_state_dict = target_network.state_dict()

# 生成示例状态和奖励
example_state = torch.FloatTensor(np.array([15, 5, 0]))  # 示例状态
example_next_state = torch.FloatTensor(np.array([17, 5, 0]))  # 示例下一状态
example_reward = 0.0  # 示例奖励
gamma = 0.99  # 折扣因子

# 生成Q值示例（手动设置，使标准DQN和Double DQN的结果有明显差异）
with torch.no_grad():
    # 在线网络的Q值预测
    online_q_values = torch.FloatTensor([0.2, 0.5, 0.1, 0.8])
    # 目标网络的Q值预测 - 使其与在线网络有较大差异
    target_q_values = torch.FloatTensor([0.3, 0.9, 0.2, 0.6])
    
    # 标准DQN的目标计算
    standard_best_action = target_q_values.argmax().item()
    standard_target_value = example_reward + gamma * target_q_values.max().item()
    
    # Double DQN的目标计算
    double_best_action = online_q_values.argmax().item()
    double_target_value = example_reward + gamma * target_q_values[double_best_action].item()

# 使用matplotlib可视化
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(action_dim)
width = 0.35

# 绘制目标网络Q值
rects1 = ax.bar(x - width/2, target_q_values.numpy(), width, label='目标网络Q值')

# 绘制在线网络Q值
rects2 = ax.bar(x + width/2, online_q_values.numpy(), width, label='在线网络Q值')

ax.set_ylabel('Q值')
ax.set_xlabel('动作')
ax.set_title('标准DQN vs Double DQN在示例状态的Q值对比')
ax.set_xticks(x)
ax.legend()

# 标注标准DQN和Double DQN选择的动作
ax.annotate(f'标准DQN选择: 动作{standard_best_action}',
            xy=(standard_best_action - width/2, target_q_values[standard_best_action]),
            xytext=(standard_best_action - 1.5, target_q_values[standard_best_action] + 0.3),
            arrowprops=dict(facecolor='red', shrink=0.05))

ax.annotate(f'Double DQN选择: 动作{double_best_action} (基于在线网络)\n然后评估为 {target_q_values[double_best_action]:.2f} (基于目标网络)',
            xy=(double_best_action + width/2, online_q_values[double_best_action]),
            xytext=(double_best_action + 0.5, online_q_values[double_best_action] + 0.3),
            arrowprops=dict(facecolor='blue', shrink=0.05))

st.pyplot(fig)

# 显示计算结果
col1, col2 = st.columns(2)

with col1:
    st.info(f"""
    **标准DQN目标计算:**
    1. 使用目标网络评估下一状态的所有动作: {[f'{v:.2f}' for v in target_q_values.numpy()]}
    2. 选择Q值最大的动作: 动作{standard_best_action} (Q值 = {target_q_values[standard_best_action]:.2f})
    3. 计算目标Q值: {example_reward} + {gamma} × {target_q_values.max().item():.2f} = **{standard_target_value:.2f}**
    """)

with col2:
    st.info(f"""
    **Double DQN目标计算:**
    1. 使用在线网络选择下一状态的最佳动作: 动作{double_best_action} (Q值 = {online_q_values[double_best_action]:.2f})
    2. 使用目标网络评估该动作: Q值 = {target_q_values[double_best_action]:.2f}
    3. 计算目标Q值: {example_reward} + {gamma} × {target_q_values[double_best_action].item():.2f} = **{double_target_value:.2f}**
    """)

st.write(f"**差异: {(standard_target_value - double_target_value):.4f}**")

# 可能的偏差解释
if standard_target_value > double_target_value:
    st.warning("""
    在这个例子中，**标准DQN的目标值高于Double DQN**，这说明可能存在过高估计。
    
    Double DQN通过使用两个网络对同一个动作分别进行选择和评估，减少了这种过高估计的可能性。
    因为两个网络的估计误差可能相互抵消，而不是像标准DQN那样累积。
    """)
else:
    st.info("""
    在这个例子中，标准DQN和Double DQN的估计很接近，或者Double DQN估计值更高。
    
    这可能是因为在当前示例状态下，不存在明显的过高估计问题，或者随机初始化使得两个网络的预测偏差方向一致。
    
    在实际训练中，随着噪声和随机性的累积，标准DQN更容易产生系统性的过高估计。
    """)

# Double DQN在21点环境中的实验
st.header("Double DQN在21点环境中的实验")

# 参数设置
col1, col2 = st.columns(2)

with col1:
    num_episodes = st.slider("训练回合数", 500, 5000, 1000, 100)
    
    # 添加网络参数设置
    hidden_dim = st.slider("隐藏层大小", 16, 256, 64, 16)
    learning_rate = st.slider("学习率", 1e-5, 1e-2, 1e-3, format="%.5f")
    gamma = st.slider("折扣因子", 0.8, 0.999, 0.99, 0.001)
    
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
    
    我们将在21点环境中训练Double DQN智能体，并可选择与标准DQN进行对比。通过比较它们的学习曲线和最终性能，
    我们可以观察Double DQN是否能够减少标准DQN的过高估计问题，从而提高学习效率和最终性能。
    
    您可以调整各种超参数来观察它们对学习过程的影响：
    - **隐藏层大小**: 更大的隐藏层可以提高模型表达能力，但可能需要更多数据训练
    - **学习率**: 控制模型参数更新的步长，太大可能不稳定，太小可能收敛慢
    - **折扣因子**: 控制对未来奖励的重视程度
    - **探索参数**: 影响智能体探索环境的策略
    """)

# 运行实验按钮
if st.button("开始实验"):
    with st.spinner("实验进行中，请稍候..."):
        # 创建环境
        env = BlackjackWrapper(gym.make('Blackjack-v1'))
        
        # 创建智能体
        double_dqn = DoubleDQN(
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
            hidden_dim=hidden_dim  # 添加隐藏层大小参数
        )
        
        # 训练Double DQN
        rewards_double, losses_double, epsilons_double = train_agent(env, double_dqn, num_episodes=num_episodes)
        
        # 评估Double DQN
        avg_reward_double, success_rate_double = evaluate_agent(env, double_dqn, num_episodes=100)
        
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
                hidden_dim=hidden_dim  # 添加隐藏层大小参数
            )
            
            # 训练标准DQN
            rewards_standard, losses_standard, epsilons_standard = train_agent(env, standard_dqn, num_episodes=num_episodes)
            
            # 评估标准DQN
            avg_reward_standard, success_rate_standard = evaluate_agent(env, standard_dqn, num_episodes=100)
            
            # 创建Double DQN智能体
            double_dqn = DoubleDQN(
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
                hidden_dim=hidden_dim  # 添加隐藏层大小参数
            )
            
            # 绘制对比图
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            
            # 奖励对比
            axes[0].plot(rewards_standard, alpha=0.3, color='blue', label='标准DQN原始奖励')
            axes[0].plot(rewards_double, alpha=0.3, color='red', label='Double DQN原始奖励')
            
            # 计算移动平均
            window = 50
            rewards_standard_smoothed = np.convolve(rewards_standard, np.ones(window)/window, mode='valid')
            rewards_double_smoothed = np.convolve(rewards_double, np.ones(window)/window, mode='valid')
            
            axes[0].plot(range(len(rewards_standard_smoothed)), rewards_standard_smoothed, color='blue', label=f'标准DQN ({window}回合平均)')
            axes[0].plot(range(len(rewards_double_smoothed)), rewards_double_smoothed, color='red', label=f'Double DQN ({window}回合平均)')
            
            axes[0].set_xlabel('回合')
            axes[0].set_ylabel('奖励')
            axes[0].set_title('标准DQN vs Double DQN: 奖励对比')
            axes[0].legend()
            axes[0].grid(True, linestyle='--', alpha=0.6)
            
            # 损失对比
            axes[1].plot(losses_standard, alpha=0.3, color='blue', label='标准DQN原始损失')
            axes[1].plot(losses_double, alpha=0.3, color='red', label='Double DQN原始损失')
            
            # 计算移动平均
            losses_standard_smoothed = np.convolve(losses_standard, np.ones(window)/window, mode='valid')
            losses_double_smoothed = np.convolve(losses_double, np.ones(window)/window, mode='valid')
            
            axes[1].plot(range(len(losses_standard_smoothed)), losses_standard_smoothed, color='blue', label=f'标准DQN ({window}回合平均)')
            axes[1].plot(range(len(losses_double_smoothed)), losses_double_smoothed, color='red', label=f'Double DQN ({window}回合平均)')
            
            axes[1].set_xlabel('更新次数')
            axes[1].set_ylabel('损失')
            axes[1].set_title('标准DQN vs Double DQN: 损失对比')
            axes[1].legend()
            axes[1].grid(True, linestyle='--', alpha=0.6)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # 显示评估结果
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("标准DQN平均奖励", f"{avg_reward_standard:.3f}")
                st.metric("标准DQN成功率", f"{success_rate_standard*100:.1f}%")
                st.metric("标准DQN最终100回合平均奖励", f"{np.mean(rewards_standard[-100:]):.3f}")
            
            with col2:
                st.metric("Double DQN平均奖励", f"{avg_reward_double:.3f}")
                st.metric("Double DQN成功率", f"{success_rate_double*100:.1f}%")
                st.metric("Double DQN最终100回合平均奖励", f"{np.mean(rewards_double[-100:]):.3f}")
            
            # 对比结论
            reward_diff = avg_reward_double - avg_reward_standard
            success_rate_diff = success_rate_double - success_rate_standard
            
            if reward_diff > 0 and success_rate_diff > 0:
                st.success(f"""
                **实验结论:** Double DQN在平均奖励(+{reward_diff:.3f})和成功率(+{success_rate_diff*100:.1f}%)上均优于标准DQN。
                
                这表明在21点环境中，Double DQN通过解决过高估计问题，确实能够提高性能。过高估计可能导致标准DQN更倾向于选择更冒险的策略，
                在21点这种存在风险和不确定性的环境中表现不佳。
                """)
            elif reward_diff > 0:
                st.info(f"""
                **实验结论:** Double DQN在平均奖励上优于标准DQN(+{reward_diff:.3f})，但在成功率上差异不大。
                
                这可能表明Double DQN在21点环境中能够学到更好的策略，但过高估计问题在该环境中影响有限。
                """)
            else:
                st.warning(f"""
                **实验结论:** 在这次实验中，标准DQN表现似乎优于Double DQN。
                
                这可能是由于多种原因：
                1. 在特定的21点环境中，过高估计问题影响可能不明显
                2. 训练回合数不够，两者差异尚未充分体现
                3. 随机性因素导致的结果波动
                
                在更复杂的环境或更长的训练过程中，Double DQN通常能展现出更稳定的学习和更好的性能。
                """)
            
            # 可视化学习到的策略
            st.subheader("学习到的策略对比")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**标准DQN策略**")
                visualize_q_values(standard_dqn, render_type='policy')
            
            with col2:
                st.write("**Double DQN策略**")
                visualize_q_values(double_dqn, render_type='policy')
            
        else:
            # 只显示Double DQN的结果
            plot_training_results(rewards_double, losses_double, epsilons_double)
            
            st.success(f"""
            **实验结果:**
            - 平均奖励: {avg_reward_double:.3f}
            - 成功率: {success_rate_double*100:.1f}%
            - 最终100回合平均奖励: {np.mean(rewards_double[-100:]):.3f}
            """)
            
            # 可视化学习到的策略
            st.subheader("Double DQN学习到的策略")
            visualize_q_values(double_dqn, render_type='policy')

# 总结
st.header("Double DQN总结")

st.markdown("""
### 核心优势
- **减轻过高估计问题**: 通过分离动作选择和动作评估，减少由于噪声和随机性导致的Q值过高估计
- **提高学习稳定性**: 尤其在某些Q值估计不准确或噪声较大的环境中表现更好
- **实现简单**: 只需要修改计算目标Q值的方式，无需额外的网络结构或参数

### 适用场景
- 在奖励结构复杂或存在较大噪声的环境中特别有效
- 当标准DQN表现不稳定或收敛较慢时，可以尝试使用Double DQN
- 几乎可以与其他DQN改进方法（如Dueling DQN、优先经验回放）无缝结合

### 代码实现关键点
在计算目标Q值时，Standard DQN和Double DQN的区别：

```python
# 标准DQN
with torch.no_grad():
    next_q_values = target_network(next_states)
    targets = rewards + gamma * next_q_values.max(1)[0] * (1 - dones)

# Double DQN
with torch.no_grad():
    # 使用在线网络选择动作
    next_actions = q_network(next_states).argmax(1, keepdim=True)
    # 使用目标网络评估这些动作
    next_q_values = target_network(next_states).gather(1, next_actions).squeeze(1)
    targets = rewards + gamma * next_q_values * (1 - dones)
```
""")

# 返回主页按钮
st.markdown("[返回主页](./)")
