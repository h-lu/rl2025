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
from models.dueling_dqn import DuelingDQN, DuelingQNetwork
from environment.blackjack_env import BlackjackWrapper
from utils.training import train_agent, evaluate_agent
from utils.visualization import plot_training_results, visualize_q_values

# 设置页面
st.set_page_config(page_title="Dueling DQN - DQN改进", layout="wide")

# 标题和介绍
st.title("Dueling DQN: 分离状态价值和动作优势")

st.markdown("""
## Dueling DQN的原理
标准DQN使用单一网络直接估计每个状态-动作对的Q值：

$$ Q(s, a; \theta) $$

而Dueling DQN将Q值分解为**状态价值函数V(s)**和**动作优势函数A(s,a)**：

$$ Q(s, a; \theta, \alpha, \beta) = V(s; \theta, \beta) + A(s, a; \theta, \alpha) $$

为了保证模型的可识别性，通常会调整优势函数使其平均值为0：

$$ Q(s, a; \theta, \alpha, \beta) = V(s; \theta, \beta) + \left(A(s, a; \theta, \alpha) - \frac{1}{|A|}\sum_{a'}A(s, a'; \theta, \alpha)\right) $$

### Dueling DQN的优势
1. **更高效的价值评估**: 在某些状态下，动作的选择并不重要，此时只需要准确估计状态价值
2. **更稳定的学习**: 状态价值函数更易学习，可以提高学习的稳定性
3. **更好的泛化**: 在未见过的状态上有更好的泛化能力
""")

# 添加Dueling DQN网络架构可视化
st.subheader("Dueling DQN网络架构可视化")

col1, col2 = st.columns([2, 3])

with col1:
    st.markdown("""
    ### Dueling DQN网络架构原理
    
    Dueling DQN将Q值分解为两个独立的部分：
    
    1. **状态价值函数 V(s)**: 
       - 评估处于特定状态的内在价值
       - 与动作选择无关
       - 回答"现在这个位置好不好"的问题
    
    2. **动作优势函数 A(s,a)**:
       - 评估在特定状态下选择某个动作相比其他动作的相对优势
       - 回答"在当前状态下，这个动作比平均水平好多少"的问题
    
    3. **聚合方法**:
       - Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
       - 减去优势的均值确保优势函数可以专注于表示相对优势
       - 这种设计解决了V值和A值的不可辨识问题
    """)

with col2:
    # 使用matplotlib绘制网络架构图
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    
    # 定义各层节点位置
    input_layer_y = np.array([0.2, 0.35, 0.5, 0.65, 0.8])
    shared_layer_y = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
    value_layer_y = np.array([0.4, 0.5, 0.6])
    advantage_layer_y = np.array([0.25, 0.35, 0.45, 0.55, 0.65, 0.75])
    output_layer_y = np.array([0.3, 0.5, 0.7])
    
    # 绘制节点
    # 输入层
    input_color = '#B3E5FC'
    for y in input_layer_y:
        circle = plt.Circle((0.1, y), 0.05, color=input_color, fill=True)
        ax.add_patch(circle)
    
    # 共享层
    shared_color = '#C8E6C9'
    for y in shared_layer_y:
        circle = plt.Circle((0.3, y), 0.05, color=shared_color, fill=True)
        ax.add_patch(circle)
    
    # 价值流
    value_color = '#FFF9C4'
    for y in value_layer_y:
        circle = plt.Circle((0.5, y), 0.05, color=value_color, fill=True)
        ax.add_patch(circle)
    
    # 优势流
    advantage_color = '#FFCCBC'
    for y in advantage_layer_y:
        circle = plt.Circle((0.5, y), 0.05, color=advantage_color, fill=True)
        ax.add_patch(circle)
    
    # 输出层
    output_color = '#E1BEE7'
    for y in output_layer_y:
        circle = plt.Circle((0.8, y), 0.05, color=output_color, fill=True)
        ax.add_patch(circle)
    
    # 绘制连接线
    # 输入层到共享层
    for y1 in input_layer_y:
        for y2 in shared_layer_y:
            ax.plot([0.15, 0.25], [y1, y2], 'gray', alpha=0.3)
    
    # 共享层到价值流和优势流
    for y1 in shared_layer_y:
        for y2 in value_layer_y:
            ax.plot([0.35, 0.45], [y1, y2], 'gray', alpha=0.3)
        for y2 in advantage_layer_y:
            ax.plot([0.35, 0.45], [y1, y2], 'gray', alpha=0.3)
    
    # 价值流和优势流到输出层
    ax.plot([0.55, 0.7], [0.5, 0.5], 'blue', alpha=0.7, linewidth=2)
    for y in advantage_layer_y:
        ax.plot([0.55, 0.65], [y, 0.5], 'red', alpha=0.5)
    
    # 合并后到输出层
    for y in output_layer_y:
        ax.plot([0.7, 0.75], [0.5, y], 'gray', alpha=0.5)
    
    # 添加文本标签
    ax.text(0.1, 0.9, '输入层', fontsize=12, ha='center')
    ax.text(0.3, 0.9, '共享特征层', fontsize=12, ha='center')
    ax.text(0.5, 0.9, '分离流', fontsize=12, ha='center')
    ax.text(0.5, 0.2, '优势流 A(s,a)', fontsize=12, ha='center', color='#D84315')
    ax.text(0.5, 0.8, '价值流 V(s)', fontsize=12, ha='center', color='#0D47A1')
    ax.text(0.7, 0.4, '聚合', fontsize=12, ha='center')
    ax.text(0.8, 0.9, '输出层\nQ(s,a)', fontsize=12, ha='center')
    
    # 添加聚合公式
    formula = r"$Q(s,a) = V(s) + A(s,a) - \frac{1}{|A|}\sum_a A(s,a)$"
    ax.text(0.7, 0.3, formula, fontsize=10, ha='center', bbox=dict(facecolor='white', alpha=0.5))
    
    # 设置图形显示范围和属性
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    st.pyplot(fig)

st.markdown("""
### Dueling DQN的优势

1. **更高效地学习状态价值**：
   - 在某些状态下，不同动作的价值差异很小，此时状态本身的价值更为重要
   - 通过显式建模状态价值，可以更快速地学习状态评估

2. **动作优势的专注学习**：
   - 优势流可以专注于学习动作间的相对优势
   - 不需要考虑状态本身的基础价值

3. **适用场景**：
   - 特别适合那些状态价值比动作选择更重要的环境
   - 或者动作空间很大但很多动作效果相似的情况

4. **稳定性提升**：
   - 分离价值和优势有助于减少训练波动
   - 特别是当某些状态下的Q值估计不稳定时
""")

# Dueling DQN的数学原理
st.subheader("Dueling DQN的数学原理")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 价值函数的分解
    
    标准DQN直接估计Q(s,a)值，而Dueling DQN将Q值分解为两个部分：
    
    - **状态价值 V(s)**: 表示处于状态s的价值（与采取什么动作无关）
    - **动作优势 A(s,a)**: 表示在状态s下采取动作a相比于其他动作的相对优势
    
    这种分解使网络能够学习哪些状态是有价值的，而不需要同时学习每个动作的效果，
    并且可以学习哪些动作在特定状态下比其他动作更好。
    """)
    
    st.latex(r'''
    Q(s, a) = V(s) + A(s, a)
    ''')

with col2:
    st.markdown("""
    ### 解决唯一性问题
    
    然而，上述分解并不是唯一的。例如，如果我们将V(s)增加一个常数c，同时将所有A(s,a)减少相同的常数c，
    得到的Q(s,a)值将保持不变。
    
    为了确保分解的唯一性，我们对优势函数进行调整，使其均值为0：
    """)
    
    st.latex(r'''
    Q(s, a) = V(s) + \left(A(s, a) - \frac{1}{|A|}\sum_{a'}A(s, a')\right)
    ''')
    
    st.markdown("""
    这样调整后，对于最优动作，优势为正；对于次优动作，优势为负。而状态价值V(s)将代表该状态下的平均动作价值。
    """)

# 模拟Dueling DQN的价值分解
st.subheader("Dueling DQN价值分解示例")

# 创建示例状态和对应的价值
example_states = ["玩家17点，庄家10点", "玩家21点，庄家6点", "玩家12点，庄家10点"]
state_values = [0.0, 0.8, -0.5]  # 状态价值
adv_stop = [-0.2, 0.1, -0.3]  # 停牌的优势
adv_hit = [0.2, -0.1, 0.3]  # 要牌的优势
q_stop = [state_values[i] + adv_stop[i] for i in range(len(example_states))]  # 停牌的Q值
q_hit = [state_values[i] + adv_hit[i] for i in range(len(example_states))]  # 要牌的Q值

# 创建表格数据
table_data = {
    "状态": example_states,
    "状态价值 V(s)": state_values,
    "停牌优势 A(s,0)": adv_stop,
    "要牌优势 A(s,1)": adv_hit,
    "停牌Q值 Q(s,0)": q_stop,
    "要牌Q值 Q(s,1)": q_hit,
    "最佳动作": ["要牌" if q_hit[i] > q_stop[i] else "停牌" for i in range(len(example_states))]
}

# 显示表格
st.table(table_data)

st.markdown("""
在上表中，我们可以看到Dueling DQN如何分解状态价值和动作优势：

1. **状态价值V(s)**: 表示该状态的整体价值（与动作无关）
2. **动作优势A(s,a)**: 表示各个动作相对于平均水平的优势
3. **最终Q值**: 状态价值加上动作优势得到最终的Q值

例如，在状态"玩家17点，庄家10点"中：
- 状态价值为0.0，表示这个状态期望收益为中性
- 停牌优势为-0.2，要牌优势为0.2
- 这表明在这种情况下要牌更有利（Q值更高）
""")

# Dueling DQN在21点环境中的实验
st.header("Dueling DQN在21点环境中的实验")

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
    
    我们将在21点环境中训练Dueling DQN智能体，并可选择与标准DQN进行对比。通过比较它们的学习曲线和最终性能，
    我们可以观察到Dueling架构是否能够提高学习效率和最终性能。
    
    特别地，我们希望观察Dueling DQN是否能更好地学习状态的价值，从而在某些状态下做出更好的决策。
    
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
        
        # 创建智能体，使用用户设置的参数
        dueling_dqn = DuelingDQN(
            state_dim=3,  # 黑杰克状态维度：玩家点数、庄家明牌、是否有可用A
            action_dim=env.action_space.n,
            hidden_dim=hidden_dim,  # 使用用户设置的隐藏层大小
            learning_rate=learning_rate,  # 使用用户设置的学习率
            gamma=gamma,  # 使用用户设置的折扣因子
            epsilon_start=epsilon_start,  # 使用用户设置的初始探索率
            epsilon_end=epsilon_end,  # 使用用户设置的最终探索率
            epsilon_decay=epsilon_decay,  # 使用用户设置的探索率衰减
            buffer_size=buffer_size,  # 使用用户设置的缓冲区大小
            batch_size=batch_size,  # 使用用户设置的批次大小
            target_update=target_update  # 使用用户设置的目标网络更新频率
        )
        
        # 训练Dueling DQN
        rewards_dueling, losses_dueling, epsilons_dueling = train_agent(env, dueling_dqn, num_episodes=num_episodes)
        
        # 评估Dueling DQN
        avg_reward_dueling, success_rate_dueling = evaluate_agent(env, dueling_dqn, num_episodes=100)
        
        # 如果需要与标准DQN对比
        if compare:
            # 创建标准DQN智能体，使用相同的超参数以便公平比较
            standard_dqn = DQN(
                state_dim=3,  # 黑杰克状态维度：玩家点数、庄家明牌、是否有可用A
                action_dim=env.action_space.n,
                hidden_dim=hidden_dim,  # 使用用户设置的隐藏层大小
                learning_rate=learning_rate,  # 使用用户设置的学习率
                gamma=gamma,  # 使用用户设置的折扣因子
                epsilon_start=epsilon_start,  # 使用用户设置的初始探索率
                epsilon_end=epsilon_end,  # 使用用户设置的最终探索率
                epsilon_decay=epsilon_decay,  # 使用用户设置的探索率衰减
                buffer_size=buffer_size,  # 使用用户设置的缓冲区大小
                batch_size=batch_size,  # 使用用户设置的批次大小
                target_update=target_update  # 使用用户设置的目标网络更新频率
            )
            
            # 训练标准DQN
            rewards_standard, losses_standard, epsilons_standard = train_agent(env, standard_dqn, num_episodes=num_episodes)
            
            # 评估标准DQN
            avg_reward_standard, success_rate_standard = evaluate_agent(env, standard_dqn, num_episodes=100)
            
            # 绘制对比图
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            
            # 奖励对比
            axes[0].plot(rewards_standard, alpha=0.3, color='blue', label='标准DQN原始奖励')
            axes[0].plot(rewards_dueling, alpha=0.3, color='green', label='Dueling DQN原始奖励')
            
            # 计算移动平均
            window = 50
            rewards_standard_smoothed = np.convolve(rewards_standard, np.ones(window)/window, mode='valid')
            rewards_dueling_smoothed = np.convolve(rewards_dueling, np.ones(window)/window, mode='valid')
            
            axes[0].plot(range(len(rewards_standard_smoothed)), rewards_standard_smoothed, color='blue', label=f'标准DQN ({window}回合平均)')
            axes[0].plot(range(len(rewards_dueling_smoothed)), rewards_dueling_smoothed, color='green', label=f'Dueling DQN ({window}回合平均)')
            
            axes[0].set_xlabel('回合')
            axes[0].set_ylabel('奖励')
            axes[0].set_title('标准DQN vs Dueling DQN: 奖励对比')
            axes[0].legend()
            axes[0].grid(True, linestyle='--', alpha=0.6)
            
            # 损失对比
            axes[1].plot(losses_standard, alpha=0.3, color='blue', label='标准DQN原始损失')
            axes[1].plot(losses_dueling, alpha=0.3, color='green', label='Dueling DQN原始损失')
            
            # 计算移动平均
            losses_standard_smoothed = np.convolve(losses_standard, np.ones(window)/window, mode='valid')
            losses_dueling_smoothed = np.convolve(losses_dueling, np.ones(window)/window, mode='valid')
            
            axes[1].plot(range(len(losses_standard_smoothed)), losses_standard_smoothed, color='blue', label=f'标准DQN ({window}回合平均)')
            axes[1].plot(range(len(losses_dueling_smoothed)), losses_dueling_smoothed, color='green', label=f'Dueling DQN ({window}回合平均)')
            
            axes[1].set_xlabel('更新次数')
            axes[1].set_ylabel('损失')
            axes[1].set_title('标准DQN vs Dueling DQN: 损失对比')
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
                st.metric("Dueling DQN平均奖励", f"{avg_reward_dueling:.3f}")
                st.metric("Dueling DQN成功率", f"{success_rate_dueling*100:.1f}%")
                st.metric("Dueling DQN最终100回合平均奖励", f"{np.mean(rewards_dueling[-100:]):.3f}")
            
            # 对比结论
            reward_diff = avg_reward_dueling - avg_reward_standard
            success_rate_diff = success_rate_dueling - success_rate_standard
            
            if reward_diff > 0 and success_rate_diff > 0:
                st.success(f"""
                **实验结论:** Dueling DQN在平均奖励(+{reward_diff:.3f})和成功率(+{success_rate_diff*100:.1f}%)上均优于标准DQN。
                
                这表明在21点环境中，Dueling架构通过分离状态价值和动作优势，能够更有效地学习策略。
                特别是在状态更为重要而动作选择相对次要的情况下，Dueling DQN的优势更为明显。
                """)
            elif reward_diff > 0:
                st.info(f"""
                **实验结论:** Dueling DQN在平均奖励上优于标准DQN(+{reward_diff:.3f})，但在成功率上差异不大。
                
                这可能表明Dueling DQN在21点环境中能够学到更好的价值估计，但在这种小规模问题上，
                与标准DQN的差异不太显著。在更复杂的环境中，Dueling结构的优势可能会更明显。
                """)
            else:
                st.warning(f"""
                **实验结论:** 在这次实验中，标准DQN表现似乎优于Dueling DQN。
                
                这可能是由于多种原因：
                1. 在21点这种相对简单的环境中，标准DQN已经足够表达所需的价值函数
                2. 训练回合数不够，两者差异尚未充分体现
                3. 随机性因素导致的结果波动
                
                通常在状态空间更大、更复杂的问题上，Dueling DQN的优势会更明显。
                """)
            
            # 可视化学习到的策略
            st.subheader("学习到的策略对比")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**标准DQN策略**")
                visualize_q_values(standard_dqn, render_type='policy')
            
            with col2:
                st.write("**Dueling DQN策略**")
                visualize_q_values(dueling_dqn, render_type='policy')
            
        else:
            # 只显示Dueling DQN的结果
            plot_training_results(rewards_dueling, losses_dueling, epsilons_dueling)
            
            st.success(f"""
            **实验结果:**
            - 平均奖励: {avg_reward_dueling:.3f}
            - 成功率: {success_rate_dueling*100:.1f}%
            - 最终100回合平均奖励: {np.mean(rewards_dueling[-100:]):.3f}
            """)
            
            # 可视化学习到的策略
            st.subheader("Dueling DQN学习到的策略")
            visualize_q_values(dueling_dqn, render_type='policy')

# 总结
st.header("Dueling DQN总结")

st.markdown("""
### 核心优势
- **更高效的价值评估**: 在某些状态下，动作的选择并不重要，可以专注于学习状态的价值
- **更有效的泛化**: 通过共享特征提取层和明确区分状态价值与动作优势，可以更好地泛化到未见过的状态
- **更稳定的学习**: 尤其在价值估计噪声较大的环境中，分离架构提供了更稳定的学习

### 适用场景
- 在某些状态下，动作选择影响较小的环境
- 状态空间大，需要更有效泛化的问题
- 通常与其他DQN改进（如Double DQN、优先经验回放）结合使用效果更佳

### 代码实现关键点
Dueling DQN的关键在于网络架构设计，将Q值分解为状态价值和动作优势：

```python
class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(DuelingQNetwork, self).__init__()
        
        # 共享特征层
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 价值流
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 优势流
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        features = self.feature_layer(x)
        
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # 组合价值和优势，确保优势的平均值为0
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values
```
""")

# 返回主页按钮
st.markdown("[返回主页](./)")
