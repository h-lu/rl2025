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
from models.per_dqn import PERDQN
from utils.buffer import ReplayBuffer, PrioritizedReplayBuffer, SumTree
from environment.blackjack_env import BlackjackWrapper
from utils.training import train_agent, evaluate_agent
from utils.visualization import plot_training_results, visualize_q_values

# 设置页面
st.set_page_config(page_title="优先经验回放DQN - DQN改进", layout="wide")

# 标题和介绍
st.title("优先经验回放: 更有效地利用经验")

st.markdown("""
## 优先经验回放(PER)的原理
标准DQN从经验回放缓冲区中**均匀随机**采样经验进行学习。这种方式忽略了一个事实：**不同经验的重要性不同**。

优先经验回放的核心思想是：**更多地从重要经验中学习**。

### PER的关键机制：
1. **根据TD误差为经验分配优先级**：TD误差越大的经验，信息量可能越大
2. **基于优先级的采样**：优先级高的经验被采样的概率更高
3. **重要性采样权重**：修正由偏向采样引入的偏差
""")

# 优先经验回放的优势和工作流程
st.subheader("优先经验回放的工作流程")

# 创建流程图
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    ### PER的优势
    
    1. **学习效率更高**：优先学习那些"惊奇"的、与预期不符的经验
    2. **样本利用率提高**：重要的稀有经验不再因为随机采样而被忽略
    3. **可以与其他DQN改进方法结合**：如Double DQN、Dueling DQN等
    
    ### PER的潜在问题
    
    1. **采样偏差**：不再是均匀随机采样，可能导致偏差
    2. **对噪声敏感**：在有噪声的环境中，噪声可能导致误导性的高TD误差
    3. **计算开销增加**：优先级队列的维护需要额外计算
    """)

# 创建一个关于SumTree的可视化
with col2:
    # 创建一个简单的SumTree可视化
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 绘制树结构
    def draw_tree(ax, depth=3):
        # 节点位置
        positions = {}
        texts = {}
        levels = {}
        
        # 设置每个节点的位置
        for i in range(2**depth - 1):
            level = int(np.floor(np.log2(i+1)))
            levels[i] = level
            horizontal_position = (i - (2**level - 1)) / (2**level) + 1/(2**(level+1))
            positions[i] = (horizontal_position, 1 - level / depth)
            
            if i == 0:  # 根节点
                texts[i] = "总优先级\n10.5"
            elif level == depth - 1:  # 叶节点 (对应经验)
                # 确定叶节点的优先级
                priorities = [3.5, 0.5, 2.0, 4.5]
                leaf_idx = i - (2**(depth-1) - 1)
                texts[i] = f"经验{leaf_idx}\n优先级: {priorities[leaf_idx]}"
            else:  # 内部节点
                if i == 1:
                    texts[i] = "6.0"
                else:
                    texts[i] = "4.5"
        
        # 绘制连线
        for i in range(1, 2**depth - 1):
            parent = (i - 1) // 2
            ax.plot([positions[parent][0], positions[i][0]], 
                    [positions[parent][1], positions[i][1]], 
                    'k-', lw=1)
        
        # 绘制节点
        for i, (pos, txt) in enumerate(zip(positions.values(), texts.values())):
            level = levels[i]
            if level == 0:  # 根节点
                node_color = 'lightblue'
            elif level == depth - 1:  # 叶节点
                node_color = 'lightgreen'
            else:  # 内部节点
                node_color = 'lightgray'
                
            circle = plt.Circle(pos, 0.05, color=node_color, ec='k', zorder=2)
            ax.add_artist(circle)
            ax.text(pos[0], pos[1], txt, ha='center', va='center', 
                   fontsize=8, bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
    
    # 绘制SumTree
    draw_tree(ax)
    
    # 标注说明
    ax.text(0.5, 0.01, "SumTree用于高效存储和采样优先级经验", ha='center', fontsize=10)
    
    # 美化
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('SumTree结构示例')
    
    st.pyplot(fig)
    
    st.markdown("""
    **SumTree工作原理**：
    - 叶节点存储单个经验的优先级
    - 非叶节点存储其子节点优先级之和
    - 根节点值是所有优先级的总和
    - 采样时，选择一个[0, 总优先级]之间的随机值，然后遍历树找到对应的叶节点
    """)

# 添加SumTree结构可视化
st.subheader("SumTree数据结构可视化")

col1, col2 = st.columns([1.5, 2])

with col1:
    st.markdown("""
    ### SumTree原理
    
    **SumTree**是优先经验回放(PER)的核心数据结构，它是一种二叉树，具有以下特点：
    
    1. **叶节点**:
       - 存储实际经验（状态、动作、奖励等）
       - 包含经验的优先级值（通常是TD误差的绝对值）
    
    2. **内部节点**:
       - 存储其子节点优先级的总和
       - 根节点值代表所有经验的优先级总和
    
    3. **关键操作**:
       - **添加**: O(log n) - 更新从叶到根的路径
       - **采样**: O(log n) - 基于优先级的高效采样
       - **更新**: O(log n) - 更新经验的优先级
    
    SumTree使得按优先级采样的复杂度从朴素方法的O(n)降低到O(log n)，
    大大提高了优先经验回放的效率。
    """)

with col2:
    # 使用matplotlib绘制SumTree结构
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    
    # 使用自定义函数绘制树节点
    def draw_node(x, y, radius, value, color, priority=None):
        circle = plt.Circle((x, y), radius, color=color, alpha=0.7)
        ax.add_patch(circle)
        if priority is not None:
            ax.text(x, y, f"{value}\n(p={priority})", ha='center', va='center', fontsize=9)
        else:
            ax.text(x, y, f"{value}", ha='center', va='center', fontsize=9)
    
    # 绘制树的连接线
    def draw_connection(x1, y1, x2, y2):
        ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.5)
    
    # 绘制箭头
    def draw_arrow(x1, y1, x2, y2, color, text=None):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color=color))
        if text:
            ax.text((x1+x2)/2, (y1+y2)/2, text, ha='center', va='center', 
                    fontsize=9, color=color)
    
    # 根节点
    root_x, root_y = 0.5, 0.9
    draw_node(root_x, root_y, 0.05, "36", "#4CAF50")
    
    # 第二层节点
    node_l_x, node_l_y = 0.25, 0.7
    node_r_x, node_r_y = 0.75, 0.7
    draw_node(node_l_x, node_l_y, 0.05, "10", "#81C784")
    draw_node(node_r_x, node_r_y, 0.05, "26", "#81C784")
    
    # 连接根节点和第二层
    draw_connection(root_x, root_y-0.05, node_l_x, node_l_y+0.05)
    draw_connection(root_x, root_y-0.05, node_r_x, node_r_y+0.05)
    
    # 第三层节点
    node_ll_x, node_ll_y = 0.125, 0.5
    node_lr_x, node_lr_y = 0.375, 0.5
    node_rl_x, node_rl_y = 0.625, 0.5
    node_rr_x, node_rr_y = 0.875, 0.5
    
    draw_node(node_ll_x, node_ll_y, 0.05, "4", "#A5D6A7")
    draw_node(node_lr_x, node_lr_y, 0.05, "6", "#A5D6A7")
    draw_node(node_rl_x, node_rl_y, 0.05, "9", "#A5D6A7")
    draw_node(node_rr_x, node_rr_y, 0.05, "17", "#A5D6A7")
    
    # 连接第二层和第三层
    draw_connection(node_l_x, node_l_y-0.05, node_ll_x, node_ll_y+0.05)
    draw_connection(node_l_x, node_l_y-0.05, node_lr_x, node_lr_y+0.05)
    draw_connection(node_r_x, node_r_y-0.05, node_rl_x, node_rl_y+0.05)
    draw_connection(node_r_x, node_r_y-0.05, node_rr_x, node_rr_y+0.05)
    
    # 叶节点（实际经验）
    leaf_y = 0.25
    leaf_xs = [0.0625, 0.1875, 0.3125, 0.4375, 0.5625, 0.6875, 0.8125, 0.9375]
    priorities = [1, 3, 2, 4, 3, 6, 7, 10]
    
    for i, (leaf_x, priority) in enumerate(zip(leaf_xs, priorities)):
        draw_node(leaf_x, leaf_y, 0.04, f"经验{i+1}", "#E8F5E9", priority)
        
    # 连接第三层和叶节点
    for i in range(2):
        draw_connection(node_ll_x, node_ll_y-0.05, leaf_xs[i], leaf_y+0.04)
    for i in range(2, 4):
        draw_connection(node_lr_x, node_lr_y-0.05, leaf_xs[i], leaf_y+0.04)
    for i in range(4, 6):
        draw_connection(node_rl_x, node_rl_y-0.05, leaf_xs[i], leaf_y+0.04)
    for i in range(6, 8):
        draw_connection(node_rr_x, node_rr_y-0.05, leaf_xs[i], leaf_y+0.04)
    
    # 箭头展示采样过程
    ax.text(0.1, 0.15, "1. 生成随机值r (0-36)", fontsize=9, color='blue')
    draw_arrow(0.15, 0.15, 0.5, 0.85, 'blue', "r=19")
    
    ax.text(0.2, 0.05, "2. 在树中导航寻找对应经验", fontsize=9, color='red')
    # 根据r=19值的导航路径
    draw_arrow(0.52, 0.85, 0.75, 0.7, 'red', "19>10")
    draw_arrow(0.77, 0.65, 0.625, 0.5, 'red', "19-10=9")
    draw_arrow(0.605, 0.45, 0.5625, 0.29, 'red', "9>6")
    
    # 设置轴范围和属性
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # 添加标题和图例
    ax.text(0.5, 0.97, "SumTree结构 - 总优先级和: 36", ha='center', fontsize=14)
    
    st.pyplot(fig)

st.markdown("""
### SumTree的关键操作

SumTree支持三个核心操作，使得优先经验回放(PER)能够高效实现：

1. **添加新经验**（O(log n)）：
   - 在叶节点层找到一个位置（通常替换最旧的数据或最低优先级的数据）
   - 存储经验及其初始优先级
   - 从叶节点向上更新路径上所有父节点的值（优先级和）

2. **基于优先级采样**（O(log n)）：
   - 生成一个随机数 r，范围在 [0, 总优先级]
   - 从根节点开始，递归向下搜索树，寻找包含随机数r的节点
   - 如果左子树的和 >= r，就向左搜索；否则，用r减去左子树的和，向右搜索
   - 一直到达叶节点，该叶节点包含的经验就是采样结果

3. **更新优先级**（O(log n)）：
   - 当计算出新的TD误差后，更新对应经验的优先级
   - 从该叶节点开始，向上更新路径上所有父节点的值

SumTree保证了采样概率正比于优先级，同时使得操作时间复杂度保持在O(log n)，非常适合大型经验缓冲区的实现。
""")

# 优先经验回放的采样过程
st.subheader("优先经验回放采样过程")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 朴素方法与SumTree对比
    
    **朴素方法**（不使用SumTree）：
    
    1. 计算所有经验的累积优先级
    2. 随机生成一个小于累积优先级的值
    3. 逐一累加各经验的优先级，直到超过随机值
    4. 复杂度：O(n)
    
    **使用SumTree**：
    
    1. 获取根节点的总优先级
    2. 随机生成一个小于总优先级的值
    3. 递归向下搜索树，在O(log n)时间内找到对应经验
    
    对于大型回放缓冲区（如包含百万级经验），
    这种优化是非常重要的，可以将采样速度提高数个数量级。
    """)

with col2:
    # 绘制采样时间对比图
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # 经验缓冲区大小
    buffer_sizes = [1000, 10000, 100000, 1000000]
    
    # 模拟的采样时间（以毫秒为单位）
    naive_times = [1, 10, 100, 1000]  # O(n)
    sumtree_times = [0.01, 0.02, 0.03, 0.04]  # O(log n)
    
    x = np.arange(len(buffer_sizes))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, naive_times, width, label='朴素方法 (O(n))', color='#F44336', alpha=0.8)
    rects2 = ax.bar(x + width/2, sumtree_times, width, label='SumTree (O(log n))', color='#4CAF50', alpha=0.8)
    
    # 添加标签和标题
    ax.set_ylabel('采样时间 (毫秒，对数尺度)')
    ax.set_xlabel('经验缓冲区大小')
    ax.set_title('不同方法的采样时间对比')
    ax.set_xticks(x)
    ax.set_xticklabels(buffer_sizes)
    ax.set_yscale('log')
    ax.set_ylim(0.005, 2000)
    ax.legend()
    
    # 添加数值标签
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                         xy=(rect.get_x() + rect.get_width()/2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    ax.grid(True, linestyle='--', alpha=0.5)
    
    st.pyplot(fig)

st.markdown("""
### 优先经验回放的完整流程

1. **初始化**：
   - 创建一个SumTree结构，用于存储经验和优先级
   - 初始经验的优先级通常设为最大优先级，确保新经验被充分学习

2. **加入新经验**：
   - 计算新经验的优先级（初次加入时可能使用最大优先级）
   - 将经验和优先级添加到SumTree中
   - 更新树中受影响的节点

3. **采样经验**：
   - 根据总优先级生成随机数
   - 使用SumTree高效找到对应的经验
   - 计算重要性采样权重，以校正引入的偏差

4. **更新优先级**：
   - 使用网络计算TD误差
   - 更新对应经验的优先级（通常是TD误差的函数）
   - 在SumTree中更新对应的节点和路径

这种方法确保了高TD误差（"令人惊讶"）的经验被更频繁地采样，同时保持了操作的高效性。
""")

# 优先经验回放在21点环境中的实验
st.header("优先经验回放在21点环境中的实验")

# 参数设置
col1, col2 = st.columns(2)

with col1:
    num_episodes = st.slider("训练回合数", 500, 5000, 1000, 100)
    
    compare = st.checkbox("与标准DQN对比", value=True)
    
    # 高级参数
    with st.expander("高级参数"):
        alpha = st.slider("优先级指数 (α)", 0.0, 1.0, 0.6, 0.1)
        beta_start = st.slider("初始β值", 0.0, 1.0, 0.4, 0.1)

with col2:
    st.info("""
    **实验设置**
    
    我们将在21点环境中训练优先经验回放DQN智能体，并可选择与标准DQN进行对比。通过比较它们的学习曲线和最终性能，
    我们可以观察优先经验回放是否能够提高学习效率和最终性能。
    
    特别地，我们希望观察优先经验回放是否能更有效地利用重要经验，加速学习过程。
    """)

# 运行实验按钮
if st.button("开始实验"):
    with st.spinner("实验进行中，请稍候..."):
        # 创建环境
        env = BlackjackWrapper(gym.make('Blackjack-v1'))
        
        # 创建智能体
        per_dqn = PERDQN(
            state_dim=3,  # 黑杰克状态维度：玩家点数、庄家明牌、是否有可用A
            action_dim=env.action_space.n,
            learning_rate=0.001,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay=0.995,
            buffer_size=10000,
            batch_size=64,
            target_update=10,
            alpha=alpha,
            beta_start=beta_start
        )
        
        # 训练PER DQN
        rewards_per, losses_per, epsilons_per = train_agent(env, per_dqn, num_episodes=num_episodes)
        
        # 评估PER DQN
        avg_reward_per, success_rate_per = evaluate_agent(env, per_dqn, num_episodes=100)
        
        # 如果需要与标准DQN对比
        if compare:
            # 创建标准DQN智能体
            standard_dqn = DQN(
                state_dim=3,  # 黑杰克状态维度：玩家点数、庄家明牌、是否有可用A
                action_dim=env.action_space.n,
                learning_rate=0.001,
                gamma=0.99,
                epsilon_start=1.0,
                epsilon_end=0.1,
                epsilon_decay=0.995,
                buffer_size=10000,
                batch_size=64,
                target_update=10
            )
            
            # 训练标准DQN
            rewards_standard, losses_standard, epsilons_standard = train_agent(env, standard_dqn, num_episodes=num_episodes)
            
            # 评估标准DQN
            avg_reward_standard, success_rate_standard = evaluate_agent(env, standard_dqn, num_episodes=100)
            
            # 绘制对比图
            fig, axes = plt.subplots(3, 1, figsize=(12, 15))
            
            # 奖励对比
            axes[0].plot(rewards_standard, alpha=0.3, color='blue', label='标准DQN原始奖励')
            axes[0].plot(rewards_per, alpha=0.3, color='purple', label='PER DQN原始奖励')
            
            # 计算移动平均
            window = 50
            rewards_standard_smoothed = np.convolve(rewards_standard, np.ones(window)/window, mode='valid')
            rewards_per_smoothed = np.convolve(rewards_per, np.ones(window)/window, mode='valid')
            
            axes[0].plot(range(len(rewards_standard_smoothed)), rewards_standard_smoothed, color='blue', label=f'标准DQN ({window}回合平均)')
            axes[0].plot(range(len(rewards_per_smoothed)), rewards_per_smoothed, color='purple', label=f'PER DQN ({window}回合平均)')
            
            axes[0].set_xlabel('回合')
            axes[0].set_ylabel('奖励')
            axes[0].set_title('标准DQN vs PER DQN: 奖励对比')
            axes[0].legend()
            axes[0].grid(True, linestyle='--', alpha=0.6)
            
            # 损失对比
            axes[1].plot(losses_standard, alpha=0.3, color='blue', label='标准DQN原始损失')
            axes[1].plot(losses_per, alpha=0.3, color='purple', label='PER DQN原始损失')
            
            # 计算移动平均
            losses_standard_smoothed = np.convolve(losses_standard, np.ones(window)/window, mode='valid')
            losses_per_smoothed = np.convolve(losses_per, np.ones(window)/window, mode='valid')
            
            axes[1].plot(range(len(losses_standard_smoothed)), losses_standard_smoothed, color='blue', label=f'标准DQN ({window}回合平均)')
            axes[1].plot(range(len(losses_per_smoothed)), losses_per_smoothed, color='purple', label=f'PER DQN ({window}回合平均)')
            
            axes[1].set_xlabel('更新次数')
            axes[1].set_ylabel('损失')
            axes[1].set_title('标准DQN vs PER DQN: 损失对比')
            axes[1].legend()
            axes[1].grid(True, linestyle='--', alpha=0.6)
            
            # 学习速度对比（累积奖励）
            cum_rewards_standard = np.cumsum(rewards_standard)
            cum_rewards_per = np.cumsum(rewards_per)
            
            axes[2].plot(cum_rewards_standard, color='blue', label='标准DQN累积奖励')
            axes[2].plot(cum_rewards_per, color='purple', label='PER DQN累积奖励')
            
            axes[2].set_xlabel('回合')
            axes[2].set_ylabel('累积奖励')
            axes[2].set_title('标准DQN vs PER DQN: 学习速度对比')
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
                st.metric("PER DQN平均奖励", f"{avg_reward_per:.3f}")
                st.metric("PER DQN成功率", f"{success_rate_per*100:.1f}%")
                st.metric("PER DQN最终100回合平均奖励", f"{np.mean(rewards_per[-100:]):.3f}")
            
            # 对比结论
            reward_diff = avg_reward_per - avg_reward_standard
            success_rate_diff = success_rate_per - success_rate_standard
            
            learn_speed_standard = cum_rewards_standard[-1] / len(rewards_standard)
            learn_speed_per = cum_rewards_per[-1] / len(rewards_per)
            speed_improvement = (learn_speed_per - learn_speed_standard) / learn_speed_standard * 100
            
            if reward_diff > 0 and success_rate_diff > 0:
                st.success(f"""
                **实验结论:** 优先经验回放DQN在平均奖励(+{reward_diff:.3f})和成功率(+{success_rate_diff*100:.1f}%)上均优于标准DQN。
                
                学习速度提升了约{speed_improvement:.1f}%。这表明在21点环境中，优先经验回放通过更有效地利用重要经验，
                能够加速学习过程并提高最终性能。
                """)
            elif reward_diff > 0:
                st.info(f"""
                **实验结论:** 优先经验回放DQN在平均奖励上优于标准DQN(+{reward_diff:.3f})，但在成功率上差异不大。
                
                学习速度提升了约{speed_improvement:.1f}%。这表明优先经验回放能够更有效地利用有信息量的经验，
                但在这种相对简单的环境中，最终性能的差异可能不太显著。
                """)
            else:
                st.warning(f"""
                **实验结论:** 在这次实验中，标准DQN表现似乎优于优先经验回放DQN。
                
                这可能是由于多种原因：
                1. α或β参数的选择可能不够理想
                2. 在21点这种环境中，噪声和随机性较大，可能导致TD误差不是好的优先级指标
                3. 需要更多的训练回合来体现优先经验回放的优势
                
                优先经验回放在更复杂的环境和更长的训练过程中通常能展现出更明显的优势。
                """)
            
            # 可视化学习到的策略
            st.subheader("学习到的策略对比")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**标准DQN策略**")
                visualize_q_values(standard_dqn, render_type='policy')
            
            with col2:
                st.write("**PER DQN策略**")
                visualize_q_values(per_dqn, render_type='policy')
            
        else:
            # 只显示PER DQN的结果
            plot_training_results(rewards_per, losses_per, epsilons_per)
            
            st.success(f"""
            **实验结果:**
            - 平均奖励: {avg_reward_per:.3f}
            - 成功率: {success_rate_per*100:.1f}%
            - 最终100回合平均奖励: {np.mean(rewards_per[-100:]):.3f}
            """)
            
            # 可视化学习到的策略
            st.subheader("优先经验回放DQN学习到的策略")
            visualize_q_values(per_dqn, render_type='policy')

# 总结
st.header("优先经验回放总结")

st.markdown("""
### 核心优势
- **提高学习效率**: 通过关注更有信息量的经验，可以加速学习过程
- **样本利用率高**: 不会浪费重要但罕见的经验
- **通用性**: 可以与其他DQN改进方法结合使用

### 实施考虑
- **α参数**: 控制优先级的影响程度，通常设为0.6左右
- **β参数**: 控制重要性采样权重，通常从0.4开始，随训练逐渐增加到1.0
- **数据结构**: SumTree的实现对效率有较大影响

### 代码实现关键点
```python
# 更新时的重要部分
def update(self):
    # 从优先经验回放缓冲区采样
    states, actions, rewards, next_states, dones, indices, weights = self.buffer.sample(self.batch_size)
    
    # 计算当前Q值
    q_values = self.q_network(states).gather(1, actions).squeeze(1)
    
    # 计算目标Q值
    with torch.no_grad():
        next_q_values = self.target_network(next_states).max(1)[0]
        targets = rewards + self.gamma * next_q_values * (1 - dones)
    
    # 计算TD误差
    td_errors = torch.abs(q_values - targets).detach().cpu().numpy()
    
    # 更新优先级
    self.buffer.update_priorities(indices, td_errors)
    
    # 计算带权重的损失
    loss = (weights * nn.MSELoss(reduction='none')(q_values, targets)).mean()
    
    # 梯度下降
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
```
""")

# 返回主页按钮
st.markdown("[返回主页](./)")
