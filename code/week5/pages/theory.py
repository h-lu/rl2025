"""
DQN理论基础页面

提供DQN算法的理论知识和核心概念解释
"""

import streamlit as st
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from utils.visualization import create_streamlit_tabs_for_concepts, configure_matplotlib_fonts

# 确保matplotlib可以显示中文字体
configure_matplotlib_fonts()

def render_theory_page():
    """渲染DQN理论基础页面"""
    st.title(f"{config.PAGE_ICONS['theory']} {config.PAGE_TITLES['theory']}")
    
    st.markdown("""
    ## 什么是深度Q网络 (Deep Q-Network, DQN)？
    
    **深度Q网络 (DQN)** 是由DeepMind团队在2013年提出的一种算法，首次将深度学习与强化学习成功结合起来，实现了从像素直接学习控制策略。
    DQN在多种Atari游戏中取得了超越人类的表现，成为深度强化学习领域的重要里程碑。
    
    DQN结合了Q-learning算法和深度神经网络，解决了传统Q-learning在处理高维状态空间时的局限性。
    """)
    
    # 创建选项卡
    tabs = st.tabs(config.THEORY_SECTIONS)
    
    # Q-learning基础
    with tabs[0]:
        st.markdown("""
        ### Q-learning 基础
        
        Q-learning是一种**无模型 (model-free)** 的**时序差分 (temporal-difference)** 强化学习算法，用于学习在给定状态下执行某个动作的价值。
        
        #### 核心思想
        
        Q-learning通过学习状态-动作值函数 Q(s, a) 来确定在状态s下执行动作a的预期未来奖励。这个函数估计：如果在状态s下执行动作a，并且之后遵循最优策略，能够获得的总回报。
        
        #### Q值更新规则
        
        Q-learning使用以下规则更新Q值：
        
        $Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [r_t + \gamma \max_a Q(s_{t+1}, a) - Q(s_t, a_t)]$
        
        其中：
        - $Q(s_t, a_t)$ 是当前状态-动作对的估计值
        - $\alpha$ 是学习率
        - $r_t$ 是执行动作$a_t$后获得的即时奖励
        - $\gamma$ 是折扣因子，控制未来奖励的重要性
        - $\max_a Q(s_{t+1}, a)$ 是下一个状态的最大Q值
        
        #### 动作选择策略
        
        Q-learning通常采用ε-贪婪策略选择动作：
        - 以概率ε随机选择动作（探索）
        - 以概率1-ε选择Q值最大的动作（利用）
        
        #### Q表格表示
        
        传统的Q-learning使用表格存储每个状态-动作对的Q值。这对于小型问题很有效，但当状态空间很大或连续时(如图像输入)，表格表示就变得不可行。
        """)
        
        # 展示传统Q-learning和DQN的比较
        st.subheader("Q表格表示的局限性")
        col1, col2 = st.columns(2)
        
        with col1:
            # 创建一个简单的Q表格示例
            q_table_data = pd.DataFrame(
                np.random.rand(5, 2) * 10,
                index=[f"状态 {i}" for i in range(5)],
                columns=["向左", "向右"]
            )
            st.write("**Q表格示例 (适用于小型状态空间)**")
            st.dataframe(q_table_data)
            st.caption("简单环境中的Q表格：每行是一个状态，每列是一个动作")
        
        with col2:
            # 创建一个图像示例
            st.write("**高维状态空间示例 (需要DQN)**")
            x = np.linspace(0, 5, 84)
            y = np.linspace(0, 5, 84)
            X, Y = np.meshgrid(x, y)
            Z = np.sin(X) * np.cos(Y)
            
            fig, ax = plt.subplots()
            im = ax.imshow(Z, cmap='viridis')
            plt.title("84x84像素的状态空间")
            plt.axis('off')
            st.pyplot(fig)
            st.caption("一个84x84的图像作为状态，可能有超过2^(84*84)种不同状态")
    
    # 深度Q网络介绍
    with tabs[1]:
        st.markdown("""
        ### 深度Q网络 (DQN) 介绍
        
        DQN是对传统Q-learning的重要扩展，用**深度神经网络**代替Q表格来近似Q函数。这使得DQN能够处理高维状态空间，例如直接从图像像素学习。
        
        #### DQN的核心思想
        
        DQN使用深度神经网络来近似状态-动作值函数Q(s, a)。网络接收状态s作为输入，输出每个可能动作a的Q值估计。
        
        #### 神经网络结构
        
        DQN的典型网络结构包括：
        - 输入层：接收状态（如游戏屏幕像素）
        - 隐藏层：通常包含多个卷积层（处理图像）和全连接层
        - 输出层：为每个可能的动作输出一个Q值
        
        #### 从表格到函数近似
        
        传统Q-learning为每个状态-动作对存储一个独立的Q值，而DQN通过函数近似（神经网络）来概括知识，使得：
        - 能够处理大规模或连续的状态空间
        - 能够泛化到未见过的状态
        - 在相似状态之间共享学习经验
        
        #### DQN的两大创新
        
        DQN算法的两个关键创新使得深度神经网络和Q-learning的结合成为可能：
        1. **经验回放 (Experience Replay)**：存储和重用过去的经验
        2. **目标网络 (Target Network)**：使用单独的网络计算目标值
        
        这两个创新极大地提高了学习的稳定性和效率。
        """)
        
        # 展示DQN网络架构
        st.subheader("DQN神经网络架构")
        
        # 创建简单的DQN架构图
        arch_fig, ax = plt.subplots(figsize=(10, 5))
        ax.axis('off')
        
        # 定义各层的位置
        layer_positions = [0.1, 0.3, 0.5, 0.7, 0.9]
        layer_names = ['输入层\n(状态)', '隐藏层1', '隐藏层2', '输出层\n(Q值)']
        layer_sizes = [4, 64, 64, 2]  # CartPole示例：4维状态空间，2个动作
        
        # 绘制各层神经元
        for i, (pos, name, size) in enumerate(zip(layer_positions, layer_names, layer_sizes)):
            # 只显示部分神经元
            if size > 10:
                show_neurons = 10
                skip_text = f"... (共{size}个神经元)"
            else:
                show_neurons = size
                skip_text = ""
                
            # 绘制神经元
            for j in range(show_neurons):
                y_pos = 0.5 + (j - show_neurons/2) * 0.04
                circle = plt.Circle((pos, y_pos), 0.01, color='blue', fill=True)
                ax.add_patch(circle)
            
            # 添加省略号
            if size > 10:
                ax.text(pos, 0.5 - show_neurons*0.02, skip_text, ha='center', va='center', fontsize=8)
            
            # 添加层名称
            ax.text(pos, 0.2, name, ha='center', va='center')
            
            # 连接前后层
            if i > 0:
                for j in range(min(show_neurons, layer_sizes[i-1])):
                    y_start = 0.5 + (j - min(show_neurons, layer_sizes[i-1])/2) * 0.04
                    for k in range(show_neurons):
                        y_end = 0.5 + (k - show_neurons/2) * 0.04
                        ax.plot([layer_positions[i-1], pos], [y_start, y_end], 'k-', alpha=0.1)
        
        st.pyplot(arch_fig)
        st.caption("DQN神经网络架构示例：输入是状态，输出是每个动作的Q值估计")
    
    # 经验回放
    with tabs[2]:
        st.markdown("""
        ### 经验回放 (Experience Replay)
        
        经验回放是DQN的第一个关键创新，它通过存储和重用过去的经验来提高学习效率和稳定性。
        
        #### 经验回放的工作原理
        
        1. 智能体与环境交互，生成经验元组 $(s_t, a_t, r_t, s_{t+1})$
        2. 将这些经验存储在一个固定大小的**回放缓冲区**中
        3. 训练时，从缓冲区中**随机抽样**一批经验
        4. 使用这批经验来更新神经网络
        
        #### 为什么需要经验回放？
        
        经验回放解决了深度强化学习中的几个关键问题：
        
        1. **打破时序相关性**：连续采样的状态通常高度相关，可能导致学习不稳定。随机抽样打破了这种相关性。
        
        2. **提高数据效率**：每个经验可以被多次使用，提高了样本利用率。
        
        3. **平滑学习分布**：随机抽样产生的批次数据分布更加平稳，减少了训练过程中的波动。
        
        #### 实验观察
        
        经验回放的有效性主要通过实验观察得出。在没有经验回放的情况下，DQN通常无法稳定收敛。DeepMind的实验表明，经验回放是DQN成功的关键因素之一。
        """)
        
        # 展示经验回放的可视化
        st.subheader("经验回放工作流程")
        
        # 创建经验回放示意图
        er_fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')
        
        # 定义坐标
        buffer_x, buffer_y = 0.5, 0.6
        buffer_width, buffer_height = 0.6, 0.3
        
        # 绘制回放缓冲区
        buffer_rect = plt.Rectangle((buffer_x - buffer_width/2, buffer_y - buffer_height/2), 
                                   buffer_width, buffer_height, 
                                   edgecolor='blue', facecolor='lightblue', alpha=0.3)
        ax.add_patch(buffer_rect)
        ax.text(buffer_x, buffer_y, "经验回放缓冲区", ha='center', va='center', fontsize=12)
        
        # 绘制环境和智能体
        env_x, env_y = 0.2, 0.2
        agent_x, agent_y = 0.8, 0.2
        
        env_circle = plt.Circle((env_x, env_y), 0.1, edgecolor='green', facecolor='lightgreen', alpha=0.5)
        agent_circle = plt.Circle((agent_x, agent_y), 0.1, edgecolor='red', facecolor='lightcoral', alpha=0.5)
        
        ax.add_patch(env_circle)
        ax.add_patch(agent_circle)
        
        ax.text(env_x, env_y, "环境", ha='center', va='center')
        ax.text(agent_x, agent_y, "智能体", ha='center', va='center')
        
        # 绘制交互和数据流
        # 智能体到环境：动作
        ax.arrow(agent_x - 0.05, agent_y, env_x - agent_x + 0.15, 0, 
                head_width=0.02, head_length=0.02, fc='black', ec='black', length_includes_head=True)
        ax.text((env_x + agent_x) / 2, agent_y - 0.05, "动作", ha='center', va='center', fontsize=10)
        
        # 环境到智能体：状态、奖励
        ax.arrow(env_x + 0.05, env_y + 0.02, agent_x - env_x - 0.15, 0, 
                head_width=0.02, head_length=0.02, fc='black', ec='black', length_includes_head=True)
        ax.text((env_x + agent_x) / 2, agent_y + 0.05, "状态、奖励", ha='center', va='center', fontsize=10)
        
        # 经验存储到缓冲区
        ax.arrow((env_x + agent_x) / 2, agent_y + 0.1, 0, buffer_y - agent_y - 0.25, 
                head_width=0.02, head_length=0.02, fc='blue', ec='blue', length_includes_head=True)
        ax.text((env_x + agent_x) / 2 - 0.1, (agent_y + buffer_y) / 2, "存储经验\n(s, a, r, s')", 
                ha='right', va='center', fontsize=10, color='blue')
        
        # 从缓冲区随机采样
        ax.arrow(buffer_x + 0.1, buffer_y - buffer_height/2 - 0.02, 0, agent_y + 0.1 - (buffer_y - buffer_height/2), 
                head_width=0.02, head_length=0.02, fc='purple', ec='purple', length_includes_head=True)
        ax.text(buffer_x + 0.2, (agent_y + buffer_y - buffer_height/2) / 2, "随机\n采样", 
                ha='left', va='center', fontsize=10, color='purple')
        
        st.pyplot(er_fig)
        st.caption("经验回放工作流程：智能体与环境交互产生经验，存储到回放缓冲区，然后随机采样用于训练")
    
    # 目标网络
    with tabs[3]:
        st.markdown("""
        ### 目标网络 (Target Network)
        
        目标网络是DQN的第二个关键创新，它通过使用单独的网络计算目标值，提高了训练稳定性。
        
        #### 目标网络的工作原理
        
        1. 维护两个神经网络：
           - **Q网络**：主网络，用于选择动作和估计当前Q值
           - **目标网络**：使用相同架构的独立网络，用于计算目标Q值
        
        2. 目标网络的参数定期从Q网络复制，而不是每次都更新
        
        3. 在计算TD目标时使用目标网络：
        
           $y_j = r_j + \gamma \max_{a'} Q(s_{j+1}, a'; \theta^-)$
           
           其中$\theta^-$是目标网络的参数
        
        #### 为什么需要目标网络？
        
        目标网络解决了深度Q学习中的一个关键问题：**目标值不稳定**。
        
        在传统Q-learning中，当我们更新Q值时，目标包含下一状态的估计值：$r + \gamma \max_a Q(s', a)$。
        
        如果使用同一个不断变化的网络来生成这些目标，会产生几个问题：
        
        1. **移动目标问题**：我们同时在修改预测值和目标值，就像追逐一个不断移动的目标
        
        2. **高度相关性**：预测和目标使用相同的参数，导致高度相关的更新
        
        3. **训练不稳定**：这种相关性和不断变化的目标可能导致参数震荡或发散
        
        目标网络通过提供一个相对稳定的目标，解决了这些问题。
        
        #### 也是经验总结
        
        目标网络的必要性也主要是通过实验观察得出的。研究表明，没有目标网络的DQN通常会出现训练不稳定或Q值过高估计的问题。
        """)
        
        # 展示目标网络的可视化
        st.subheader("目标网络与Q网络")
        
        # 创建目标网络与Q网络示意图
        tn_fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')
        
        # 定义坐标
        q_net_x, q_net_y = 0.3, 0.5
        target_net_x, target_net_y = 0.7, 0.5
        q_net_width, q_net_height = 0.25, 0.4
        
        # 绘制Q网络
        q_net_rect = plt.Rectangle((q_net_x - q_net_width/2, q_net_y - q_net_height/2), 
                                 q_net_width, q_net_height, 
                                 edgecolor='blue', facecolor='lightblue', alpha=0.5)
        ax.add_patch(q_net_rect)
        ax.text(q_net_x, q_net_y, "Q网络\n(频繁更新)", ha='center', va='center', fontsize=12)
        
        # 绘制目标网络
        target_net_rect = plt.Rectangle((target_net_x - q_net_width/2, target_net_y - q_net_height/2), 
                                      q_net_width, q_net_height, 
                                      edgecolor='green', facecolor='lightgreen', alpha=0.5)
        ax.add_patch(target_net_rect)
        ax.text(target_net_x, target_net_y, "目标网络\n(定期更新)", ha='center', va='center', fontsize=12)
        
        # 参数复制箭头
        ax.arrow(q_net_x + q_net_width/2 + 0.02, q_net_y, 
                target_net_x - q_net_x - q_net_width - 0.04, 0, 
                head_width=0.02, head_length=0.02, fc='red', ec='red', length_includes_head=True)
        ax.text((q_net_x + target_net_x) / 2, q_net_y + 0.05, 
                "每C步复制参数\nθ- ← θ", ha='center', va='center', fontsize=10, color='red')
        
        # 添加训练流程
        # 经验输入
        ax.arrow(0.1, 0.7, q_net_x - q_net_width/2 - 0.1, 0, 
                head_width=0.02, head_length=0.02, fc='black', ec='black', length_includes_head=True)
        ax.text(0.1, 0.75, "样本批次\n(s, a, r, s')", ha='center', va='center', fontsize=10)
        
        # Q网络计算当前Q值
        ax.arrow(q_net_x, q_net_y - q_net_height/2 - 0.02, 0, -0.1, 
                head_width=0.02, head_length=0.02, fc='blue', ec='blue', length_includes_head=True)
        ax.text(q_net_x - 0.15, q_net_y - q_net_height/2 - 0.08, 
                "当前Q值\nQ(s, a; θ)", ha='center', va='center', fontsize=10, color='blue')
        
        # 目标网络计算目标Q值
        ax.arrow(target_net_x, target_net_y - q_net_height/2 - 0.02, 0, -0.1, 
                head_width=0.02, head_length=0.02, fc='green', ec='green', length_includes_head=True)
        ax.text(target_net_x + 0.15, target_net_y - q_net_height/2 - 0.08, 
                "目标Q值\nmax Q(s', a'; θ-)", ha='center', va='center', fontsize=10, color='green')
        
        # 计算损失
        loss_x, loss_y = 0.5, 0.2
        ax.text(loss_x, loss_y, "计算损失: (r + γ·max Q(s', a'; θ-) - Q(s, a; θ))²", 
                ha='center', va='center', fontsize=11, bbox=dict(facecolor='white', alpha=0.7))
        
        # 反向传播更新Q网络
        ax.arrow(loss_x - 0.05, loss_y + 0.02, q_net_x - loss_x, q_net_y - q_net_height/2 - loss_y - 0.03, 
                head_width=0.02, head_length=0.02, fc='purple', ec='purple', length_includes_head=True, ls='--')
        ax.text((loss_x + q_net_x)/2 - 0.1, (loss_y + q_net_y - q_net_height/2)/2, 
                "反向传播\n更新θ", ha='center', va='center', fontsize=10, color='purple')
        
        st.pyplot(tn_fig)
        st.caption("目标网络与Q网络工作流程：Q网络频繁更新，而目标网络仅定期从Q网络复制参数")
    
    # DQN算法流程
    with tabs[4]:
        st.markdown("""
        ### DQN算法流程
        
        下面是DQN算法的完整流程，汇集了我们讨论的所有核心概念：
        """)
        
        # 显示DQN伪代码
        st.code(config.DQN_PSEUDOCODE, language=None)
        
        st.markdown("""
        #### 算法详解
        
        1. **初始化阶段**：
           - 创建Q网络和目标网络，初始时参数相同
           - 创建经验回放缓冲区
        
        2. **交互阶段**：
           - 根据当前状态和ε-贪婪策略选择动作
           - 执行动作，观察奖励和下一个状态
           - 将经验存储到回放缓冲区
        
        3. **学习阶段**：
           - 从回放缓冲区随机抽样经验批次
           - 使用目标网络计算目标Q值
           - 使用Q网络计算当前Q值
           - 计算损失并更新Q网络参数
           - 定期将Q网络参数复制到目标网络
        
        #### DQN的特点
        
        DQN算法结合了以下几个关键特点：
        
        1. **深度神经网络作为函数逼近器**：能够处理高维状态空间
        2. **经验回放**：打破数据相关性，提高样本效率
        3. **目标网络**：稳定训练过程，减少参数振荡
        4. **ε-贪婪策略**：平衡探索与利用
        
        DQN算法通过这些创新解决了之前在复杂环境中应用强化学习的关键障碍，开创了深度强化学习的新时代。
        """)
        
        # 添加DQN算法流程图
        st.subheader("DQN算法流程图")
        
        # 创建DQN算法流程图
        algo_fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('off')
        
        # 坐标和尺寸设置
        box_width, box_height = 0.2, 0.1
        
        # 初始化框
        init_x, init_y = 0.5, 0.9
        init_box = plt.Rectangle((init_x - box_width/2, init_y - box_height/2), 
                                box_width, box_height, 
                                edgecolor='black', facecolor='lightgray', alpha=0.7)
        ax.add_patch(init_box)
        ax.text(init_x, init_y, "初始化网络和缓冲区", ha='center', va='center', fontsize=10)
        
        # 重置环境框
        reset_x, reset_y = 0.5, 0.8
        reset_box = plt.Rectangle((reset_x - box_width/2, reset_y - box_height/2), 
                                box_width, box_height, 
                                edgecolor='black', facecolor='lightgreen', alpha=0.7)
        ax.add_patch(reset_box)
        ax.text(reset_x, reset_y, "重置环境", ha='center', va='center', fontsize=10)
        
        # 选择动作框
        action_x, action_y = 0.5, 0.7
        action_box = plt.Rectangle((action_x - box_width/2, action_y - box_height/2), 
                                 box_width, box_height, 
                                 edgecolor='black', facecolor='lightblue', alpha=0.7)
        ax.add_patch(action_box)
        ax.text(action_x, action_y, "ε-贪婪选择动作", ha='center', va='center', fontsize=10)
        
        # 执行动作框
        execute_x, execute_y = 0.5, 0.6
        execute_box = plt.Rectangle((execute_x - box_width/2, execute_y - box_height/2), 
                                  box_width, box_height, 
                                  edgecolor='black', facecolor='lightblue', alpha=0.7)
        ax.add_patch(execute_box)
        ax.text(execute_x, execute_y, "执行动作\n观察奖励和下一状态", ha='center', va='center', fontsize=10)
        
        # 存储经验框
        store_x, store_y = 0.5, 0.5
        store_box = plt.Rectangle((store_x - box_width/2, store_y - box_height/2), 
                                box_width, box_height, 
                                edgecolor='black', facecolor='lightyellow', alpha=0.7)
        ax.add_patch(store_box)
        ax.text(store_x, store_y, "存储经验到缓冲区", ha='center', va='center', fontsize=10)
        
        # 采样经验框
        sample_x, sample_y = 0.5, 0.4
        sample_box = plt.Rectangle((sample_x - box_width/2, sample_y - box_height/2), 
                                 box_width, box_height, 
                                 edgecolor='black', facecolor='lightyellow', alpha=0.7)
        ax.add_patch(sample_box)
        ax.text(sample_x, sample_y, "从缓冲区采样批次", ha='center', va='center', fontsize=10)
        
        # 计算目标框
        target_x, target_y = 0.5, 0.3
        target_box = plt.Rectangle((target_x - box_width/2, target_y - box_height/2), 
                                 box_width, box_height, 
                                 edgecolor='black', facecolor='lightcoral', alpha=0.7)
        ax.add_patch(target_box)
        ax.text(target_x, target_y, "计算目标Q值\n(使用目标网络)", ha='center', va='center', fontsize=10)
        
        # 更新Q网络框
        update_x, update_y = 0.5, 0.2
        update_box = plt.Rectangle((update_x - box_width/2, update_y - box_height/2), 
                                 box_width, box_height, 
                                 edgecolor='black', facecolor='lightcoral', alpha=0.7)
        ax.add_patch(update_box)
        ax.text(update_x, update_y, "更新Q网络", ha='center', va='center', fontsize=10)
        
        # 更新目标网络框
        update_target_x, update_target_y = 0.5, 0.1
        update_target_box = plt.Rectangle((update_target_x - box_width/2, update_target_y - box_height/2), 
                                        box_width, box_height, 
                                        edgecolor='black', facecolor='lightcoral', alpha=0.7)
        ax.add_patch(update_target_box)
        ax.text(update_target_x, update_target_y, "定期更新目标网络", ha='center', va='center', fontsize=10)
        
        # 连接箭头
        arrow_coords = [
            (init_x, init_y - box_height/2, 0, -0.05),  # 初始化到重置环境
            (reset_x, reset_y - box_height/2, 0, -0.05),  # 重置环境到选择动作
            (action_x, action_y - box_height/2, 0, -0.05),  # 选择动作到执行动作
            (execute_x, execute_y - box_height/2, 0, -0.05),  # 执行动作到存储经验
            (store_x, store_y - box_height/2, 0, -0.05),  # 存储经验到采样经验
            (sample_x, sample_y - box_height/2, 0, -0.05),  # 采样经验到计算目标
            (target_x, target_y - box_height/2, 0, -0.05),  # 计算目标到更新Q网络
            (update_x, update_y - box_height/2, 0, -0.05),  # 更新Q网络到更新目标网络
            (update_target_x + box_width/2 + 0.05, update_target_y, 0.3, 0),  # 横向连接
            (update_target_x + box_width/2 + 0.35, update_target_y, 0, action_y - update_target_y),  # 向上连接
            (update_target_x + box_width/2 + 0.35, action_y, -(update_target_x + box_width/2 + 0.35 - action_x - box_width/2), 0)  # 连回选择动作
        ]
        
        for x, y, dx, dy in arrow_coords:
            ax.arrow(x, y, dx, dy, head_width=0.02, head_length=0.02, fc='black', ec='black', length_includes_head=True)
        
        # 添加终止检测（虚线）
        ax.plot([execute_x + box_width/2, execute_x + box_width/2 + 0.15], 
               [execute_y, execute_y], 'k--')
        ax.plot([execute_x + box_width/2 + 0.15, execute_x + box_width/2 + 0.15], 
               [execute_y, reset_y], 'k--')
        ax.plot([execute_x + box_width/2 + 0.15, reset_x - box_width/2], 
               [reset_y, reset_y], 'k--')
        ax.text(execute_x + box_width/2 + 0.2, (execute_y + reset_y)/2, 
               "如果回合结束", ha='center', va='center', fontsize=8, rotation=90)
        
        st.pyplot(algo_fig)
        st.caption("DQN算法完整流程")
    
    # 显示核心概念选项卡
    st.subheader("DQN核心概念")
    create_streamlit_tabs_for_concepts() 