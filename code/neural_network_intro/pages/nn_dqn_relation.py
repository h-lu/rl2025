import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils.data_utils import create_dqn_example_data

def show_nn_dqn_relation():
    """显示神经网络与DQN关系页面"""
    st.title("神经网络与DQN的关系")
    
    st.markdown("""
    ## 深度Q网络 (DQN) 简介
    
    **深度Q网络 (DQN)** 是将深度学习与强化学习相结合的典型算法，由DeepMind团队于2013年提出，并因在Atari游戏上取得的突破性成果而闻名。
    
    DQN的核心思想是使用神经网络来近似Q函数，从而实现大规模状态空间下的强化学习。
    """)
    
    # DQN与传统Q学习的对比
    st.markdown("""
    ## DQN与传统Q学习的对比
    
    ### 传统Q学习
    
    传统的Q学习使用表格来存储和更新每个状态-动作对的Q值：
    
    $$Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$$
    
    **局限性**：
    - 只能处理离散且有限的状态空间
    - 无法处理高维输入（如图像）
    - 无法泛化到未见过的状态
    
    ### 深度Q网络 (DQN)
    
    DQN使用神经网络来近似Q函数：
    
    $$Q(s, a; \theta) \approx Q^*(s, a)$$
    
    其中 $\theta$ 是神经网络的参数。
    
    **优势**：
    - 可以处理连续或高维状态空间
    - 能从原始感知数据中学习（如像素）
    - 可以泛化到类似但未见过的状态
    """)
    
    # 神经网络在DQN中的作用
    st.markdown("""
    ## 神经网络在DQN中的作用
    
    在DQN中，神经网络扮演着**函数近似器**的角色，具体作用包括：
    
    1. **状态表示学习**：从原始输入（如像素）中提取有用特征
    2. **价值函数近似**：预测每个动作的期望累积奖励（Q值）
    3. **泛化能力**：将学到的知识应用到相似但未见过的状态
    4. **端到端学习**：直接从原始输入到动作决策的映射
    """)
    
    # DQN的网络架构
    st.markdown("""
    ## DQN的网络架构
    
    DQN的神经网络架构根据输入类型而有所不同：
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 处理图像输入的DQN
        
        **典型架构**：
        1. **卷积层**：提取空间特征
        2. **全连接层**：整合特征
        3. **输出层**：每个动作的Q值
        
        **例如**：Atari游戏DQN使用：
        - 3个卷积层
        - 2个全连接层
        - 输出层神经元数等于动作数
        """)
        
        # 图像输入的DQN结构示意图
        fig, ax = plt.subplots(figsize=(7, 5))
        
        # 绘制网络层
        layers = ['图像输入', 'Conv1', 'Conv2', 'Conv3', 'FC1', 'FC2', 'Q值输出']
        layer_widths = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
        layer_heights = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
        layer_positions = [1, 2, 3, 4, 5, 6, 7]
        layer_colors = ['skyblue', 'salmon', 'salmon', 'salmon', 'lightgreen', 'lightgreen', 'gold']
        
        for i, (layer, width, height, pos, color) in enumerate(zip(
            layers, layer_widths, layer_heights, layer_positions, layer_colors)):
            
            rect = plt.Rectangle((pos - width/2, 0), width, height, color=color, alpha=0.8)
            ax.add_patch(rect)
            ax.text(pos, height + 0.1, layer, ha='center', va='center', fontsize=10)
            
            # 添加连接线
            if i > 0:
                prev_pos = layer_positions[i-1]
                prev_width = layer_widths[i-1]
                prev_height = layer_heights[i-1]
                
                ax.plot([prev_pos + prev_width/2, pos - width/2], 
                        [prev_height/2, height/2], 'k-', alpha=0.5)
        
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 1.2)
        ax.axis('off')
        st.pyplot(fig)
    
    with col2:
        st.markdown("""
        ### 处理向量输入的DQN
        
        **典型架构**：
        1. **全连接层**：直接处理向量输入
        2. **多个隐藏层**：提取特征
        3. **输出层**：每个动作的Q值
        
        **例如**：处理传感器数据的DQN：
        - 输入层：传感器读数
        - 2-3个隐藏层
        - 输出层：离散动作的Q值
        """)
        
        # 向量输入的DQN结构示意图
        fig, ax = plt.subplots(figsize=(7, 5))
        
        # 绘制网络层
        layers = ['状态向量', '隐藏层1', '隐藏层2', 'Q值输出']
        layer_widths = [0.8, 0.6, 0.4, 0.2]
        layer_heights = [0.8, 0.6, 0.4, 0.2]
        layer_positions = [1.5, 3, 4.5, 6]
        layer_colors = ['skyblue', 'lightgreen', 'lightgreen', 'gold']
        
        for i, (layer, width, height, pos, color) in enumerate(zip(
            layers, layer_widths, layer_heights, layer_positions, layer_colors)):
            
            rect = plt.Rectangle((pos - width/2, 0), width, height, color=color, alpha=0.8)
            ax.add_patch(rect)
            ax.text(pos, height + 0.1, layer, ha='center', va='center', fontsize=10)
            
            # 添加连接线
            if i > 0:
                prev_pos = layer_positions[i-1]
                prev_width = layer_widths[i-1]
                prev_height = layer_heights[i-1]
                
                ax.plot([prev_pos + prev_width/2, pos - width/2], 
                        [prev_height/2, height/2], 'k-', alpha=0.5)
        
        ax.set_xlim(0, 7.5)
        ax.set_ylim(0, 1.2)
        ax.axis('off')
        st.pyplot(fig)
    
    # DQN训练过程
    st.markdown("""
    ## DQN的训练过程
    
    DQN的训练过程结合了深度学习与强化学习的技术，具有一些独特特点：
    
    ### 1. 损失函数
    
    DQN使用**时序差分误差**作为损失函数：
    
    $$L(\theta) = \mathbb{E}_{(s,a,r,s')} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]$$
    
    其中：
    - $\theta$ 是当前网络参数
    - $\theta^-$ 是目标网络参数
    - $\gamma$ 是折扣因子
    """)
    
    # DQN训练的关键技术
    st.markdown("""
    ### 2. DQN的关键稳定性技术
    
    DQN引入了两项关键技术来解决结合神经网络与Q学习的不稳定性：
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 经验回放 (Experience Replay)
        
        **核心思想**：存储和重用过去的经验
        
        **实现方式**：
        - 维护一个经验缓冲区
        - 存储四元组 $(s, a, r, s')$
        - 随机采样小批量进行训练
        
        **优势**：
        - 打破样本相关性
        - 提高数据利用效率
        - 减少方差和不稳定性
        """)
    
    with col2:
        st.markdown("""
        #### 目标网络 (Target Network)
        
        **核心思想**：使用单独的网络生成训练目标
        
        **实现方式**：
        - 维护两个结构相同的网络
        - 主网络用于选择动作和更新
        - 目标网络用于计算目标Q值
        - 周期性地更新目标网络
        
        **优势**：
        - 提供稳定的训练目标
        - 减少参数更新的振荡
        - 防止发散
        """)
    
    # DQN算法流程
    st.markdown("""
    ### 3. DQN算法流程
    
    DQN的训练过程可概括为以下步骤：
    """)
    
    # 创建DQN训练流程图
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 定义步骤和位置
    steps = [
        "初始化网络参数",
        "与环境交互\n收集经验",
        "存储经验\n到回放缓冲区",
        "从缓冲区\n采样小批量",
        "计算目标Q值\n和损失函数",
        "更新主网络参数",
        "周期性更新\n目标网络"
    ]
    
    positions = [
        (1, 6),
        (3, 6),
        (5, 6),
        (5, 4),
        (5, 2),
        (3, 2),
        (1, 2)
    ]
    
    # 绘制步骤框
    for i, (step, pos) in enumerate(zip(steps, positions)):
        rect = plt.Rectangle((pos[0] - 0.8, pos[1] - 0.6), 1.6, 1.2, 
                           color='lightblue', alpha=0.8, ec='blue')
        ax.add_patch(rect)
        ax.text(pos[0], pos[1], step, ha='center', va='center', fontsize=10)
    
    # 添加箭头连接
    arrows = [
        (positions[0], positions[1]),
        (positions[1], positions[2]),
        (positions[2], positions[3]),
        (positions[3], positions[4]),
        (positions[4], positions[5]),
        (positions[5], positions[1]),
        (positions[5], positions[6]),
        (positions[6], positions[0], 'dotted')
    ]
    
    for i, arrow in enumerate(arrows):
        if len(arrow) == 2:
            start, end = arrow
            style = '-'
        else:
            start, end, style = arrow
        
        if start[0] == end[0]:  # 垂直箭头
            ax.annotate("", xy=(end[0], end[1] + 0.6), xytext=(start[0], start[1] - 0.6),
                      arrowprops=dict(arrowstyle="->", linestyle=style))
        elif start[1] == end[1]:  # 水平箭头
            ax.annotate("", xy=(end[0] - 0.8, end[1]), xytext=(start[0] + 0.8, start[1]),
                      arrowprops=dict(arrowstyle="->", linestyle=style))
        else:  # 斜向箭头
            ax.annotate("", xy=(end[0], end[1]), xytext=(start[0], start[1]),
                      arrowprops=dict(arrowstyle="->", linestyle=style))
    
    # 添加说明
    ax.text(3, 4, "训练循环", ha='center', va='center', fontsize=12, 
           bbox=dict(facecolor='lightyellow', alpha=0.8, boxstyle='round'))
    
    ax.text(6.5, 6, "经验回放缓冲区", ha='center', va='center', fontsize=12, 
           bbox=dict(facecolor='lightgreen', alpha=0.8, boxstyle='round'))
    
    ax.text(0.5, 4, "目标网络\n(稳定目标)", ha='center', va='center', fontsize=12, 
           bbox=dict(facecolor='lightcoral', alpha=0.8, boxstyle='round'))
    
    # 设置图表
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 8)
    ax.set_title("DQN训练流程")
    ax.axis('off')
    
    st.pyplot(fig)
    
    # DQN计算示例
    st.markdown("""
    ## DQN计算示例
    
    让我们通过一个简单的例子来理解神经网络如何在DQN中用于Q值更新：
    """)
    
    # 获取DQN示例数据
    dqn_data = create_dqn_example_data()
    
    # 创建一个交互式示例计算
    st.markdown("### 选择一个状态-动作对查看Q值更新计算")
    
    state_idx = st.selectbox("选择状态", [0, 1, 2, 3], format_func=lambda x: f"状态 {x+1}")
    
    # 获取所选状态的信息
    state = dqn_data['states'][state_idx]
    action = dqn_data['actions'][state_idx]
    reward = dqn_data['rewards'][state_idx]
    next_state = dqn_data['next_states'][state_idx]
    q_values = dqn_data['q_values'][state_idx]
    target_q = dqn_data['target_q'][state_idx]
    
    # 显示计算细节
    st.markdown(f"""
    ### 状态 {state_idx+1} 的计算详情
    
    **当前状态**: {state}
    
    **选择的动作**: {action}
    
    **获得的奖励**: {reward}
    
    **下一状态**: {next_state}
    
    #### Q值计算
    
    **当前Q值** (由神经网络预测):
    - 动作0: {q_values[0]:.4f}
    - 动作1: {q_values[1]:.4f}
    
    **下一状态的最大Q值**: 计算为 {np.max(dqn_data['q_values'][state_idx]):.4f}
    
    **目标Q值** (使用贝尔曼方程计算):
    $Q_{{target}}(s, a) = r + \gamma \cdot \max_{{a'}} Q(s', a')$
    
    对于动作 {action}:
    $Q_{{target}}(s_{{{state_idx+1}}}, {action}) = {reward} + 0.9 \cdot {np.max(dqn_data['q_values'][state_idx]):.4f} = {target_q[action]:.4f}$
    
    **损失计算**:
    $Loss = (Q_{{target}}(s, a) - Q(s, a))^2 = ({target_q[action]:.4f} - {q_values[action]:.4f})^2 = {(target_q[action] - q_values[action])**2:.4f}$
    
    神经网络的参数将通过反向传播和梯度下降根据这个损失进行更新。
    """)
    
    # 可视化Q值预测与目标
    st.markdown("### Q值预测与目标的可视化")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bar_width = 0.35
    x = np.arange(2)  # 两个动作
    
    # 绘制当前Q值和目标Q值
    ax.bar(x - bar_width/2, q_values, bar_width, label='当前Q值预测')
    ax.bar(x + bar_width/2, target_q, bar_width, label='目标Q值')
    
    # 高亮选择的动作
    ax.bar(action - bar_width/2, q_values[action], bar_width, color='red', alpha=0.7)
    ax.bar(action + bar_width/2, target_q[action], bar_width, color='darkred', alpha=0.7)
    
    # 添加文本标签
    for i, v in enumerate(q_values):
        ax.text(i - bar_width/2, v + 0.05, f"{v:.2f}", ha='center', va='bottom')
    
    for i, v in enumerate(target_q):
        ax.text(i + bar_width/2, v + 0.05, f"{v:.2f}", ha='center', va='bottom')
    
    # 设置图表
    ax.set_xticks(x)
    ax.set_xticklabels(['动作 0', '动作 1'])
    ax.set_ylabel('Q值')
    ax.set_title(f'状态 {state_idx+1} 的Q值对比')
    ax.legend()
    
    st.pyplot(fig)
    
    # 神经网络训练对DQN的挑战
    st.markdown("""
    ## 神经网络训练对DQN的挑战
    
    将神经网络应用于DQN带来了一系列独特的挑战：
    
    ### 1. 目标不稳定性
    
    **问题**：Q学习的训练目标依赖于当前的估计值，导致"自举"问题。
    
    **解决方案**：
    - 使用目标网络
    - 调整更新频率
    - 软更新（DDQN、DDPG等算法）
    
    ### 2. 样本相关性
    
    **问题**：连续收集的样本高度相关，违背了SGD的独立同分布假设。
    
    **解决方案**：
    - 经验回放
    - 优先级经验回放
    - 多步学习
    
    ### 3. 探索与利用
    
    **问题**：神经网络倾向于迅速过度利用当前策略。
    
    **解决方案**：
    - ε-贪婪策略
    - 玻尔兹曼探索
    - 内在奖励/好奇心驱动探索
    """)
    
    # DQN的变体和改进
    with st.expander("DQN的变体和改进"):
        st.markdown("""
        ### DQN的主要改进变体
        
        随着研究的深入，DQN有了许多改进变体：
        
        1. **Double DQN (DDQN)**
           - 解决Q值高估问题
           - 使用主网络选择动作，目标网络评估动作
        
        2. **Dueling DQN**
           - 将Q值分解为状态价值和优势函数
           - 更有效地评估状态价值
        
        3. **Prioritized Experience Replay (PER)**
           - 根据TD误差大小为经验分配优先级
           - 更频繁地重放有学习价值的样本
        
        4. **Rainbow DQN**
           - 结合多种DQN改进
           - 包括DDQN、Dueling、PER、多步学习等
        """)
    
    # 神经网络设计对DQN的影响
    st.markdown("""
    ## 神经网络设计对DQN的影响
    
    神经网络的设计选择对DQN性能有重要影响：
    
    ### 1. 网络架构选择
    
    - **卷积网络 (CNN)**：适用于图像输入，能学习空间特征
    - **全连接网络 (MLP)**：适用于低维状态向量
    - **循环网络 (RNN/LSTM)**：适用于部分可观测问题或序列决策
    
    ### 2. 关键超参数影响
    
    | 超参数 | 影响 | 建议设置 |
    |--------|------|----------|
    | 学习率 | 收敛速度和稳定性 | 0.0001-0.001，可使用学习率调度 |
    | 批量大小 | 训练稳定性和效率 | 32-128，取决于内存和计算资源 |
    | 网络大小 | 表达能力和过拟合风险 | 根据问题复杂度调整，通常2-3层 |
    | 目标网络更新频率 | 学习稳定性 | 每100-10000步，取决于问题 |
    | 折扣因子 | 时间视野 | 0.9-0.99，长期任务用较大值 |
    """)
    
    # 实现DQN的最佳实践
    st.markdown("""
    ## 实现DQN的最佳实践
    
    基于前面学习的神经网络知识，以下是实现DQN的一些最佳实践：
    
    1. **网络初始化**：使用合适的权重初始化方法（如He初始化）
    2. **批归一化**：在隐藏层间添加批归一化层，加速训练
    3. **学习率调度**：随训练进程逐渐降低学习率
    4. **Dropout**：在全连接层间使用Dropout防止过拟合
    5. **梯度裁剪**：限制梯度范数，防止梯度爆炸
    6. **预训练**：在简化环境中预训练网络，再迁移到复杂环境
    7. **模型集成**：使用多个网络的平均预测，提高稳定性
    """)
    
    # 基础练习
    st.markdown("## 基础练习")
    
    st.markdown("""
    1. 计算一个给定状态-动作对的TD误差和目标Q值。
    2. 分析不同网络架构对DQN性能的潜在影响。
    3. 思考如何将本课程学到的神经网络训练技巧应用于DQN实现。
    """)
    
    # 侧边栏资源
    st.sidebar.markdown("""
    ### 深入学习资源
    
    - 原始DQN论文：[Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
    - 改进版DQN：[Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
    - OpenAI Spinning Up：[DQN 实现](https://spinningup.openai.com/)
    - Stable Baselines3：[DQN 文档](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html)
    """)
    
    # 下一章预告
    st.markdown("""
    ## 下一章预告
    
    在下一章，我们将通过**练习环节**巩固所学知识，实践神经网络和DQN的相关概念。
    """) 