import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from utils import create_q_table_df
import time

def single_color_heatmap(df, title="Q值热力图", color_scale="Blues"):
    """创建单一颜色由浅到深的热力图"""
    fig = px.imshow(
        df.iloc[:, 1:].values,
        x=df.columns[1:],
        y=df['状态'],
        color_continuous_scale=color_scale,
        labels=dict(x="动作", y="状态", color="Q值"),
        title=title
    )
    return fig

def render_q_table_basics():
    st.subheader("Q表是什么？")
    
    # 思维链：首先解释Q表的基本概念
    st.markdown("""
    ### 第一步：理解Q表的基本结构
    
    Q表本质上是一个二维表格：
    - **行**：代表环境中的不同**状态**
    - **列**：代表智能体可以采取的不同**动作**
    - **单元格值**：代表在特定状态下采取特定动作的预期累积奖励（Q值）
    
    下面我们一起构建一个简单的Q表：
    """)
    
    # 交互式创建一个简单的Q表
    col1, col2 = st.columns(2)
    
    with col1:
        num_states = st.slider("选择状态数量", 2, 5, 3)
        states = [f"状态{i}" for i in range(1, num_states+1)]
    
    with col2:
        num_actions = st.slider("选择动作数量", 2, 4, 3)
        actions = [f"动作{i}" for i in range(1, num_actions+1)]
    
    # 初始化Q表
    st.markdown("### 第二步：观察初始Q表")
    st.markdown("""
    Q-Learning算法开始时，Q表通常被初始化为全0或小的随机值。这表示智能体对环境一无所知，需要通过交互学习。
    """)
    
    init_zeros = st.checkbox("初始化为0", value=True)
    
    q_table = {}
    if init_zeros:
        # 初始化为0
        for state in states:
            for action in actions:
                q_table[(state, action)] = 0.0
    else:
        # 初始化为小的随机值
        for state in states:
            for action in actions:
                q_table[(state, action)] = np.random.uniform(-0.1, 0.1)
    
    # 显示初始Q表
    q_df = create_q_table_df(states, actions, q_table)
    st.dataframe(q_df, use_container_width=True)
    
    # 思维链：解释如何从Q表中提取策略
    st.markdown("""
    ### 第三步：理解如何从Q表中提取策略
    
    Q表的目的是指导智能体的行为。在每个状态下，智能体应该选择Q值最大的动作，这就是"利用"。
    
    **最优策略**是一个从状态到动作的映射，指导智能体在每个状态下选择哪个动作。
    """)
    
    # 显示添加了一些值的Q表
    st.markdown("让我们给Q表添加一些值，然后观察如何从中提取策略：")
    
    # 添加一些演示值
    demo_q_table = q_table.copy()
    demo_q_table[(states[0], actions[1])] = 5.0  # 状态1的最佳动作是动作2
    if len(states) > 1:
        demo_q_table[(states[1], actions[0])] = 3.0  # 状态2的最佳动作是动作1
    if len(states) > 2:
        demo_q_table[(states[2], actions[2])] = 7.0  # 状态3的最佳动作是动作3
    
    # 显示更新后的Q表
    demo_q_df = create_q_table_df(states, actions, demo_q_table)
    st.dataframe(demo_q_df, use_container_width=True)
    
    # 提取并显示策略
    policy_data = []
    for state in states:
        max_q = max([demo_q_table.get((state, action), 0.0) for action in actions])
        best_action = max(actions, key=lambda a: demo_q_table.get((state, a), 0.0))
        policy_data.append({
            '状态': state,
            '最优动作': best_action,
            '对应Q值': max_q
        })
    
    policy_df = pd.DataFrame(policy_data)
    st.markdown("#### 从Q表提取的最优策略：")
    st.table(policy_df)
    
    # 热力图可视化
    st.markdown("### 第四步：通过热力图可视化Q表")
    st.markdown("""
    热力图是理解Q表的强大工具，颜色深浅直观地表示Q值的大小。颜色越深，Q值越大，表示该状态-动作对越有价值。
    """)
    
    fig = single_color_heatmap(demo_q_df, "Q值分布热力图")
    st.plotly_chart(fig, use_container_width=True)

# 添加一个Q-learning算法函数
def run_q_learning(max_iterations=10000, alpha=0.1, gamma=0.9, epsilon=0.1, convergence_threshold=0.001, updates_per_iteration=1):
    """
    执行Q-learning算法并返回学习过程和收敛信息
    
    参数:
    - max_iterations: 最大迭代次数
    - alpha: 学习率
    - gamma: 折扣因子
    - epsilon: 探索率
    - convergence_threshold: Q值收敛阈值
    - updates_per_iteration: 每次迭代更新的状态-动作对数量
    
    返回:
    - q_history: 每次迭代后的Q表历史
    - convergence_iteration: 收敛迭代次数
    - policy_history: 策略历史
    - q_change_history: Q值变化历史
    """
    # 定义状态和动作
    states = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'G']
    actions = ['上', '右', '下', '左']
    
    # 初始化Q表
    q_table = {}
    for state in states:
        if state == 'S5':  # 障碍物
            for action in actions:
                q_table[(state, action)] = float('nan')
        elif state == 'G':  # 终点
            for action in actions:
                q_table[(state, action)] = 0.0
        else:
            for action in actions:
                q_table[(state, action)] = 0.0
    
    # 定义网格世界映射 - 从状态ID转换为(行,列)坐标
    grid_map = {
        'S0': (0, 0), 'S1': (0, 1), 'S2': (0, 2), 'S3': (0, 3),
        'S4': (1, 0), 'S5': (1, 1), 'S6': (1, 2), 'S7': (1, 3),
        'S8': (2, 0), 'S9': (2, 1), 'S10': (2, 2), 'S11': (2, 3),
        'S12': (3, 0), 'S13': (3, 1), 'S14': (3, 2), 'G': (3, 3)
    }
    
    # 反向映射 - 从(行,列)坐标转换为状态ID
    inverse_grid_map = {v: k for k, v in grid_map.items()}
    
    # 定义动作到方向的映射
    action_map = {
        '上': (-1, 0),
        '右': (0, 1),
        '下': (1, 0),
        '左': (0, -1)
    }
    
    # 记录Q表历史和策略历史
    q_history = [q_table.copy()]
    policy_history = []
    
    # 新增：记录Q值变化历史
    q_change_history = []
    
    # 用于判断策略收敛
    previous_policy = {}
    stable_policy_count = 0
    convergence_iteration = max_iterations
    
    # 开始迭代
    for iteration in range(1, max_iterations + 1):
        # 记录这次迭代中的最大Q值变化
        max_q_change = 0
        
        # 进行指定数量的Q值更新
        for update in range(updates_per_iteration):
            # 选择起始状态（随机选择一个非障碍物、非终点的状态）
            valid_start_states = [s for s in states if s != 'S5' and s != 'G']
            state = np.random.choice(valid_start_states)
            
            # 当前状态坐标
            current_row, current_col = grid_map[state]
            
            # ε-贪婪策略选择动作
            if np.random.random() < epsilon:
                # 探索：随机选择动作
                valid_actions = [a for a in actions if not pd.isna(q_table.get((state, a), 0.0))]
                action = np.random.choice(valid_actions)
            else:
                # 利用：选择最大Q值的动作
                q_values = [(a, q_table.get((state, a), float('-inf'))) for a in actions 
                           if not pd.isna(q_table.get((state, a), 0.0))]
                action = max(q_values, key=lambda x: x[1])[0]
            
            # 计算下一个状态
            row_change, col_change = action_map[action]
            new_row, new_col = current_row + row_change, current_col + col_change
            
            # 检查是否合法移动
            if 0 <= new_row < 4 and 0 <= new_col < 4:
                next_state_coords = (new_row, new_col)
                next_state = inverse_grid_map[next_state_coords]
                
                # 如果是障碍物，保持原位置
                if next_state == 'S5':
                    next_state = state
            else:
                # 超出边界，保持原位置
                next_state = state
            
            # 计算奖励
            if next_state == 'G':
                reward = 10  # 到达终点
                # 注意：在单次更新模式下，我们不会因为到达终点而结束这次更新
            else:
                reward = -1  # 移动成本
            
            # 计算next_state的最大Q值
            if next_state == 'G':
                max_next_q = 0
            else:
                next_q_values = [q_table.get((next_state, a), float('-inf')) for a in actions 
                                if not pd.isna(q_table.get((next_state, a), 0.0))]
                max_next_q = max(next_q_values) if next_q_values else 0
            
            # 更新Q值
            old_q = q_table.get((state, action), 0.0)
            # Q-learning公式: Q(s,a) = Q(s,a) + α[r + γ·max_a'(Q(s',a')) - Q(s,a)]
            new_q = old_q + alpha * (reward + gamma * max_next_q - old_q)
            q_table[(state, action)] = new_q
            
            # 记录Q值变化
            if not pd.isna(old_q) and not pd.isna(new_q):
                q_change = abs(new_q - old_q)
                max_q_change = max(max_q_change, q_change)
        
        # 记录此次迭代后的Q表
        q_history.append(q_table.copy())
        
        # 记录Q值最大变化
        q_change_history.append(max_q_change)
        
        # 提取当前策略
        current_policy = {}
        for s in states:
            if s != 'G' and s != 'S5':  # 排除终点和障碍物
                valid_q_values = [(a, q_table.get((s, a), float('-inf'))) for a in actions 
                                 if not pd.isna(q_table.get((s, a), 0.0))]
                best_action = max(valid_q_values, key=lambda x: x[1])[0]
                current_policy[s] = best_action
        
        policy_history.append(current_policy.copy())
        
        # 策略收敛检测 - 检查策略是否稳定
        if current_policy == previous_policy:
            stable_policy_count += 1
        else:
            stable_policy_count = 0
        
        previous_policy = current_policy.copy()
        
        # 收敛条件：
        # 1. 连续30次迭代策略不变 - 表示策略已经稳定
        # 2. 或最大Q值变化小于阈值 - 表示Q值变化微小，已经接近收敛
        if (stable_policy_count >= 30) or (max_q_change < convergence_threshold):
            convergence_iteration = iteration
            # 确保记录了足够的迭代历史用于可视化
            while len(q_history) <= max_iterations:
                q_history.append(q_table.copy())
                policy_history.append(current_policy.copy())
                q_change_history.append(0.0)  # 已收敛，变化为0
                
            # 输出收敛信息
            print(f"Q-learning在第{convergence_iteration}次迭代收敛。")
            print(f"收敛条件: {'策略稳定' if stable_policy_count >= 30 else '最大Q值变化小于阈值'}")
            break
    
    return q_history, convergence_iteration, policy_history, q_change_history

def render_q_table_update():
    st.subheader("Q表更新过程")
    
    # 创建一个4x4的网格世界用于展示
    st.markdown("""
    让我们通过一个简单的4x4网格世界示例，来理解Q表的更新过程。
    
    在这个环境中:
    - S0是起点（左上角）
    - G是终点（右下角）
    - 方块S5是障碍物（不可通行）
    - 每一步可以选择上、下、左、右四个方向移动
    - 到达终点获得+10的奖励，其他步骤获得-1的奖励（表示移动的成本）
    - 撞到障碍物或边界会停留在原地，同样得到-1的奖励
    """)
    
    # 定义4x4网格世界的初始Q表
    initial_q_table = pd.DataFrame(
        0.0, 
        index=['S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'G'],
        columns=['上', '右', '下', '左']
    )
    
    # 将障碍物S5对应的值设为NaN（表示不可用）
    for action in initial_q_table.columns:
        initial_q_table.loc['S5', action] = float('nan')
    
    # 将终点G对应的值设为0（终点不需要继续行动）
    for action in initial_q_table.columns:
        initial_q_table.loc['G', action] = 0.0
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**网格世界示意图**")
        
        # 创建并显示网格世界图
        grid_data = [
            ['S0', 'S1', 'S2', 'S3'],
            ['S4', 'S5', 'S6', 'S7'],
            ['S8', 'S9', 'S10', 'S11'],
            ['S12', 'S13', 'S14', 'G']
        ]
        
        grid_df = pd.DataFrame(grid_data)
        
        # 创建样式
        def grid_style(val):
            color = '#D6EAF8'  # 默认浅蓝色
            
            if val == 'S0':
                color = '#ABEBC6'  # 起点为绿色
            elif val == 'G':
                color = '#F9E79F'  # 终点为黄色
            elif val == 'S5':
                color = '#E74C3C'  # 障碍物为红色
                
            return f'background-color: {color}; color: black; text-align: center; font-weight: bold; padding: 10px;'
        
        styled_grid = grid_df.style.applymap(grid_style)
        st.write(styled_grid.to_html(), unsafe_allow_html=True)
    
    st.markdown("""
        **状态说明：**
        - 绿色(S0): 起点
        - 黄色(G): 终点(+10奖励)
        - 红色(S5): 障碍物(不可通行)
        - 蓝色: 普通状态(-1奖励)
        """)
    
    with col2:
        st.markdown("**初始Q表**")
        st.dataframe(initial_q_table)
        
        st.markdown("""
        初始状态下，Q表中所有值均为0（除了障碍物S5设为NaN表示不可用）。
        
        随着智能体在环境中探索，Q表将根据获得的奖励和经验不断更新。
        """)
    
    st.markdown("---")
    
    # 多次更新演示
    st.markdown("### Q表的迭代更新")
    
    # 创建列，使UI更清晰
    col1, col2 = st.columns([1, 3])
    
    with col1:
        iteration_slider = st.slider(
            "训练迭代次数", 
            min_value=1, 
            max_value=10000, 
            value=100,
            help="训练迭代次数指的是Q-learning算法执行的更新次数。每次迭代，会根据设置更新一个或多个状态-动作对的Q值。"
        )
        
        # 添加一个按钮，显示收敛时的Q表值
        show_converged = st.checkbox("显示收敛时的Q表", value=False)
        
        # 添加算法参数设置
        alpha = st.slider("学习率(α)", 0.01, 0.5, 0.1, 0.01)
        gamma = st.slider("折扣因子(γ)", 0.5, 0.99, 0.9, 0.01)
        epsilon = st.slider("探索率(ε)", 0.01, 0.5, 0.1, 0.01)
        
        # 添加每次迭代更新的状态-动作对数量选择
        updates_per_iteration = st.slider(
            "每次迭代更新的状态-动作对数量", 
            min_value=1, 
            max_value=20, 
            value=1,
            help="控制每次迭代中更新多少个状态-动作对的Q值。设置为1时，最符合标准Q-learning算法的定义。"
        )
        
        # 运行Q-learning按钮
        run_button = st.button("运行Q-learning算法")
        
        # 清除缓存按钮
        clear_cache = st.button("清除缓存")
        
        if clear_cache:
            if 'q_history' in st.session_state:
                del st.session_state.q_history
            if 'convergence_iteration' in st.session_state:
                del st.session_state.convergence_iteration
            if 'policy_history' in st.session_state:
                del st.session_state.policy_history
            if 'q_change_history' in st.session_state:
                del st.session_state.q_change_history
            st.experimental_rerun()
        
        # 运行Q-learning算法计算真实的Q表
        if run_button or ('q_history' not in st.session_state):
            with st.spinner('正在运行Q-learning算法...'):
                q_history, convergence_iteration, policy_history, q_change_history = run_q_learning(
                    max_iterations=10000, 
                    alpha=alpha, 
                    gamma=gamma, 
                    epsilon=epsilon,
                    updates_per_iteration=updates_per_iteration
                )
                st.session_state.q_history = q_history
                st.session_state.convergence_iteration = convergence_iteration
                st.session_state.policy_history = policy_history
                st.session_state.q_change_history = q_change_history
                st.success(f"Q-learning完成！在第{convergence_iteration}次迭代收敛。")
        
        # 获取缓存的结果
        q_history = st.session_state.q_history
        convergence_iteration = st.session_state.convergence_iteration
        policy_history = st.session_state.policy_history
        q_change_history = st.session_state.get('q_change_history', [0.0] * len(q_history))
                
        # 设置要使用的迭代
        if show_converged:
            st.info(f"显示收敛状态下的Q表（在第{convergence_iteration}次迭代后收敛）")
            # 使用收敛标志而不是迭代次数
            use_converged_values = True
            iteration_to_use = convergence_iteration  # 使用实际收敛的迭代次数
        else:
            use_converged_values = False
            iteration_to_use = min(iteration_slider, len(q_history) - 1)
        
        # 添加关于如何确定收敛迭代次数的说明
        if show_converged:
            st.markdown("""
            ### 关于收敛迭代次数的确定
            
            **本演示中的收敛迭代次数是如何确定的？**
            
            在本演示中，收敛迭代次数是由算法自动确定的：
            1. 当连续30次迭代中，最优策略不再变化时认为收敛
            2. 或当最大Q值变化小于阈值(0.001)时认为收敛
            
            **实际应用中如何确定收敛：**
            
            在实际应用中，我们通常不预先设定固定的迭代次数，而是使用一个收敛判断标准：
            1. **策略稳定性检查**：连续N次迭代中，所有状态的最优动作保持不变
            2. **Q值变化率监控**：当最大Q值变化率小于某个阈值ε（如0.001）时认为收敛
            3. **奖励稳定性**：评估策略的平均回报不再显著提高
            
            伪代码示例：
            ```python
            # 策略稳定性检查
            if current_policy == previous_policy for last_n_iterations:
                consider_converged = True
                
            # Q值变化率检查
            max_delta_q = max(|Q_new(s,a) - Q_old(s,a)|) for all s,a
            if max_delta_q < epsilon:
                consider_converged = True
            ```
            
            **收敛时间与环境复杂度的关系：**
            
            收敛所需的迭代次数与多种因素相关：
            - 状态空间大小：状态越多，收敛通常需要更多迭代
            - 环境随机性：随机性越大，收敛通常需要更多迭代
            - 学习参数设置：学习率、折扣因子、探索率等设置直接影响收敛速度
            """)
    
    # 创建一个基于实际计算的Q表
    # 从Q历史中获取指定迭代的Q表
    q_table_selected = q_history[iteration_to_use]
    
    # 将Q表转换为DataFrame格式
    q_df = pd.DataFrame(index=sorted(['S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'G']), 
                        columns=sorted(['上', '右', '下', '左']))
    
    for state in q_df.index:
        for action in q_df.columns:
            # 确保值是浮点数类型
            q_value = q_table_selected.get((state, action), 0.0)
            q_df.loc[state, action] = float(q_value) if not pd.isna(q_value) else np.nan
    
    with col2:
        st.write("#### 经过多次迭代更新后的Q表")
        st.dataframe(q_df, use_container_width=True)
        
        if use_converged_values:
            st.markdown(f"""
            上表展示了完全收敛状态下的Q表（最终稳定值）。
            
            **收敛的含义**：
            
            收敛意味着即使继续训练，Q表中的**策略**不再发生变化，尽管Q值可能仍有微小波动。在这种状态下：
            
            1. 对于每个状态，最优动作已经确定且保持稳定
            2. Q值可能仍有微小变化，但不足以改变最优动作的选择
            3. 算法已经找到了从任意状态到达目标的最佳路径
            
            **收敛过程的特点**：
            1. 初期：Q值变化大，最优动作频繁改变
            2. 中期：Q值变化减小，最优动作开始稳定
            3. 后期：Q值变化很小，最优动作完全稳定，此时称为"策略收敛"
            
            **完全收敛与值收敛的区别**：
            - **策略收敛**：每个状态的最优动作不再改变（已达到）
            - **值收敛**：Q值完全停止变化（理论上需要无限次迭代才能达到）
            
            **收敛所需的迭代次数**：
            
            对于这个4x4的网格世界问题，Q-learning算法在第{convergence_iteration}次迭代达到策略收敛。
            实际收敛速度取决于多种因素：
            - 学习率(α)：本例中使用{alpha}
            - 折扣因子(γ)：本例中使用{gamma}
            - 探索策略：ε-贪婪策略，ε={epsilon}
            - 环境复杂度：状态空间大小、奖励分布等
            """)
        else:
            st.markdown(f"""
            上表展示了经过{iteration_to_use}次迭代后的Q表。
            
            **每次迭代更新的状态-动作对数量**：

            在本演示中，您可以通过设置"每次迭代更新的状态-动作对数量"参数来控制每次迭代更新多少个Q值：

            1. **设置为1（标准Q-learning）**：每次迭代只更新一个状态-动作对的Q值，这是标准Q-learning算法的定义方式。
               - 在迭代次数为1时，Q表中只有一个值被更新
               - 迭代次数增加时，更多的状态-动作对会被更新

            2. **设置为更大的值**：每次迭代会更新多个状态-动作对的Q值，这可以加速学习过程：
               - 相当于在每次迭代中执行多次Q值更新
               - 可以更快地传播奖励信息，但与标准Q-learning定义有所不同

            当前设置：每次迭代更新{updates_per_iteration}个状态-动作对

            **Q表更新的起点与传播方向**：

            Q表更新是**从起点开始**的，而不是从终点开始：
            1. 更新过程从随机选择的状态开始（在标准的强化学习任务中通常是智能体当前所处的状态）
            2. 每次到达终点并获得正奖励后，这个正值会通过Q-learning更新公式向前传播
            3. 初始时，只有靠近终点的状态会有较高的Q值
            4. 随着训练进行，高Q值会逐渐向远离终点的状态传播
            5. 价值传播方向是从终点到起点，但更新过程是从当前状态开始的

            这种价值传播机制类似于水流从高处向低处流动，但智能体的实际移动是从低处（起点）向高处（终点）的过程。

            **训练迭代次数的含义**：

            在Q-learning中，迭代次数表示算法更新Q表的总次数：
            1. 每次迭代，算法会更新一个或多个状态-动作对的Q值
            2. 更新使用的是Q-learning公式：Q(s,a) = Q(s,a) + α[r + γ·max_a'(Q(s',a')) - Q(s,a)]
            3. 收敛通常需要成千上万次迭代，具体取决于环境的复杂度和学习参数

            在实际训练中，我们通常会执行足够多的迭代，直到Q表收敛到策略稳定（即使Q值可能仍有微小变化）。根据算法计算，这个4x4网格世界在第{convergence_iteration}次迭代达到策略收敛。
            """)
    
    # 策略可视化
    if iteration_to_use > 10 or use_converged_values:
        st.markdown("### 当前迭代的最优策略")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**网格世界中的最优路径**")
            
            # 定义网格世界
            grid_matrix = [
                ['S0', 'S1', 'S2', 'S3'],
                ['S4', 'S5', 'S6', 'S7'],
                ['S8', 'S9', 'S10', 'S11'],
                ['S12', 'S13', 'S14', 'G']
            ]
            
            # 从Q表中提取每个状态的最优动作
            optimal_actions = {}
            for state in q_df.index:
                if state == 'G':
                    optimal_actions[state] = 'G'  # 终点
                elif state == 'S5':
                    optimal_actions[state] = 'S5'  # 障碍物
                else:
                    # 安全获取最大Q值的索引
                    try:
                        # 确保处理的是数值类型
                        row_values = q_df.loc[state].copy().astype(float, errors='ignore')
                        # 手动过滤掉NaN值
                        valid_values = {}
                        for action, value in row_values.items():
                            if not pd.isna(value) and value != 0:
                                valid_values[action] = value
                        
                        if valid_values:
                            best_action = max(valid_values.items(), key=lambda x: x[1])[0]
                            optimal_actions[state] = best_action
                        else:
                            optimal_actions[state] = '未知'
                    except Exception as e:
                        optimal_actions[state] = '未知'
                        st.error(f"处理状态{state}的Q值时出错: {e}")
            
            # 创建动作对应的箭头符号
            action_symbols = {'上': '↑', '右': '→', '下': '↓', '左': '←', '未知': '?'}
            
            # 生成网格数据
            optimal_path_data = []
            for i in range(4):  # 4行
                row = []
                for j in range(4):  # 4列
                    cell_state = grid_matrix[i][j]
                    if cell_state == 'G':
                        row.append('G')
                    elif cell_state == 'S5':
                        row.append('S5')
                    else:
                        action = optimal_actions.get(cell_state, '未知')
                        if action in action_symbols:
                            row.append(f"{cell_state}{action_symbols[action]}")
                        else:
                            row.append(cell_state)
                optimal_path_data.append(row)
            
            optimal_path_df = pd.DataFrame(optimal_path_data)
            
            # 创建样式
            def path_style(val):
                color = '#D6EAF8'  # 默认浅蓝色
                
                if '↑' in val or '→' in val or '↓' in val or '←' in val:
                    color = '#85C1E9'  # 路径为深蓝色
                
                if val.startswith('S0'):
                    color = '#ABEBC6'  # 起点为绿色
                elif val == 'G':
                    color = '#F9E79F'  # 终点为黄色
                elif val == 'S5':
                    color = '#E74C3C'  # 障碍物为红色
                elif '?' in val:
                    color = '#F5CBA7'  # 未知动作为橙色
                    
                return f'background-color: {color}; color: black; text-align: center; font-weight: bold; padding: 10px;'
            
            styled_path = optimal_path_df.style.applymap(path_style)
            st.write(styled_path.to_html(), unsafe_allow_html=True)
            
            st.markdown("""
            **最优路径说明：**
            - 箭头表示在该状态下基于当前Q表计算出的最优动作方向
            - 路径是从Q表值自动生成的，不是硬编码的预设路径
            - 从S0出发，沿着箭头指示的方向可以找到到达终点G的最优路径
            """)
        
        with col2:
            st.markdown("**Q表的策略提取**")
            
            # 从当前Q表中提取策略
            policy_table = pd.DataFrame(index=q_df.index, columns=['最优动作', '预期价值'])
            
            for state in q_df.index:
                if state == 'G':
                    policy_table.loc[state] = ['终点', 0.0]
                elif state == 'S5':
                    policy_table.loc[state] = ['障碍', float('nan')]
                else:
                    try:
                        # 转换为浮点数类型以确保数值运算
                        row_values = q_df.loc[state].copy().astype(float, errors='ignore')
                        # 检查是否有非NaN和非零值
                        valid_values = {}
                        for action, value in row_values.items():
                            if not pd.isna(value) and value != 0:
                                valid_values[action] = value
                        
                        if valid_values:
                            best_action = max(valid_values.items(), key=lambda x: x[1])[0]
                            best_value = float(valid_values[best_action])
                            policy_table.loc[state] = [best_action, best_value]
                        else:
                            policy_table.loc[state] = ['未知', 0.0]
                    except Exception as e:
                        policy_table.loc[state] = ['未知', 0.0]
                        st.error(f"处理状态{state}的最优动作时出错: {e}")
            
            st.dataframe(policy_table, use_container_width=True)
            
            st.markdown(f"""
            **策略提取说明：**
            - 对于每个状态，选择Q值最大的动作作为最优动作
            - 最优动作的Q值代表从该状态出发，遵循最优策略能获得的预期累积奖励
            - 负值越大（越接近0），表示该路径更优（奖励更高）
            - 最优策略提取直接基于Q表，确保了与网格世界中显示的路径一致性
            - 完全收敛需要约{convergence_iteration}次迭代，此时策略已稳定不变
            """)

    # 在策略可视化部分之后添加Q表变化可视化
    st.markdown("### Q表变化过程")
    st.markdown("""
    这个图表展示了Q-learning算法训练过程中，Q值的最大变化量随迭代次数的变化情况。
    通常情况下，随着训练的进行，Q值的变化会逐渐减小，直到最终收敛。
    """)

    # 获取Q值变化历史数据
    change_data = pd.DataFrame({
        '迭代次数': range(1, len(q_change_history) + 1),
        'Q值最大变化': q_change_history
    })

    # 绘制Q值变化曲线
    fig = px.line(change_data, x='迭代次数', y='Q值最大变化', title='Q值变化趋势')
    fig.add_vline(x=convergence_iteration, line_dash="dash", line_color="red", 
                  annotation_text=f"收敛点: 第{convergence_iteration}次迭代", 
                  annotation_position="top right")
    fig.update_layout(
        xaxis_title="迭代次数",
        yaxis_title="Q值最大变化",
        height=400,
        xaxis_range=[0, convergence_iteration]
    )
    st.plotly_chart(fig, use_container_width=True)

    # 添加关于收敛的说明
    st.markdown(f"""
    **关于Q-learning收敛的观察**:

    1. **收敛速度**：在本例中，算法在第{convergence_iteration}次迭代达到收敛。
    2. **收敛特征**：观察Q值变化曲线可以看到，随着训练进行，Q值变化逐渐趋向于零。
    3. **学习参数影响**：
       - 学习率(α={alpha})：较高的学习率可能导致快速但不稳定的学习，较低的学习率学习更平稳但更慢。
       - 折扣因子(γ={gamma})：较高的值会使算法更重视长期奖励，可能需要更多迭代才能收敛。
       - 探索率(ε={epsilon})：较高的探索率会增加环境探索，但可能延长收敛时间。

    4. **收敛判断标准**：本算法使用了两个收敛标准：
       - 策略稳定性：连续30次迭代策略保持不变
       - Q值变化：最大Q值变化低于阈值(0.001)
    """)

def render_value_propagation_tab():
    st.subheader("价值传播演示")
    
    st.markdown("""
    ## 价值传播与Q表的关系
    
    价值传播是**理解Q-Learning算法工作原理**的重要概念，它与Q表有密切关系：
    
    ### Q表和价值函数的关系
    1. **Q表表示状态-动作对的价值**：Q(s,a)表示在状态s下采取动作a的长期累积价值
    2. **状态价值函数**：V(s)表示状态s的价值，等于该状态下所有可能动作的最大Q值： V(s) = max_a Q(s,a)
    3. **价值传播过程**：Q-learning算法通过不断更新Q表，实现价值从高奖励区域（如终点）向其他状态的传播
    
    ### 本演示的目的
    这个价值传播演示**简化了Q-learning的核心原理**，通过直接展示状态价值V(s)如何从终点传播到整个环境，帮助你理解Q-learning算法的基础工作机制。
    
    Q-learning算法其实就是通过不断更新Q表，实现价值在整个状态空间的传播，最终形成指向目标的"梯度场"，引导智能体找到最优路径。
    """)
    
    # 创建一个简单的价值传播示例
    st.markdown("### 价值传播实例")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 初始状态")
        
        # 创建一个7x7的网格，只有终点有价值
        initial_values = np.zeros((7, 7))
        initial_values[6, 6] = 10  # 终点奖励为10
        
        # 显示初始价值图
        fig1 = px.imshow(
            initial_values, 
            color_continuous_scale="Blues",
            labels=dict(x="列", y="行", color="价值"),
            title="初始状态下的价值分布"
        )
        
        # 添加文本注释
        for i in range(7):
            for j in range(7):
                value = initial_values[i, j]
                if value == 10:
                    text = "终点\n(10)"
                    fig1.add_annotation(
                        x=j, y=i,
                        text=text,
                        showarrow=False,
                        font=dict(color="black", size=10)
                    )
        
        st.plotly_chart(fig1)
        
        st.markdown("""
        **初始状态说明：**
        - 7×7网格世界，只有右下角终点有正奖励(+10)
        - 其他位置初始价值均为0
        - 智能体可以上、下、左、右四个方向移动
        - 每走一步获得-1的奖励（表示移动成本）
        """)
    
    # 滑块控制传播次数
    propagation_steps = st.slider(
        "价值传播次数", 
        min_value=1, 
        max_value=20, 
        value=5,
        help="价值传播次数指的是执行贝尔曼方程更新的轮数。每次更新，状态的价值会根据相邻状态的价值进行调整。"
    )
    
    # 计算价值传播
    gamma = 0.9  # 折扣因子
    propagated_values = initial_values.copy()
    
    # 执行价值传播
    for _ in range(propagation_steps):
        new_values = propagated_values.copy()
        for i in range(7):
            for j in range(7):
                # 跳过终点
                if i == 6 and j == 6:
                    continue
                
                # 获取上、右、下、左四个方向的价值（如果有效）
                neighbors_values = []
                if i > 0:  # 上
                    neighbors_values.append(propagated_values[i-1, j])
                if j < 6:  # 右
                    neighbors_values.append(propagated_values[i, j+1])
                if i < 6:  # 下
                    neighbors_values.append(propagated_values[i+1, j])
                if j > 0:  # 左
                    neighbors_values.append(propagated_values[i, j-1])
                
                # 计算新价值：-1(移动成本) + gamma * 最大邻居价值
                if neighbors_values:
                    new_values[i, j] = -1 + gamma * max(neighbors_values)
        
        propagated_values = new_values
    
    with col2:
        st.markdown(f"#### 价值传播后状态（{propagation_steps}次传播）")
        
        # 显示传播后的价值图
        fig2 = px.imshow(
            propagated_values, 
            color_continuous_scale="Blues",
            labels=dict(x="列", y="行", color="价值"),
            title=f"经过{propagation_steps}次传播后的价值分布"
        )
        
        # 添加文本注释 - 显示数值
        for i in range(7):
            for j in range(7):
                value = propagated_values[i, j]
                if value != 0:  # 只显示非零值
                    text = f"{value:.2f}"
                    if i == 6 and j == 6:
                        text = "终点\n(10)"
                    fig2.add_annotation(
                        x=j, y=i,
                        text=text,
                        showarrow=False,
                        font=dict(color="black", size=9)
                    )
        
        st.plotly_chart(fig2)
    
    st.markdown("""
    ### 价值传播计算过程详解
    
    价值传播使用的是**贝尔曼方程**，计算公式为：
    
    V(s) = max_a [ R(s,a) + γ · V(s') ]
    
    其中：
    - V(s)是状态s的价值
    - R(s,a)是在状态s采取动作a获得的即时奖励（本例中为-1）
    - γ是折扣因子（本例中为0.9）
    - V(s')是执行动作a后到达的新状态s'的价值
    - max_a表示选择能获得最大价值的动作
    
    #### 具体计算示例：
    """)
    
    # 显示第一次传播计算过程
    st.markdown("""
    **以位置(5,6)为例（终点上方）**:
    
    第1次传播计算过程：
    1. 识别(5,6)位置的邻居：
       - 上邻居(4,6)：初始价值为0
       - 右邻居：不存在（已到边界）
       - 下邻居(6,6)：终点，价值为10
       - 左邻居(5,5)：初始价值为0
       
    2. 按贝尔曼方程计算新价值：
       V(5,6) = -1 + 0.9 × max(0, 不存在, 10, 0)
              = -1 + 0.9 × 10
              = -1 + 9
              = 8.0
              
    第一次传播后，位置(5,6)的价值从0更新为8.0
    """)
    
    # 显示第二次传播计算过程
    one_step_values = initial_values.copy()
    # 执行一次价值传播以获取真实值
    for i in range(7):
        for j in range(7):
            if i == 6 and j == 6:  # 跳过终点
                continue
            
            # 获取邻居价值
            neighbors_values = []
            if i > 0:  # 上
                neighbors_values.append(initial_values[i-1, j])
            if j < 6:  # 右
                neighbors_values.append(initial_values[i, j+1])
            if i < 6:  # 下
                neighbors_values.append(initial_values[i+1, j])
            if j > 0:  # 左
                neighbors_values.append(initial_values[i, j-1])
            
            # 计算新价值
            if neighbors_values:
                one_step_values[i, j] = -1 + gamma * max(neighbors_values)

    # 执行第二次价值传播以获取真实的第二次传播值
    two_step_values = one_step_values.copy()
    for i in range(7):
        for j in range(7):
            if i == 6 and j == 6:  # 跳过终点
                continue
            
            # 获取邻居价值
            neighbors_values = []
            if i > 0:  # 上
                neighbors_values.append(one_step_values[i-1, j])
            if j < 6:  # 右
                neighbors_values.append(one_step_values[i, j+1])
            if i < 6:  # 下
                neighbors_values.append(one_step_values[i+1, j])
            if j > 0:  # 左
                neighbors_values.append(one_step_values[i, j-1])
            
            # 计算新价值
            if neighbors_values:
                two_step_values[i, j] = -1 + gamma * max(neighbors_values)
    
    # 获取真实的邻居值
    neighbors_5_6 = []
    # 上邻居(4,6)
    neighbors_5_6.append((one_step_values[4, 6], "上邻居(4,6)"))
    # 右邻居不存在（已到边界）
    # 下邻居(6,6)
    neighbors_5_6.append((one_step_values[6, 6], "下邻居(6,6)"))
    # 左邻居(5,5)
    neighbors_5_6.append((one_step_values[5, 5], "左邻居(5,5)"))
    
    # 按价值排序
    neighbors_5_6.sort(key=lambda x: x[0], reverse=True)
    
    neighbors_text = ""
    for value, name in neighbors_5_6:
        value_str = f"{value:.2f}" if value != 10 else "10"
        neighbors_text += f"       - {name}：第一次传播后的价值为{value_str}\n"
    
    st.markdown(f"""
    第2次传播计算过程：
    1. 识别(5,6)位置的邻居：
{neighbors_text}
    2. 按贝尔曼方程计算新价值：
       V(5,6) = -1 + 0.9 × max({", ".join([f"{v[0]:.2f}" if v[0] != 10 else "10" for v in neighbors_5_6])})
              = -1 + 0.9 × {max([v[0] for v in neighbors_5_6]):.2f}
              = -1 + {0.9 * max([v[0] for v in neighbors_5_6]):.2f}
              = {-1 + 0.9 * max([v[0] for v in neighbors_5_6]):.2f}
              
    第二次传播后，位置(5,6)的价值从{one_step_values[5, 6]:.2f}更新为{two_step_values[5, 6]:.2f}
    """)
    
    st.markdown("""
    ### 价值传播与Q-learning的联系
    
    价值传播演示展示了**价值如何从高奖励区域向整个环境传播**的过程，这与Q-learning算法的核心原理相同：
    
    1. **Q表更新公式与贝尔曼方程的关系**：
       - Q-learning更新公式：Q(s,a) ← Q(s,a) + α[r + γ·maxQ(s',a') - Q(s,a)]
       - 贝尔曼方程：V(s) = max_a [R(s,a) + γ·V(s')]
       
       这两个公式本质上都在实现同一个目标：将未来可能获得的奖励（折扣后）传播回当前状态。
    
    2. **从Q表到价值函数的映射**：
       - Q表中存储了每个状态-动作对的价值 Q(s,a)
       - 而状态价值函数 V(s) = max_a Q(s,a)，即Q表中该状态下最大值的动作对应的值
       - 这个演示中直接使用了状态价值函数V(s)，而Q-learning则是通过更新Q(s,a)间接更新V(s)
       
    3. **从Q表到最优策略**：
       - 在Q-learning中，我们通过Q表提取最优策略：在每个状态选择Q值最大的动作
       - 在价值传播演示中，我们通过状态价值梯度提取最优路径：选择邻居中价值最高的方向移动
       
    价值传播演示可以帮助你直观理解Q-learning算法背后的核心思想—通过不断迭代，让价值从高奖励区域向其他状态传播，最终形成一个指向目标的"价值梯度场"。
    """)
    
    # 添加动画演示
    st.markdown("### 价值传播动画演示")
    
    if st.button("生成价值传播动画"):
        # 创建价值传播动画数据
        all_steps_values = [initial_values.copy()]
        propagated_values = initial_values.copy()
        
        # 执行价值传播并保存每一步
        for _ in range(20):  # 保存20步
            new_values = propagated_values.copy()
            for i in range(7):
                for j in range(7):
                    # 跳过终点
                    if i == 6 and j == 6:
                        continue
                    
                    # 获取上、右、下、左四个方向的价值（如果有效）
                    neighbors_values = []
                    if i > 0:  # 上
                        neighbors_values.append(propagated_values[i-1, j])
                    if j < 6:  # 右
                        neighbors_values.append(propagated_values[i, j+1])
                    if i < 6:  # 下
                        neighbors_values.append(propagated_values[i+1, j])
                    if j > 0:  # 左
                        neighbors_values.append(propagated_values[i, j-1])
                    
                    # 计算新价值
                    if neighbors_values:
                        new_values[i, j] = -1 + gamma * max(neighbors_values)
            
            propagated_values = new_values.copy()
            all_steps_values.append(propagated_values.copy())
        
        # 创建动画
        frames = []
        for step, values in enumerate(all_steps_values):
            fig = px.imshow(
                values, 
                color_continuous_scale="Blues",
                labels=dict(x="列", y="行", color="价值"),
                title=f"价值传播过程：第{step}步"
            )
            
            # 添加文本注释 - 只在特定位置显示数值
            for i in range(7):
                for j in range(7):
                    if (i >= 4 and j >= 4) or (i == 6 or j == 6):  # 只在终点附近显示数值
                        value = values[i, j]
                        if value != 0:  # 只显示非零值
                            text = f"{value:.1f}"
                            if i == 6 and j == 6:
                                text = "10"
                            fig.add_annotation(
                                x=j, y=i,
                                text=text,
                                showarrow=False,
                                font=dict(color="black", size=9)
                            )
            
            fig.update_layout(height=400)
            frames.append(fig)
        
        # 显示动画
        animation_placeholder = st.empty()
        
        for frame in frames:
            animation_placeholder.plotly_chart(frame, use_container_width=True)
            time.sleep(0.5)  # 控制播放速度
        
        st.success("动画播放完成！价值已从终点传播到整个网格。")
        
        # 最优路径可视化
        st.markdown("### 最终收敛后的最优路径")
        
        # 根据最终价值计算最优策略
        final_values = all_steps_values[-1]
        optimal_policy = np.zeros((7, 7), dtype=object)
        
        for i in range(7):
            for j in range(7):
                if i == 6 and j == 6:
                    optimal_policy[i, j] = "G"  # 终点
                    continue
                
                # 查找价值最高的邻居方向
                neighbor_values = []
                directions = []
                
                if i > 0:  # 上
                    neighbor_values.append(final_values[i-1, j])
                    directions.append("↑")
                if j < 6:  # 右
                    neighbor_values.append(final_values[i, j+1])
                    directions.append("→")
                if i < 6:  # 下
                    neighbor_values.append(final_values[i+1, j])
                    directions.append("↓")
                if j > 0:  # 左
                    neighbor_values.append(final_values[i, j-1])
                    directions.append("←")
                
                if neighbor_values:
                    best_idx = np.argmax(neighbor_values)
                    optimal_policy[i, j] = directions[best_idx]
        
        # 创建策略图
        policy_df = pd.DataFrame(optimal_policy)
        
        st.dataframe(policy_df)
        
        st.markdown("""
        **最优路径说明：**
        - 箭头表示在该位置应该选择的最优移动方向
        - 从任意位置出发，沿着箭头指示的方向移动，最终都能找到到达终点(G)的最优路径
        - 这正是价值传播的最终结果—形成一个指向终点的"梯度场"
    """)

def show():
    st.title("Q表可视化演示")
    
    tabs = ["Q表更新", "价值传播演示", "V表与Q表比较"]
    selected_tab = st.radio("选择演示内容：", tabs, horizontal=True)
    
    if selected_tab == "Q表更新":
        render_q_table_update()
    elif selected_tab == "价值传播演示":
        render_value_propagation_tab()
    elif selected_tab == "V表与Q表比较":
        # 导入V-Q比较模块
        import v_q_comparison
        v_q_comparison.show()
        
if __name__ == "__main__":
    show() 