import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import io
import base64
import matplotlib
import random
matplotlib.use('Agg')

def show():
    st.title("V表与Q表的深度比较")
    
    st.markdown("""
    ## 核心概念解析
    
    在强化学习中，我们有两种主要的价值函数表示方法：**状态价值函数(V表)** 和 **动作价值函数(Q表)**。
    虽然两者都是估计未来回报的方式，但它们在实际应用中有着显著差异。
    """)
    
    # 创建基本概念解释
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### V表 (状态价值函数)
        
        **定义**: V(s)表示在状态s下，遵循特定策略π可获得的期望累积回报。
        
        **公式**: 
        ```
        V^π(s) = E_π[ R_t+1 + γR_t+2 + γ²R_t+3 + ... | S_t = s ]
        ```
        
        **贝尔曼方程**:
        ```
        V(s) = max_a [ R(s,a) + γ·V(s') ]
        ```
        
        **直观理解**: 
        "在这个位置站着值多少钱？"
        """)
        
    with col2:
        st.markdown("""
        ### Q表 (动作价值函数)
        
        **定义**: Q(s,a)表示在状态s下执行动作a，然后遵循策略π可获得的期望累积回报。
        
        **公式**:
        ```
        Q^π(s,a) = E_π[ R_t+1 + γR_t+2 + γ²R_t+3 + ... | S_t = s, A_t = a ]
        ```
        
        **贝尔曼方程**:
        ```
        Q(s,a) = R(s,a) + γ·max_a' Q(s',a')
        ```
        
        **直观理解**:
        "在这个位置执行这个动作值多少钱？"
        """)
    
    # 添加V表和Q表的图示
    st.markdown("## 直观图解")
    
    # 生成SVG图像的函数
    def create_v_table_svg():
        fig, ax = plt.subplots(figsize=(5, 4))
        grid_size = 4
        values = np.array([
            [0.0, 2.5, 7.0, 10.0],
            [0.5, 3.0, 8.0, 7.0],
            [1.0, 4.0, 5.0, 3.0],
            [0.2, 1.0, 2.0, 1.0]
        ])
        
        im = ax.imshow(values, cmap='Blues')
        
        # 添加文本
        for i in range(grid_size):
            for j in range(grid_size):
                text = ax.text(j, i, f"V={values[i, j]:.1f}",
                               ha="center", va="center", color="black", fontsize=9)
        
        ax.set_title('V表 (状态价值)')
        ax.set_xticks(np.arange(grid_size))
        ax.set_yticks(np.arange(grid_size))
        ax.set_xticklabels([f"列{i}" for i in range(grid_size)])
        ax.set_yticklabels([f"行{i}" for i in range(grid_size)])
        
        fig.tight_layout()
        
        # 将图像转换为SVG
        buf = io.BytesIO()
        fig.savefig(buf, format='svg')
        buf.seek(0)
        svg_data = buf.getvalue().decode('utf-8')
        buf.close()
        plt.close(fig)
        
        return svg_data
    
    def create_q_table_svg():
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # 创建一个表格数据
        data = np.zeros((4, 4, 4))  # 4x4网格，每个单元有4个动作
        # 上、右、下、左的Q值
        data[0, 0] = [0.1, 0.4, 0.2, 0.0]  # (0,0)位置的4个动作
        data[0, 1] = [0.3, 2.1, 1.0, 0.5]
        data[0, 2] = [1.0, 5.0, 2.0, 2.5]
        data[0, 3] = [3.0, 0.0, 0.0, 7.0]
        
        data[1, 0] = [0.3, 1.0, 0.2, 0.0]
        data[1, 1] = [2.0, 3.0, 1.0, 1.0]
        data[1, 2] = [3.0, 6.0, 2.5, 3.0]
        data[1, 3] = [5.0, 0.0, 0.0, 8.0]
        
        data[2, 0] = [0.7, 1.5, 0.0, 0.0]
        data[2, 1] = [2.5, 4.0, 1.0, 1.5]
        data[2, 2] = [3.5, 4.5, 2.0, 4.0]
        data[2, 3] = [2.0, 0.0, 0.0, 5.0]
        
        data[3, 0] = [0.0, 0.5, 0.0, 0.0]
        data[3, 1] = [0.0, 2.0, 0.0, 0.5]
        data[3, 2] = [0.0, 3.0, 0.0, 1.0]
        data[3, 3] = [0.0, 0.0, 0.0, 2.0]
        
        grid_size = 4
        cell_size = 1.0
        
        # 绘制网格
        for i in range(grid_size):
            for j in range(grid_size):
                ax.add_patch(plt.Rectangle((j, i), cell_size, cell_size, fill=False, edgecolor='black'))
                
                # 添加上、右、下、左的Q值
                # 上
                ax.text(j + 0.5, i + 0.25, f"{data[i, j, 0]:.1f}", ha='center', va='center', fontsize=8)
                # 右
                ax.text(j + 0.75, i + 0.5, f"{data[i, j, 1]:.1f}", ha='center', va='center', fontsize=8)
                # 下
                ax.text(j + 0.5, i + 0.75, f"{data[i, j, 2]:.1f}", ha='center', va='center', fontsize=8)
                # 左
                ax.text(j + 0.25, i + 0.5, f"{data[i, j, 3]:.1f}", ha='center', va='center', fontsize=8)
                
                # 添加箭头指示方向
                # 找出最大Q值对应的动作
                best_action = np.argmax(data[i, j])
                arrows = ['↑', '→', '↓', '←']
                if np.max(data[i, j]) > 0:  # 只有当有非零值时才显示箭头
                    ax.text(j + 0.5, i + 0.5, arrows[best_action], ha='center', va='center', fontsize=15, color='red')
        
        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, grid_size)
        ax.invert_yaxis()  # 反转y轴，使(0,0)在左上角
        ax.set_aspect('equal')
        ax.set_title('Q表 (状态-动作价值)')
        ax.set_xticks(np.arange(grid_size) + 0.5)
        ax.set_yticks(np.arange(grid_size) + 0.5)
        ax.set_xticklabels([f"列{i}" for i in range(grid_size)])
        ax.set_yticklabels([f"行{i}" for i in range(grid_size)])
        ax.set_xticks(np.arange(0, grid_size + 1), minor=True)
        ax.set_yticks(np.arange(0, grid_size + 1), minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
        ax.tick_params(which="minor", size=0)
        
        # 添加图例
        ax.text(0.5, -0.5, "↑: 向上的Q值  →: 向右的Q值  ↓: 向下的Q值  ←: 向左的Q值", ha='center', transform=ax.transAxes)
        ax.text(0.5, -0.6, "红色箭头表示最优动作方向", ha='center', transform=ax.transAxes, color='red')
        
        fig.tight_layout()
        
        # 将图像转换为SVG
        buf = io.BytesIO()
        fig.savefig(buf, format='svg')
        buf.seek(0)
        svg_data = buf.getvalue().decode('utf-8')
        buf.close()
        plt.close(fig)
        
        return svg_data
    
    def create_model_free_vs_model_based_svg():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # 模型已知 (V表)
        grid_size = 3
        ax1.set_title('V表 (模型已知)')
        
        # 创建简单网格
        for i in range(grid_size):
            for j in range(grid_size):
                ax1.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black'))
                
                # 绘制状态转移概率
                if i == 1 and j == 1:  # 中间状态
                    # 向上转移
                    ax1.arrow(j+0.5, i+0.5, 0, -0.3, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
                    ax1.text(j+0.6, i+0.3, "p=0.8", fontsize=8)
                    # 向右转移
                    ax1.arrow(j+0.5, i+0.5, 0.3, 0, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
                    ax1.text(j+0.7, i+0.6, "p=0.1", fontsize=8)
                    # 向左转移
                    ax1.arrow(j+0.5, i+0.5, -0.3, 0, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
                    ax1.text(j+0.1, i+0.6, "p=0.1", fontsize=8)
                    
                    # 添加V值
                    ax1.text(j+0.5, i+0.8, "V=8.5", fontsize=10, ha='center')
        
        # 标记终点
        ax1.add_patch(plt.Rectangle((0, 0), 1, 1, fill=True, facecolor='lightgreen', alpha=0.5, edgecolor='black'))
        ax1.text(0.5, 0.5, "目标\nV=10", ha='center', va='center')
        
        ax1.text(1.5, -0.3, "需要知道状态转移概率", ha='center', fontsize=9)
        
        # 模型未知 (Q表)
        ax2.set_title('Q表 (模型未知)')
        
        # 创建简单网格
        for i in range(grid_size):
            for j in range(grid_size):
                ax2.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black'))
                
                # 中间状态的Q值
                if i == 1 and j == 1:
                    # 添加Q值
                    ax2.text(j+0.5, i+0.25, "Q(上)=8.0", fontsize=8, ha='center')
                    ax2.text(j+0.5, i+0.45, "Q(右)=5.2", fontsize=8, ha='center')
                    ax2.text(j+0.5, i+0.65, "Q(下)=3.1", fontsize=8, ha='center')
                    ax2.text(j+0.5, i+0.85, "Q(左)=4.6", fontsize=8, ha='center')
                    
                    # 最优动作 (向上)
                    ax2.arrow(j+0.5, i+0.5, 0, -0.3, head_width=0.1, head_length=0.1, fc='red', ec='red')
        
        # 标记终点
        ax2.add_patch(plt.Rectangle((0, 0), 1, 1, fill=True, facecolor='lightgreen', alpha=0.5, edgecolor='black'))
        ax2.text(0.5, 0.5, "目标", ha='center', va='center')
        
        ax2.text(1.5, -0.3, "无需知道状态转移，直接从经验中学习", ha='center', fontsize=9)
        
        # 配置轴
        for ax in [ax1, ax2]:
            ax.set_xlim(0, grid_size)
            ax.set_ylim(0, grid_size)
            ax.invert_yaxis()  # 反转y轴，使(0,0)在左上角
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
        
        fig.tight_layout()
        
        # 将图像转换为SVG
        buf = io.BytesIO()
        fig.savefig(buf, format='svg')
        buf.seek(0)
        svg_data = buf.getvalue().decode('utf-8')
        buf.close()
        plt.close(fig)
        
        return svg_data
    
    # 创建并显示SVG图像
    col1, col2 = st.columns(2)
    
    with col1:
        v_table_svg = create_v_table_svg()
        st.components.v1.html(v_table_svg, height=300, width=350)
        st.markdown("""
        **V表特点**:
        - 每个状态只有一个值
        - 表示"在该状态下的价值"
        - 不直接告诉应采取什么动作
        """)
    
    with col2:
        q_table_svg = create_q_table_svg()
        st.components.v1.html(q_table_svg, height=300, width=450)
        st.markdown("""
        **Q表特点**:
        - 每个状态有多个值(对应不同动作)
        - 表示"在该状态下执行特定动作的价值"
        - 直接指示最优动作(最大Q值对应的动作)
        """)
    
    # 模型已知vs模型未知图示
    st.markdown("## 模型已知vs模型未知")
    
    model_free_vs_model_based_svg = create_model_free_vs_model_based_svg()
    st.components.v1.html(model_free_vs_model_based_svg, height=350, width=700)
    
    st.markdown("""
    **关键区别**:
    1. **V表需要环境模型**：要更新V表，需要知道"执行动作a后会到达哪个状态s'"以及转移概率
    2. **Q表不需要环境模型**：可以直接从经验样本(s,a,r,s')学习
    """)
    
    # 详细比较
    st.markdown("## 详细比较")
    
    comparison_data = {
        "特性": [
            "存储空间", 
            "计算复杂度", 
            "环境模型需求", 
            "学习方式",
            "策略获取",
            "收敛速度",
            "适用场景",
            "算法示例"
        ],
        "V表 (状态价值函数)": [
            "更小 (|S|个值)",
            "更低",
            "需要已知模型",
            "通常需要完整扫描（动态规划）",
            "需要额外计算(需模型)",
            "在已知模型情况下更快",
            "环境规则已知的场景",
            "策略迭代、值迭代"
        ],
        "Q表 (动作价值函数)": [
            "更大 (|S|×|A|个值)",
            "更高",
            "无需已知模型",
            "可以直接从样本中学习",
            "直接获取(max_a Q(s,a))",
            "通常需要更多样本",
            "真实世界的未知环境",
            "Q-learning、SARSA"
        ]
    }
    
    df = pd.DataFrame(comparison_data)
    st.table(df)
    
    # 实际场景分析
    st.markdown("## 实际应用场景分析")
    
    st.markdown("""
    ### 为什么V表在演示中看起来更简单？
    
    在价值传播演示中，V表看起来更容易理解，因为：
    
    1. **已知环境**: 演示中的环境完全已知（移动规则、网格大小、终点位置等）
    2. **确定性环境**: 每个动作的结果是确定的（向上移动总是到达上方格子）
    3. **可视化简单**: 每个状态只有一个值，便于直观展示
    
    ### 现实世界为何更常用Q-learning？
    
    1. **未知环境模型**: 大多数实际场景中，我们不知道环境的精确规则
        - 例如：机器人不知道在未知地形上移动会滑多远
        - 自动驾驶汽车不知道其他车辆会如何反应
        
    2. **无法获得转移概率**: 即使环境部分已知，精确的状态转移概率通常无法获得
        - 例如：在游戏AI中，无法预测所有可能的对手行为
        
    3. **直接策略提取**: Q表允许直接获取最优动作，无需额外计算
    """)
    
    # 创建动图来说明Q-learning的学习过程
    st.markdown("## Q-learning学习过程示例")
    
    # 记录不同迭代次数的Q表
    iterations_to_record = [0, 5, 20, 100, 500, 1000]
    
    # 创建Q表学习过程的动态展示
    def create_q_learning_process():
        # 实现一个真实的Q-learning算法来替代模拟数据
        # 定义简单网格世界
        grid_size = 4
        
        # 定义状态空间
        states = [(i, j) for i in range(grid_size) for j in range(grid_size)]
        
        # 定义动作空间：上、右、下、左 [0, 1, 2, 3]
        actions = [0, 1, 2, 3]
        action_deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # 上、右、下、左
        
        # 终点在(0,0)位置
        goal_state = (0, 0)
        
        # 奖励设计
        def get_reward(state, next_state):
            if next_state == goal_state:
                return 10.0  # 到达终点奖励
            else:
                return -1.0  # 每一步移动成本
        
        # 状态转移
        def get_next_state(state, action):
            i, j = state
            di, dj = action_deltas[action]
            next_i, next_j = i + di, j + dj
            
            # 检查边界，如果超出边界则保持原位
            if 0 <= next_i < grid_size and 0 <= next_j < grid_size:
                return (next_i, next_j)
            else:
                return state
        
        # 检查一个动作是否有效（不会撞墙）
        def is_valid_action(state, action):
            i, j = state
            di, dj = action_deltas[action]
            next_i, next_j = i + di, j + dj
            return 0 <= next_i < grid_size and 0 <= next_j < grid_size
        
        # 初始化Q表
        q_table = {}
        for state in states:
            q_table[state] = {}
            for action in actions:
                if is_valid_action(state, action):
                    q_table[state][action] = 0.0
        
        # 特殊处理终点状态
        q_table[goal_state] = {a: 0.0 for a in actions if is_valid_action(goal_state, a)}
        
        # Q-learning参数
        alpha = 0.1      # 学习率
        gamma = 0.9      # 折扣因子
        epsilon = 0.1    # 探索率
        
        # 记录不同迭代次数的Q表
        q_tables_history = []
        
        # 记录初始Q表
        q_tables_history.append((0, {s: q_table[s].copy() for s in states}))
        
        # 运行Q-learning算法
        max_iterations = 1000  # 增加到1000次迭代
        for iteration in range(1, max_iterations + 1):
            # 随机选择一个非终点状态作为起始状态
            start_states = [(i, j) for i in range(grid_size) for j in range(grid_size) if (i, j) != goal_state]
            state = random.choice(start_states)
            
            # 执行一个回合，直到到达终点或者执行了足够多的步骤
            max_steps = 100
            for step in range(max_steps):
                # 使用ε-greedy策略选择动作
                valid_actions = [a for a in actions if is_valid_action(state, a)]
                
                if random.random() < epsilon:
                    # 探索：随机选择一个有效动作
                    action = random.choice(valid_actions)
                else:
                    # 利用：选择当前状态下Q值最大的动作
                    # 在Q值相同的情况下随机选一个，避免偏好特定动作
                    best_value = float('-inf')
                    best_actions = []
                    
                    for a in valid_actions:
                        q_value = q_table[state][a]
                        if q_value > best_value:
                            best_value = q_value
                            best_actions = [a]
                        elif q_value == best_value:
                            best_actions.append(a)
                    
                    action = random.choice(best_actions)
                
                # 执行动作，获取下一个状态和奖励
                next_state = get_next_state(state, action)
                reward = get_reward(state, next_state)
                
                # 获取下一状态可能的最大Q值
                if next_state in q_table and q_table[next_state]:
                    max_next_q = max(q_table[next_state].values())
                else:
                    max_next_q = 0.0
                
                # 更新Q值
                q_table[state][action] += alpha * (reward + gamma * max_next_q - q_table[state][action])
                
                # 更新状态
                state = next_state
                
                # 如果到达终点，结束当前回合
                if state == goal_state:
                    break
            
            # 记录指定迭代次数的Q表
            if iteration in iterations_to_record:
                q_tables_history.append((iteration, {s: q_table[s].copy() for s in states}))
        
        # 将Q表转换为可视化图形
        figures = []
        
        for iteration, q_table_snapshot in q_tables_history:
            fig, ax = plt.subplots(figsize=(5, 4))  # 减小尺寸
            
            # 标记终点
            ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=True, facecolor='lightgreen', alpha=0.3))
            ax.text(0.5, 0.5, "目标", ha='center', va='center')
            
            # 添加最优动作箭头
            for i in range(grid_size):
                for j in range(grid_size):
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black'))
                    
                    state = (i, j)
                    
                    # 跳过终点
                    if state == goal_state:
                        continue
                    
                    # 检查状态是否在Q表中
                    if state in q_table_snapshot and q_table_snapshot[state]:
                        # 找出最大Q值对应的动作
                        valid_actions = [a for a in q_table_snapshot[state].keys()]
                        if valid_actions:
                            best_action = max(valid_actions, key=lambda a: q_table_snapshot[state][a])
                            best_q_value = q_table_snapshot[state][best_action]
                            
                            # 只显示非零值或非极小值
                            if best_q_value > 0.01:
                                arrows = ['↑', '→', '↓', '←']
                                
                                # 显示最优动作箭头
                                ax.text(j+0.5, i+0.5, arrows[best_action], ha='center', va='center', 
                                        fontsize=15, color='red', fontweight='bold')
                                
                                # 显示最大Q值
                                ax.text(j+0.5, i+0.25, f"{best_q_value:.1f}", ha='center', va='center', 
                                        fontsize=8, color='black')
            
            ax.set_xlim(0, grid_size)
            ax.set_ylim(0, grid_size)
            ax.invert_yaxis()  # 反转y轴，使(0,0)在左上角
            ax.set_aspect('equal')
            ax.set_title(f'Q-learning学习过程 (迭代 {iteration})')
            ax.set_xticks(np.arange(grid_size) + 0.5)
            ax.set_yticks(np.arange(grid_size) + 0.5)
            ax.set_xticklabels([f"{i}" for i in range(grid_size)])
            ax.set_yticklabels([f"{i}" for i in range(grid_size)])
            ax.set_xticks(np.arange(0, grid_size + 1), minor=True)
            ax.set_yticks(np.arange(0, grid_size + 1), minor=True)
            ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
            ax.tick_params(which="minor", size=0)
            
            fig.tight_layout()
            
            # 将图像转换为SVG
            buf = io.BytesIO()
            fig.savefig(buf, format='svg')
            buf.seek(0)
            svg_data = buf.getvalue().decode('utf-8')
            buf.close()
            plt.close(fig)
            
            figures.append((iteration, svg_data))
        
        return figures
    
    # 获取Q-learning过程的所有图
    q_learning_figures = create_q_learning_process()
    
    # 创建选择器来显示不同迭代阶段
    selected_iteration = st.select_slider(
        "选择迭代次数以查看Q表的演变过程",
        options=iterations_to_record
    )
    
    # 显示选择的迭代阶段的图
    for iteration, svg_data in q_learning_figures:
        if iteration == selected_iteration:
            st.components.v1.html(svg_data, height=380, width=450)
            
            # 添加对应阶段的解释
            if iteration == 0:
                st.markdown("""
                **初始状态 (迭代0)**:
                - Q表所有值初始化为0
                - 智能体没有任何知识
                - 还不知道哪些动作更好
                """)
            elif iteration == 5:
                st.markdown("""
                **早期学习 (迭代5)**:
                - 靠近目标的几个状态开始获得正向Q值
                - 价值开始从目标逐渐向外传播
                - 大多数状态仍未被探索
                """)
            elif iteration == 20:
                st.markdown("""
                **持续学习 (迭代20)**:
                - 更多状态获得有意义的Q值
                - 开始形成一条指向目标的路径
                - 远离目标的状态价值仍较低
                """)
            elif iteration == 100:
                st.markdown("""
                **扩展学习 (迭代100)**:
                - 价值传播到更远的状态
                - 最优路径变得更加清晰
                - 大部分状态已被探索
                """)
            elif iteration == 500:
                st.markdown("""
                **接近收敛 (迭代500)**:
                - 几乎所有状态都有意义的Q值
                - 清晰的价值梯度从远处指向目标
                - 最优策略基本确定
                """)
            elif iteration == 1000:
                st.markdown("""
                **完全收敛 (迭代1000)**:
                - Q表收敛到稳定值
                - 价值梯度清晰地指向目标
                - 从任意位置都能找到通往目标的最优路径
                """)
    
    # 总结
    st.markdown("""
    ## 结论
    
    虽然从演示来看V表(状态价值函数)计算看起来更加直观，但Q表(状态-动作价值函数)在实际应用中更为普遍，原因是：
    
    1. **无需环境模型**: Q-learning可以直接从经验中学习，无需知道环境的转移函数和奖励函数
    
    2. **直接获取策略**: Q表直接告诉智能体在每个状态应该采取什么动作
    
    3. **适应性更强**: 可以应用于复杂、随机且部分可观测的环境
    
    4. **端到端学习**: 无需分离策略评估和策略改进步骤
    
    ### 关于本演示的说明
    
    本页面中的Q-learning学习过程是**真实的算法执行结果**，而不是预设的模拟数据：
    
    - 我们实现了一个4×4网格世界，终点位于(0,0)位置
    - 每次移动的奖励为-1，到达终点的奖励为+10
    - 使用标准的Q-learning算法，通过多次尝试学习最优策略
    - 您可以观察到价值如何从终点开始向外传播，最终形成一个指向终点的梯度场
    - 这正是强化学习的核心思想：通过经验逐步学习到最优策略
    
    ### 实际应用中的挑战
    
    现代强化学习面临的主要挑战是状态空间太大，无法使用表格存储所有值，因此出现了函数近似方法：
    
    - **深度Q网络(DQN)**: 使用神经网络近似Q函数，而不是使用表格
    - **策略梯度方法**: 直接学习策略，而不是通过价值函数间接获取
    - **Actor-Critic方法**: 结合策略学习和价值估计的优点
    
    这些方法都建立在理解V表和Q表基本原理的基础上，但能够处理更复杂的问题。
    """)

if __name__ == "__main__":
    show() 