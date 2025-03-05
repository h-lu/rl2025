import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import plotly.express as px
import plotly.graph_objects as go
from utils import create_q_table_df, plot_heatmap

def simple_update_q_table(state, action, reward, next_state, next_actions, q_table, alpha, gamma):
    """简单实现Q值更新公式"""
    # 获取当前Q值
    current_q = q_table.get((state, action), 0.0)
    
    # 获取下一状态的最大Q值
    next_max_q = max([q_table.get((next_state, a), 0.0) for a in next_actions])
    
    # TD更新公式
    new_q = current_q + alpha * (reward + gamma * next_max_q - current_q)
    
    return new_q

def show():
    st.header("练习与探索")
    
    st.markdown("""
    本节包含一些互动练习，帮助巩固对Q-Learning的理解，并探索算法在不同参数下的表现。
    """)
    
    # 使用标签页分类不同的练习
    tab1, tab2, tab3 = st.tabs(["Q值更新练习", "最优策略提取", "探索率影响实验"])
    
    with tab1:
        st.subheader("练习1: Q值更新计算")
        
        st.markdown("""
        在这个练习中，你将亲自计算Q值的更新，并与正确答案比较。
        
        给定以下情境：
        - 你控制一个机器人，目标是找到最优路径
        - 当前状态(s): 机器人在位置A
        - 动作(a): 机器人向右移动
        - 奖励(r): 获得-1（表示每走一步有小的惩罚）
        - 下一状态(s'): 机器人到达位置B
        
        已知Q表中的当前值：
        - Q(A, 右) = 0.5
        - Q(B, 上) = 8.0
        - Q(B, 下) = 6.0
        - Q(B, 左) = 4.0
        - Q(B, 右) = 10.0
        
        参数：
        - 学习率(α) = 0.1
        - 折扣因子(γ) = 0.9
        """)
        
        # 创建交互式计算器
        st.markdown("### 交互式Q值更新计算器")
        
        # 设置计算所需的值
        current_q = 0.5
        reward = -1
        next_max_q = 10.0  # B状态中最大的Q值是Q(B, 右)
        
        # 用户输入更新的Q值
        user_new_q = st.number_input(
            "根据Q-Learning更新公式，计算Q(A, 右)的新值:",
            min_value=-10.0, max_value=10.0, value=0.5, step=0.1
        )
        
        # 计算正确答案
        alpha = 0.1
        gamma = 0.9
        correct_new_q = current_q + alpha * (reward + gamma * next_max_q - current_q)
        
        # 验证按钮
        if st.button("验证答案"):
            if abs(user_new_q - correct_new_q) < 0.01:
                st.success(f"恭喜！你的答案是正确的。新的Q值是 {correct_new_q:.4f}")
                
                # 解释计算过程
                st.markdown(f"""
                ### 计算过程详解
                
                Q-Learning的更新公式是：
                $$Q(s, a) \\leftarrow Q(s, a) + \\alpha [r + \\gamma \\max_{{a'}} Q(s', a') - Q(s, a)]$$
                
                按照我们的例子，计算过程如下：

                1. 当前Q值: Q(A, 右) = {current_q}
                2. 奖励: r = {reward}
                3. 下一状态的最大Q值: max Q(B, a') = {next_max_q}
                4. 学习率: α = {alpha}
                5. 折扣因子: γ = {gamma}
                
                套用公式:
                
                Q(A, 右) = {current_q} + {alpha} * ({reward} + {gamma} * {next_max_q} - {current_q})
                        = {current_q} + {alpha} * ({reward + gamma * next_max_q - current_q:.4f})
                        = {current_q} + {alpha * (reward + gamma * next_max_q - current_q):.4f}
                        = {correct_new_q:.4f}
                
                **这个更新的实际含义是什么？**
                
                简单来说，我们在调整Q(A, 右)的估计值:
                
                1. 发现了即时奖励是-1（每步的小惩罚）
                2. 但下一个状态(B)有一个很高的预期回报(10.0)
                3. 考虑到折扣因子(0.9)，未来奖励的当前价值是9.0
                4. 因此，尽管当前得到了负奖励，这个动作长期来看是值得的
                5. 我们将Q值从0.5调整到{correct_new_q:.4f}，表示对这个动作价值的更新估计
                """)
            else:
                st.error(f"不太对。正确答案是 {correct_new_q:.4f}。再试一次！")
        
        # 示意图
        st.markdown("### Q值更新示意图")
        st.markdown("""
        下面的示意图展示了更新过程中考虑的各个元素：
        
        <img src="./images/td_update.svg" width="600"/>
        
        对比公式和图示，你能更直观地理解Q值如何在学习过程中不断调整。
        """, unsafe_allow_html=True)
    
    with tab2:
        st.subheader("练习2: 从Q表提取最优策略")
        
        st.markdown("""
        在这个练习中，你将学习如何从一个已经训练好的Q表中提取最优策略。
        
        最优策略(π*)告诉智能体在每个状态下应该采取什么动作，以获得最大的长期奖励。
        
        提取策略的原则：在每个状态s，选择Q值最大的动作a。
        
        $$π^*(s) = \\arg\\max_a Q(s,a)$$
        """)
        
        # 创建一个示例Q表
        states = ["起点", "岔路", "隧道", "桥梁", "终点前"]
        actions = ["上", "下", "左", "右"]
        
        example_q_table = {
            ("起点", "上"): -1.0, ("起点", "下"): 2.5, ("起点", "左"): 0.8, ("起点", "右"): 1.2,
            ("岔路", "上"): 3.2, ("岔路", "下"): 1.5, ("岔路", "左"): -0.5, ("岔路", "右"): 0.7,
            ("隧道", "上"): 0.3, ("隧道", "下"): -2.0, ("隧道", "左"): 5.1, ("隧道", "右"): 2.2,
            ("桥梁", "上"): 4.6, ("桥梁", "下"): 1.7, ("桥梁", "左"): 3.4, ("桥梁", "右"): 7.8,
            ("终点前", "上"): 9.2, ("终点前", "下"): 5.3, ("终点前", "左"): 6.4, ("终点前", "右"): 4.1,
        }
        
        # 显示Q表
        q_df = create_q_table_df(states, actions, example_q_table)
        st.markdown("### 示例Q表")
        st.dataframe(q_df, use_container_width=True)
        
        # 显示Q表热力图
        st.markdown("### Q表热力图")
        fig = plot_heatmap(q_df, "Q值分布")
        st.plotly_chart(fig, use_container_width=True)
        
        # 提取最优策略的交互式练习
        st.markdown("### 提取最优策略")
        st.markdown("对于每个状态，选择你认为最优的动作（Q值最高的动作）：")
        
        user_policy = {}
        for state in states:
            user_action = st.selectbox(
                f"状态 '{state}' 的最优动作:", 
                actions,
                key=f"policy_{state}"
            )
            user_policy[state] = user_action
        
        # 计算正确答案
        correct_policy = {}
        for state in states:
            correct_policy[state] = max(actions, key=lambda a: example_q_table.get((state, a), 0.0))
        
        # 验证按钮
        if st.button("检查策略"):
            correct_count = sum([1 for state in states if user_policy[state] == correct_policy[state]])
            
            if correct_count == len(states):
                st.success("太棒了！你的策略完全正确！")
            else:
                st.warning(f"你的策略部分正确，得分: {correct_count}/{len(states)}")
            
            # 显示正确策略
            correct_policy_data = []
            for state in states:
                correct_action = correct_policy[state]
                q_value = example_q_table.get((state, correct_action), 0.0)
                user_action = user_policy[state]
                user_q_value = example_q_table.get((state, user_action), 0.0)
                is_correct = user_action == correct_action
                
                correct_policy_data.append({
                    "状态": state,
                    "你的选择": user_action,
                    "你选择的Q值": user_q_value,
                    "最优动作": correct_action,
                    "最优Q值": q_value,
                    "是否正确": "✓" if is_correct else "✗"
                })
            
            correct_df = pd.DataFrame(correct_policy_data)
            st.table(correct_df)
            
            # 可视化比较
            if correct_count < len(states):
                st.markdown("### 你的策略与最优策略的Q值比较")
                
                comparison_data = []
                for state in states:
                    opt_action = correct_policy[state]
                    opt_q = example_q_table.get((state, opt_action), 0.0)
                    usr_action = user_policy[state]
                    usr_q = example_q_table.get((state, usr_action), 0.0)
                    
                    if opt_action != usr_action:
                        comparison_data.append({
                            "状态": state,
                            "策略类型": "最优策略",
                            "Q值": opt_q
                        })
                        comparison_data.append({
                            "状态": state,
                            "策略类型": "你的策略",
                            "Q值": usr_q
                        })
                
                if comparison_data:
                    comp_df = pd.DataFrame(comparison_data)
                    fig = px.bar(
                        comp_df, 
                        x="状态", 
                        y="Q值", 
                        color="策略类型",
                        barmode="group",
                        title="不正确状态的策略比较"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("练习3: 探索率(ε)影响实验")
        
        st.markdown("""
        在这个实验中，我们将探索不同的探索率(ε)如何影响Q-Learning的学习效果。
        
        探索率决定了智能体探索新动作与利用已知最优动作之间的平衡：
        - 高探索率: 智能体更倾向于尝试新动作，有助于发现潜在的更优策略
        - 低探索率: 智能体更倾向于选择当前最优动作，有助于稳定收敛
        
        我们将在一个简单的网格世界中，使用不同的探索率训练智能体，并比较学习曲线。
        """)
        
        # 创建一个简单的网格世界环境
        class GridWorld:
            def __init__(self):
                # 4x4网格，(0,0)是左上角，(3,3)是右下角
                self.rows = 4
                self.cols = 4
                self.state = (0, 0)  # 起始位置
                self.goal = (3, 3)   # 目标位置
                self.obstacles = [(1, 1), (2, 1), (1, 2)]  # 障碍位置
                
                self.actions = ["上", "下", "左", "右"]
                self.states = [(r, c) for r in range(self.rows) for c in range(self.cols) if (r, c) not in self.obstacles]
                
                # 奖励设置
                self.step_reward = -0.1  # 每一步的小惩罚
                self.goal_reward = 10.0  # 到达目标的奖励
                self.obstacle_reward = -5.0  # 撞到障碍的惩罚
                
            def reset(self):
                self.state = (0, 0)
                return self.state
            
            def step(self, action):
                r, c = self.state
                
                # 根据动作更新位置
                if action == "上" and r > 0:
                    r -= 1
                elif action == "下" and r < self.rows - 1:
                    r += 1
                elif action == "左" and c > 0:
                    c -= 1
                elif action == "右" and c < self.cols - 1:
                    c += 1
                
                # 检查是否是障碍
                if (r, c) in self.obstacles:
                    reward = self.obstacle_reward
                    # 碰到障碍物后不移动
                    r, c = self.state
                # 检查是否达到目标
                elif (r, c) == self.goal:
                    reward = self.goal_reward
                # 普通步骤
                else:
                    reward = self.step_reward
                
                self.state = (r, c)
                done = (r, c) == self.goal
                
                return self.state, reward, done
            
            def render(self):
                grid = np.zeros((self.rows, self.cols))
                
                # 标记障碍
                for r, c in self.obstacles:
                    grid[r, c] = -1
                
                # 标记目标
                grid[self.goal[0], self.goal[1]] = 2
                
                # 标记当前位置
                grid[self.state[0], self.state[1]] = 1
                
                return grid
        
        # 实验设置
        st.markdown("### 实验设置")
        
        col1, col2 = st.columns(2)
        
        with col1:
            epsilon_values = [0.1, 0.3, 0.5, 0.7, 0.9]
            selected_epsilons = st.multiselect(
                "选择要比较的探索率 (ε):",
                epsilon_values,
                default=[0.1, 0.5, 0.9]
            )
            
            num_episodes = st.slider("训练回合数:", 10, 1000, 500, 10)
        
        with col2:
            alpha = st.slider("学习率 (α):", 0.01, 1.0, 0.1, 0.01)
            gamma = st.slider("折扣因子 (γ):", 0.1, 1.0, 0.9, 0.1)
            runs_per_epsilon = st.slider("每个探索率的运行次数:", 1, 10, 3)
        
        # 运行实验按钮
        if st.button("运行实验"):
            if len(selected_epsilons) == 0:
                st.warning("请至少选择一个探索率。")
            else:
                # 进度条
                progress_bar = st.progress(0)
                
                # 创建环境
                env = GridWorld()
                
                # 存储每个探索率的结果
                all_results = {}
                
                # 为每个探索率运行多次实验
                total_runs = len(selected_epsilons) * runs_per_epsilon
                run_count = 0
                
                for epsilon in selected_epsilons:
                    epsilon_results = []
                    
                    for run in range(runs_per_epsilon):
                        # 初始化Q表
                        q_table = {}
                        for state in env.states:
                            for action in env.actions:
                                q_table[(state, action)] = 0.0
                        
                        # 训练
                        episode_rewards = []
                        for episode in range(num_episodes):
                            state = env.reset()
                            total_reward = 0
                            done = False
                            
                            while not done:
                                # ε-贪婪策略选择动作
                                if np.random.random() < epsilon:
                                    action = np.random.choice(env.actions)
                                else:
                                    # 选择当前状态下Q值最大的动作
                                    action = max(env.actions, key=lambda a: q_table.get((state, a), 0.0))
                                
                                # 执行动作
                                next_state, reward, done = env.step(action)
                                
                                # 更新Q值
                                next_max_q = max([q_table.get((next_state, a), 0.0) for a in env.actions])
                                current_q = q_table.get((state, action), 0.0)
                                new_q = current_q + alpha * (reward + gamma * next_max_q - current_q)
                                q_table[(state, action)] = new_q
                                
                                # 更新状态和累积奖励
                                state = next_state
                                total_reward += reward
                            
                            episode_rewards.append(total_reward)
                        
                        epsilon_results.append(episode_rewards)
                        run_count += 1
                        progress_bar.progress(run_count / total_runs)
                    
                    # 计算平均奖励
                    avg_rewards = np.mean(epsilon_results, axis=0)
                    all_results[epsilon] = avg_rewards
                
                # 绘制结果
                st.markdown("### 不同探索率的学习曲线")
                
                # 创建学习曲线图
                fig = go.Figure()
                
                for epsilon, rewards in all_results.items():
                    # 添加原始数据线
                    fig.add_trace(go.Scatter(
                        x=list(range(1, num_episodes + 1)),
                        y=rewards,
                        mode='lines',
                        name=f'ε = {epsilon}',
                        opacity=0.7
                    ))
                    
                    # 添加移动平均线（如果回合数足够多）
                    if num_episodes > 50:
                        window_size = min(50, num_episodes // 10)
                        smoothed = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                        fig.add_trace(go.Scatter(
                            x=list(range(window_size, num_episodes + 1)),
                            y=smoothed,
                            mode='lines',
                            line=dict(width=2, dash='dash'),
                            name=f'ε = {epsilon} (平滑)',
                            showlegend=False
                        ))
                
                fig.update_layout(
                    title="不同探索率(ε)的学习曲线对比",
                    xaxis_title="回合数",
                    yaxis_title="累积奖励",
                    legend_title="探索率(ε)",
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 分析结果
                st.markdown("### 实验结果分析")
                
                # 计算每个探索率的最终性能
                final_performance = {}
                for epsilon, rewards in all_results.items():
                    # 使用最后100回合的平均奖励作为最终性能
                    final_window = min(100, num_episodes // 5)
                    final_performance[epsilon] = np.mean(rewards[-final_window:])
                
                # 创建柱状图
                perf_df = pd.DataFrame({
                    '探索率(ε)': list(final_performance.keys()),
                    '最终平均奖励': list(final_performance.values())
                })
                
                fig_bar = px.bar(
                    perf_df,
                    x='探索率(ε)',
                    y='最终平均奖励',
                    title="不同探索率的最终性能比较",
                    color='探索率(ε)',
                    color_continuous_scale="Blues",
                )
                
                st.plotly_chart(fig_bar, use_container_width=True)
                
                # 文字分析
                best_epsilon = max(final_performance, key=final_performance.get)
                worst_epsilon = min(final_performance, key=final_performance.get)
                
                st.markdown(f"""
                #### 结果解读
                
                从实验结果可以看出：
                
                1. **最佳探索率**: ε = {best_epsilon} 在最终阶段取得了最好的性能，平均奖励为 {final_performance[best_epsilon]:.2f}
                2. **最差探索率**: ε = {worst_epsilon} 的表现相对较差，平均奖励为 {final_performance[worst_epsilon]:.2f}
                
                **探索与利用的权衡**:
                
                - **低探索率** (ε = 0.1): 更快地利用已知信息，但可能陷入局部最优
                - **高探索率** (ε = 0.9): 更全面地探索状态空间，但收敛较慢
                - **中等探索率**: 往往能在探索和利用之间取得良好的平衡
                
                **实际应用建议**:
                
                在实际应用中，一个常见的策略是使用衰减的探索率——在学习初期使用高探索率充分探索环境，随着学习的进行逐渐降低探索率，专注于利用已学到的知识。
                """)
        else:
            st.info("点击「运行实验」按钮开始比较不同探索率的效果。")

if __name__ == "__main__":
    show() 