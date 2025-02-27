import streamlit as st
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import altair as alt

# 添加父目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.visualizations import plot_exploration_exploitation

def show():
    """显示探索与利用的平衡页面"""
    st.title("探索与利用的平衡")
    
    st.warning("""
    探索 (Exploration) 与利用 (Exploitation) 是强化学习中一个核心的权衡问题。
    """)
    
    # 探索与利用的概念
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 探索 (Exploration)
        
        - 尝试**新的动作**，探索未知的状态和动作空间
        - 收集更多信息，发现潜在的更好策略
        - 可能在短期内获得较低的奖励
        - 有助于**避免陷入局部最优**
        
        #### 例子
        - 尝试新的餐厅，可能发现更好的美食
        - 探索新的投资机会，可能获得更高回报
        - 游戏中尝试新的策略，可能找到更有效的方法
        """)
    
    with col2:
        st.markdown("""
        ### 利用 (Exploitation)
        
        - 根据**已知经验**选择当前认为最优的动作
        - 利用已有知识，获取即时奖励
        - 保证短期内稳定的收益
        - 可能**错过更好的策略**
        
        #### 例子
        - 去你最喜欢的餐厅，能享受满意的美食
        - 投资已知表现良好的股票，获得稳定回报
        - 游戏中使用已知有效的策略，确保胜率
        """)
    
    # 探索-利用窘境
    st.subheader("探索-利用窘境")
    
    st.markdown("""
    探索-利用窘境 (Exploration-Exploitation Dilemma) 是强化学习中的一个基本问题：
    
    - **太多探索**：可能花费大量时间在无用的尝试上，错过了利用已知好策略的机会
    - **太多利用**：可能过早锁定在次优策略上，错过了发现最优策略的机会
    
    这个问题没有完美解决方案，但有多种常用策略来平衡探索和利用。
    """)
    
    # 可视化探索与利用的平衡
    st.subheader("探索与利用的平衡图示")
    fig = plot_exploration_exploitation()
    st.pyplot(fig)
    
    # 常见的探索策略
    st.subheader("常见的探索策略")
    
    tab1, tab2, tab3, tab4 = st.tabs(["$\epsilon$-greedy", "衰减的 $\epsilon$-greedy", "Softmax", "UCB"])
    
    with tab1:
        st.markdown("""
        ### $\epsilon$-greedy 策略
        
        $\epsilon$-greedy 策略是最简单也是最常用的探索策略之一：
        
        - 以概率 $\epsilon$ 随机选择动作 (探索)
        - 以概率 $1-\epsilon$ 选择当前最优动作 (利用)
        
        ```python
        def epsilon_greedy(q_values, epsilon):
            # 探索：以概率epsilon随机选择动作
            if np.random.random() < epsilon:
                return np.random.randint(len(q_values))
            # 利用：选择Q值最大的动作
            else:
                return np.argmax(q_values)
        ```
        
        **优点**：简单、易实现、有理论保证
        
        **缺点**：探索是完全随机的，不考虑动作的潜在价值
        """)
        
        # 交互式演示：$\epsilon$-greedy策略
        st.subheader("交互式演示：$\epsilon$-greedy策略")
        
        # 使用滑块调整epsilon值
        epsilon = st.slider("$\epsilon$ 值", 0.0, 1.0, 0.1, 0.1, key="epsilon_tab1")
        
        # 随机生成Q值
        np.random.seed(42)
        q_values = np.random.normal(0, 1, 4)
        
        # 创建DataFrame用于展示Q值
        q_df = pd.DataFrame({
            '动作': ['上', '右', '下', '左'],
            'Q值': q_values
        })
        
        # 展示Q值表格
        st.dataframe(q_df)
        
        # 执行多次$\epsilon$-greedy选择，并统计结果
        n_trials = 1000
        action_counts = np.zeros(4)
        
        for _ in range(n_trials):
            if np.random.random() < epsilon:
                action = np.random.randint(4)
            else:
                action = np.argmax(q_values)
            action_counts[action] += 1
        
        # 创建DataFrame用于展示选择频率
        freq_df = pd.DataFrame({
            '动作': ['上', '右', '下', '左'],
            '选择频率': action_counts / n_trials,
            '探索/利用': ['利用' if i == np.argmax(q_values) else '探索' for i in range(4)]
        })
        
        # 使用Altair创建可视化
        chart = alt.Chart(freq_df).mark_bar().encode(
            x=alt.X('动作:N', title='动作'),
            y=alt.Y('选择频率:Q', title='选择频率'),
            color=alt.Color('探索/利用:N', scale=alt.Scale(domain=['探索', '利用'], 
                                                range=['#ff7f0e', '#1f77b4']))
        ).properties(width=400, height=300, title=f'$\epsilon$-greedy策略 (ε={epsilon})的动作选择频率')
        
        st.altair_chart(chart, use_container_width=True)
        
        st.markdown(f"""
        在上面的演示中，我们使用 $\epsilon={epsilon}$ 的 $\epsilon$-greedy 策略选择动作。
        
        - 动作 "{q_df.iloc[np.argmax(q_values)]['动作']}" 具有最高的Q值 ({q_values[np.argmax(q_values)]:.4f})，是利用选择
        - 其他动作是探索选择
        - 当 $\epsilon={epsilon}$ 时，探索的理论概率为 {epsilon}，利用的理论概率为 {1-epsilon}
        - 在 {n_trials} 次试验中，"{q_df.iloc[np.argmax(q_values)]['动作']}" 的实际选择频率为 {freq_df.iloc[np.argmax(q_values)]['选择频率']:.4f}
        """)
    
    with tab2:
        st.markdown("""
        ### 衰减的 $\epsilon$-greedy 策略
        
        衰减的 $\epsilon$-greedy 策略是 $\epsilon$-greedy 的改进版：
        
        - 初始时 $\epsilon$ 值较大，鼓励探索
        - 随着学习的进行，$\epsilon$ 值逐渐减小，更多地利用已有知识
        - 实现了从"多探索少利用"到"少探索多利用"的平滑过渡
        
        ```python
        def decaying_epsilon_greedy(q_values, epsilon_start, epsilon_end, decay_rate, step):
            # 计算当前的epsilon值
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-decay_rate * step)
            
            # 基于当前epsilon执行epsilon-greedy策略
            if np.random.random() < epsilon:
                return np.random.randint(len(q_values))
            else:
                return np.argmax(q_values)
        ```
        
        **优点**：平衡了探索和利用的动态需求，适合大多数强化学习任务
        
        **缺点**：需要调整额外的超参数（初始epsilon、最终epsilon、衰减率）
        """)
        
        # 交互式演示：衰减的$\epsilon$-greedy策略
        st.subheader("交互式演示：衰减的$\epsilon$-greedy策略")
        
        col_eps1, col_eps2, col_decay = st.columns(3)
        
        with col_eps1:
            epsilon_start = st.number_input("初始 $\epsilon$ 值", 0.0, 1.0, 1.0, 0.1)
        
        with col_eps2:
            epsilon_end = st.number_input("最终 $\epsilon$ 值", 0.0, 1.0, 0.01, 0.01)
        
        with col_decay:
            decay_rate = st.number_input("衰减率", 0.001, 0.5, 0.01, 0.001)
        
        # 生成不同时间步的epsilon值
        steps = np.arange(0, 500)
        epsilons = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-decay_rate * steps)
        
        # 创建DataFrame
        epsilon_df = pd.DataFrame({
            '时间步': steps,
            'Epsilon值': epsilons
        })
        
        # 使用Altair创建可视化
        chart = alt.Chart(epsilon_df).mark_line().encode(
            x=alt.X('时间步:Q', title='时间步'),
            y=alt.Y('Epsilon值:Q', title='Epsilon值', scale=alt.Scale(domain=[0, 1]))
        ).properties(width=600, height=300, title='衰减的$\epsilon$-greedy策略')
        
        st.altair_chart(chart, use_container_width=True)
        
        st.markdown(f"""
        上图展示了衰减的 $\epsilon$-greedy 策略中 $\epsilon$ 值随时间的变化。
        
        - 初始 $\epsilon$ 值为 {epsilon_start}，表示在学习初期高度探索
        - 最终 $\epsilon$ 值为 {epsilon_end}，表示在学习后期仍保留少量探索
        - 衰减率为 {decay_rate}，控制 $\epsilon$ 值下降的速度
        
        这种衰减策略允许智能体在开始时广泛探索环境，随着对环境了解的增加，逐渐转向更多地利用已获得的知识。
        """)
    
    with tab3:
        st.markdown("""
        ### Softmax 策略（玻尔兹曼探索）
        
        Softmax策略根据动作的估计价值对探索进行加权，而不是像$\epsilon$-greedy那样完全随机探索：
        
        - 使用Softmax函数将Q值转换为选择概率
        - 温度参数 $\\tau$ 控制探索的程度（高温更随机，低温更确定）
        - 价值更高的动作有更高的被选择概率，即使在探索时
        
        ```python
        def softmax(q_values, temperature):
            # 避免数值溢出
            q_exp = np.exp((q_values - np.max(q_values)) / temperature)
            probabilities = q_exp / np.sum(q_exp)
            
            # 根据概率选择动作
            return np.random.choice(len(q_values), p=probabilities)
        ```
        
        **优点**：考虑了动作的相对价值，比$\epsilon$-greedy更有针对性地探索
        
        **缺点**：对Q值的比例较敏感，需要仔细调整温度参数
        """)
        
        # 交互式演示：Softmax策略
        st.subheader("交互式演示：Softmax策略")
        
        # 使用滑块调整温度
        temperature = st.slider("温度 (τ)", 0.01, 2.0, 0.5, 0.01, key="temperature_tab3")
        
        # 随机生成Q值
        np.random.seed(42)
        q_values = np.random.normal(0, 1, 4)
        
        # 创建DataFrame用于展示Q值
        q_df = pd.DataFrame({
            '动作': ['上', '右', '下', '左'],
            'Q值': q_values
        })
        
        # 展示Q值表格
        st.dataframe(q_df)
        
        # 计算Softmax概率
        def softmax_probs(q_vals, temp):
            q_exp = np.exp((q_vals - np.max(q_vals)) / temp)
            return q_exp / np.sum(q_exp)
        
        probs = softmax_probs(q_values, temperature)
        
        # 创建DataFrame用于展示选择概率
        prob_df = pd.DataFrame({
            '动作': ['上', '右', '下', '左'],
            '选择概率': probs
        })
        
        # 使用Altair创建可视化
        chart = alt.Chart(prob_df).mark_bar().encode(
            x=alt.X('动作:N', title='动作'),
            y=alt.Y('选择概率:Q', title='选择概率')
        ).properties(width=400, height=300, title=f'Softmax策略 (τ={temperature})的动作选择概率')
        
        st.altair_chart(chart, use_container_width=True)
        
        st.markdown(f"""
        在上面的演示中，我们使用温度为 τ={temperature} 的Softmax策略计算动作的选择概率。
        
        - 动作 "{q_df.iloc[np.argmax(q_values)]['动作']}" 具有最高的Q值 ({q_values[np.argmax(q_values)]:.4f})，因此有最高的选择概率 ({probs[np.argmax(q_values)]:.4f})
        - 其他动作的选择概率按照它们的Q值比例分配
        - 较高的温度 (τ) 使概率分布更均匀，增加探索
        - 较低的温度 (τ) 使概率分布更集中于最佳动作，增加利用
        """)
    
    with tab4:
        st.markdown("""
        ### 上置信界 (Upper Confidence Bound, UCB)
        
        UCB策略通过考虑动作的不确定性来平衡探索和利用：
        
        - 每个动作的选择价值 = 估计的Q值 + 不确定性奖励
        - 不确定性奖励与动作的选择次数成反比
        - 较少选择的动作有更高的不确定性奖励，鼓励探索
        - 常用公式：$UCB(a) = Q(a) + c \\sqrt{{\\frac{{\\ln(t)}}{{N(a)}}}}$
        
        ```python
        def ucb(q_values, visit_counts, total_steps, c=2.0):
            # 避免除以零
            counts = np.maximum(visit_counts, 1e-10)
            
            # 计算UCB值
            ucb_values = q_values + c * np.sqrt(np.log(total_steps) / counts)
            
            # 选择UCB值最大的动作
            return np.argmax(ucb_values)
        ```
        
        **优点**：理论上有更好的保证，更有针对性地探索不确定的动作
        
        **缺点**：需要跟踪每个动作的访问次数，实现稍复杂
        """)
        
        # 交互式演示：UCB策略
        st.subheader("交互式演示：UCB策略")
        
        # 设置参数
        c_param = st.slider("探索参数 (c)", 0.1, 5.0, 2.0, 0.1, key="c_param_tab4")
        total_steps = st.slider("总时间步数", 10, 1000, 100, 10, key="total_steps_tab4")
        
        # 随机生成Q值
        np.random.seed(42)
        q_values = np.random.normal(0, 1, 4)
        
        # 随机生成访问次数
        visit_counts = np.random.randint(1, 50, 4)
        
        # 创建DataFrame展示基本数据
        data_df = pd.DataFrame({
            '动作': ['上', '右', '下', '左'],
            'Q值': q_values,
            '访问次数': visit_counts
        })
        
        # 展示数据表格
        st.dataframe(data_df)
        
        # 计算UCB值
        ucb_values = q_values + c_param * np.sqrt(np.log(total_steps) / visit_counts)
        
        # 创建DataFrame用于展示UCB值
        ucb_df = pd.DataFrame({
            '动作': ['上', '右', '下', '左'],
            'UCB值': ucb_values
        })
        
        # 创建长格式DataFrame用于堆叠条形图
        ucb_long_data = []
        for i, action in enumerate(['上', '右', '下', '左']):
            ucb_long_data.append({
                '动作': action,
                '值': q_values[i],
                '组成': 'Q值'
            })
            ucb_long_data.append({
                '动作': action,
                '值': c_param * np.sqrt(np.log(total_steps) / visit_counts[i]),
                '组成': '探索奖励'
            })
        
        ucb_long = pd.DataFrame(ucb_long_data)
        
        # 使用Altair创建堆叠条形图
        base = alt.Chart(ucb_long).mark_bar().encode(
            x=alt.X('动作:N', title='动作'),
            y=alt.Y('值:Q', title='值'),
            color=alt.Color('组成:N', scale=alt.Scale(domain=['Q值', '探索奖励'], 
                                                   range=['#1f77b4', '#ff7f0e']))
        ).properties(width=400, height=300, title=f'UCB策略 (c={c_param}, t={total_steps})的动作UCB值')
        
        # 添加标记最大UCB值的图层
        text = alt.Chart(ucb_df).mark_text(
            align='center',
            baseline='bottom',
            dy=-5
        ).encode(
            x='动作:N',
            y=alt.Y('UCB值:Q'),
            text=alt.Text('UCB值:Q', format='.2f')
        )
        
        chart = alt.layer(base, text)
        
        st.altair_chart(chart, use_container_width=True)
        
        best_action_idx = np.argmax(ucb_values)
        best_action = data_df.iloc[best_action_idx]['动作']
        
        st.markdown(f"""
        在上面的演示中，我们使用UCB策略计算每个动作的UCB值。
        
        UCB值由两部分组成：
        1. **Q值**：代表当前对动作价值的估计（利用部分）
        2. **探索奖励**：$c \\sqrt{{\\frac{{\\ln(t)}}{{N(a)}}}}$，反映动作的不确定性（探索部分）
        
        UCB策略会选择具有最高UCB值的动作，在这个例子中是动作 "{best_action}"。
        
        注意：
        - 访问次数少的动作有更高的探索奖励
        - 参数c控制探索的程度
        - 随着总时间步数t的增加，探索奖励也会增加，鼓励重新访问之前探索过的动作
        """)
    
    # 商业案例
    st.subheader("商业案例：探索与利用的平衡")
    
    st.markdown("""
    探索与利用的平衡不仅在强化学习中很重要，在商业决策中也有广泛应用：
    
    ### 新市场尝试 vs. 现有市场深耕
    
    #### 新市场尝试 (探索)
    - **进入新的市场领域**：拓展业务到新的地域或行业
    - **开发新的产品线**：投资研发创新产品
    - **拓展新的客户群体**：接触潜在的新用户群体
    
    #### 现有市场深耕 (利用)
    - **优化现有产品**：改进已有产品的性能和用户体验
    - **提高客户满意度**：加强现有客户关系和服务质量
    - **提升市场份额**：在已有市场中争取更多份额
    
    #### 平衡策略
    - **双元策略**：同时投资现有业务和新业务，如谷歌的70-20-10规则
    - **阶段性策略**：在不同发展阶段调整探索与利用的比例
    - **组合投资**：根据风险和回报构建投资组合
    """)
    
    # 实践建议
    st.subheader("在强化学习中平衡探索与利用的实践建议")
    
    st.markdown("""
    1. **在训练初期增加探索**：当对环境了解有限时，多探索有助于收集更多信息
    
    2. **随着训练进行减少探索**：当已经收集了足够的经验后，逐渐增加利用
    
    3. **根据任务特点选择合适的策略**：
       - 复杂环境中使用更高级的探索策略（如UCB或Softmax）
       - 简单环境中可以使用基本的$\epsilon$-greedy策略
    
    4. **持续监控探索的有效性**：如果学习停滞，可能需要临时增加探索
    
    5. **避免过早收敛**：保持一定程度的随机性，避免陷入局部最优
    """)

if __name__ == "__main__":
    show() 