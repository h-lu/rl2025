import streamlit as st
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns

# 添加父目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.visualizations import plot_value_function

def show():
    """显示价值函数页面"""
    st.title("价值函数 (Value Function)")
    
    st.info("""
    价值函数用于评估在给定状态或状态-动作对下，未来预期累积奖励的期望值。价值函数是强化学习算法的核心概念之一。
    """)
    
    # 价值函数介绍
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### V 函数 (状态价值函数)
        
        - $V_{\pi}(s)$ 表示在策略 $\pi$ 下，从状态 $s$ 出发，未来可以获得的期望累积奖励
        - V 函数评估的是**状态的价值**，即处于某个状态的好坏程度
        - V 函数越大，表示当前状态越好，未来可以获得的期望奖励越高
        """)
        
        # 状态价值函数的数学表达式
        st.latex(r"V_{\pi}(s) = \mathbb{E}_{\pi}\left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} | S_t = s \right]")
    
    with col2:
        st.markdown("""
        ### Q 函数 (动作价值函数)
        
        - $Q_{\pi}(s, a)$ 表示在策略 $\pi$ 下，从状态 $s$ 出发，选择动作 $a$ 后的期望累积奖励
        - Q 函数评估的是**状态-动作对的价值**
        - Q 函数越大，表示在当前状态下，该动作越好
        """)
        
        # 动作价值函数的数学表达式
        st.latex(r"Q_{\pi}(s, a) = \mathbb{E}_{\pi}\left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} | S_t = s, A_t = a \right]")
    
    # V函数和Q函数的关系
    st.subheader("V 函数和 Q 函数的关系")
    
    st.latex(r"V_{\pi}(s) = \sum_{a \in A} \pi(a|s) Q_{\pi}(s, a)")
    
    st.markdown("""
    这个等式表明，状态价值是在该状态下所有可能动作的动作价值的加权平均，权重是策略 $\pi$ 选择各个动作的概率。
    """)
    
    # 贝尔曼方程
    st.subheader("贝尔曼方程")
    
    st.markdown("""
    贝尔曼方程是强化学习中的核心等式，描述了当前状态的价值与下一个状态价值之间的关系。
    """)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("#### V 函数的贝尔曼方程")
        st.latex(r"V_{\pi}(s) = \sum_{a} \pi(a|s) \sum_{s',r} p(s',r|s,a)[r + \gamma V_{\pi}(s')]")
    
    with col4:
        st.markdown("#### Q 函数的贝尔曼方程")
        st.latex(r"Q_{\pi}(s,a) = \sum_{s',r} p(s',r|s,a)[r + \gamma \sum_{a'} \pi(a'|s') Q_{\pi}(s',a')]")
    
    # 最优价值函数
    st.subheader("最优价值函数")
    
    st.markdown("""
    强化学习的目标是找到最优策略，对应的最优价值函数定义为:
    """)
    
    col5, col6 = st.columns(2)
    
    with col5:
        st.markdown("#### 最优状态价值函数")
        st.latex(r"V^*(s) = \max_{\pi} V_{\pi}(s) = \max_{a} Q^*(s,a)")
    
    with col6:
        st.markdown("#### 最优动作价值函数")
        st.latex(r"Q^*(s,a) = \max_{\pi} Q_{\pi}(s,a) = \mathbb{E}\left[R_{t+1} + \gamma \max_{a'} Q^*(S_{t+1},a') | S_t=s, A_t=a\right]")
    
    st.markdown("""
    如果我们知道最优价值函数，最优策略可以通过选择每个状态下价值最高的动作得到:
    """)
    
    st.latex(r"\pi^*(s) = \arg\max_{a} Q^*(s,a)")
    
    # 交互式演示：价值函数在网格世界中的可视化
    st.subheader("交互式演示：价值函数热图")
    
    st.markdown("""
    下面是Grid World中一个状态价值函数的热图可视化。颜色越热（红色）表示价值越高，颜色越冷（蓝色）表示价值越低。
    目标状态通常具有最高的价值。
    """)
    
    # 添加一个滑块控制网格大小
    grid_size = st.slider("网格大小", min_value=3, max_value=10, value=5, step=1)
    
    # 添加一个滑块控制折扣因子
    gamma = st.slider("折扣因子 (γ)", min_value=0.0, max_value=1.0, value=0.9, step=0.1)
    
    # 生成随机价值函数
    def generate_value_function(size, gamma):
        """生成一个示例价值函数"""
        # 初始化价值函数
        values = np.zeros((size, size))
        
        # 设置目标位置（右下角）
        goal_value = 1.0
        values[size-1, size-1] = goal_value
        
        # 使用简化的值迭代来生成价值函数
        for _ in range(50):  # 简单迭代几次
            new_values = np.copy(values)
            for i in range(size):
                for j in range(size):
                    # 跳过目标状态
                    if i == size-1 and j == size-1:
                        continue
                    
                    # 计算四个方向的值
                    neighbors_values = []
                    for di, dj in [(-1, 0), (0, 1), (1, 0), (0, -1)]:  # 上右下左
                        ni, nj = i + di, j + dj
                        if 0 <= ni < size and 0 <= nj < size:
                            neighbors_values.append(values[ni, nj])
                        else:
                            neighbors_values.append(values[i, j])  # 如果出界，使用当前值
                    
                    # 更新价值函数（使用最大值，假设是最优策略）
                    new_values[i, j] = -0.1 + gamma * max(neighbors_values)
            
            values = new_values
        
        return values
    
    # 生成并显示价值函数热图
    value_function = generate_value_function(grid_size, gamma)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(value_function, annot=True, fmt=".2f", cmap="coolwarm", 
                cbar_kws={'label': '状态价值 V(s)'}, ax=ax)
    
    # 设置标题和轴标签
    ax.set_title(f"Grid World 状态价值函数 (γ={gamma})", fontsize=14)
    ax.set_xlabel("列坐标", fontsize=12)
    ax.set_ylabel("行坐标", fontsize=12)
    
    st.pyplot(fig)
    
    st.markdown(f"""
    上图展示了一个 {grid_size}×{grid_size} 的网格世界中的状态价值函数。目标位置在右下角，具有最高的价值。
    其他单元格的价值取决于它们到目标的距离以及折扣因子 γ={gamma}。
    
    注意到随着单元格离目标越远，其价值越低。这反映了强化学习中的折扣累积奖励概念：未来的奖励会被折扣，越远的奖励影响越小。
    """)
    
    # 价值函数在实际中的应用
    st.subheader("价值函数在实际中的应用")
    
    st.markdown("""
    价值函数在强化学习中有广泛的应用:
    
    1. **游戏AI**: 评估游戏中不同状态的优劣，指导AI选择最佳动作
    2. **机器人控制**: 学习最优控制策略，如平衡倒立摆
    3. **推荐系统**: 评估不同推荐动作的长期价值
    4. **资源分配**: 在有限资源下做出最优决策
    5. **医疗决策**: 在治疗过程中选择最佳干预措施
    
    价值函数是大多数强化学习算法的基础，包括:
    
    - **值迭代 (Value Iteration)**
    - **策略迭代 (Policy Iteration)**
    - **Q-Learning**
    - **Sarsa**
    - **深度Q网络 (DQN)**
    """)

if __name__ == "__main__":
    show() 