import streamlit as st
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# 添加父目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.visualizations import plot_mdp_diagram

def show():
    """显示MDP介绍页面"""
    st.title("强化学习框架与迷宫环境")
    
    st.markdown("""
    ## 本周学习目标
    
    - 理解马尔可夫决策过程 (MDP) 的基本思想
    - 掌握策略 (Policy)、价值函数 (Value Function) 的概念
    - 理解探索 (Exploration) 与利用 (Exploitation) 的平衡
    - 学习使用 Gymnasium 库搭建迷宫环境 (Grid World)
    - 掌握使用 AI 辅助工具进行代码补全和修改
    """)
    
    st.header("马尔可夫决策过程 (MDP)")
    
    st.info("""
    马尔可夫决策过程 (Markov Decision Process, MDP) 是强化学习的核心框架，用于形式化描述智能体与环境的交互过程。
    """)
    
    # 显示MDP示意图
    st.subheader("MDP 过程示意图")
    mdp_fig = plot_mdp_diagram()
    st.pyplot(mdp_fig)
    
    st.subheader("MDP 的核心要素")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 状态 (State, S)
        - 环境的描述，包含了智能体做出决策所需的信息
        - **马尔可夫性质**: 当前状态包含了所有历史信息，未来的状态只依赖于当前状态和动作，而与过去的历史无关
        - 在迷宫环境中，状态可以是智能体在迷宫中的位置坐标
        
        ### 动作 (Action, A)
        - 智能体在每个状态下可以采取的行为
        - 在迷宫环境中，动作可以是向上、下、左、右移动
        """)
    
    with col2:
        st.markdown("""
        ### 转移概率 (Transition Probability, P)
        - 智能体在状态 $s$ 采取动作 $a$ 后，转移到下一个状态 $s'$ 的概率
        - $P(s'|s, a)$ 表示在状态 $s$ 采取动作 $a$ 转移到状态 $s'$ 的概率
        - 在确定性迷宫环境中，转移概率是确定的；在非确定性环境中可能存在随机性
        
        ### 奖励 (Reward, R)
        - 智能体在与环境交互后获得的反馈信号，用于评价动作的好坏
        - $R(s, a, s')$ 表示在状态 $s$ 采取动作 $a$ 转移到状态 $s'$ 后获得的奖励
        - 在迷宫寻宝游戏中，到达宝藏获得正奖励，撞墙或陷阱获得负奖励
        """)
    
    st.markdown("""
    ### 策略 (Policy, $\pi$)
    - 智能体根据当前状态选择动作的规则，可以是确定性的或随机性的
    - $\pi(a|s)$ 表示在状态 $s$ 下选择动作 $a$ 的概率
    - 强化学习的目标是学习最优策略，使得智能体获得最大的累积奖励
    """)
    
    # 交互式演示：马尔可夫性质
    st.subheader("交互式演示：马尔可夫性质")
    
    st.markdown("""
    马尔可夫性质是强化学习的核心假设之一，它表示系统的下一个状态只依赖于当前状态和动作，而与过去的历史无关。
    
    下面是一个简单的例子：想象一个移动的小球，其未来位置只取决于当前位置和移动方向（动作），而不依赖于它是如何到达当前位置的。
    """)
    
    # 使用滑块创建交互式演示
    time_step = st.slider("时间步长", 0, 10, 5)
    
    # 创建一个简单的马尔可夫链轨迹示例
    np.random.seed(42)
    states = np.cumsum(np.random.normal(0, 1, 11))
    
    # 绘制轨迹
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(11), states, 'o-', color='blue', alpha=0.6)
    
    # 高亮显示当前时间步
    ax.plot(time_step, states[time_step], 'ro', markersize=10)
    
    # 添加标注
    ax.annotate(f"当前状态 $s_{time_step}$", 
               (time_step, states[time_step]),
               xytext=(time_step+0.5, states[time_step]+0.5),
               arrowprops=dict(facecolor='black', shrink=0.05))
    
    # 设置轴标签
    ax.set_xlabel("时间步")
    ax.set_ylabel("状态值")
    ax.set_title("马尔可夫链轨迹")
    ax.grid(True, linestyle='--', alpha=0.7)
    
    st.pyplot(fig)
    
    st.markdown(f"""
    在时间步 {time_step} 中，系统处于状态 {states[time_step]:.2f}。根据马尔可夫性质，系统在下一个时间步的状态只依赖于当前状态，
    而不依赖于它之前的轨迹。这大大简化了我们对系统的建模。
    """)
    
    st.subheader("本周课程内容")
    
    st.markdown("""
    1. **第一次课**：强化学习框架与迷宫环境 (Grid World) 搭建
       - 掌握 MDP 的基本概念
       - 理解价值函数的定义与作用
       - 学习探索与利用的平衡
       - 了解如何使用 Gymnasium 搭建迷宫环境
    
    2. **第二次课**：小组项目一：迷宫寻宝 (Grid World) 环境搭建
       - 独立完成迷宫环境的搭建
       - 扩展迷宫地图，设计更复杂的场景
       - 实现基本的环境渲染
       - 学习使用 AI 辅助工具提高开发效率
    """)
    
    # 添加相关资源
    st.subheader("相关学习资源")
    
    st.markdown("""
    - Sutton & Barto 强化学习教材第3章：马尔可夫决策过程
    - David Silver 强化学习课程第2讲：马尔可夫决策过程
    - Gymnasium 官方文档：[https://gymnasium.farama.org/](https://gymnasium.farama.org/)
    - PyGame 文档：[https://www.pygame.org/docs/](https://www.pygame.org/docs/)
    """)

if __name__ == "__main__":
    show() 