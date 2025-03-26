"""
可视化工具模块

提供DQN训练过程和结果的可视化功能
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.animation import FuncAnimation
import io
import base64
from IPython.display import HTML
import altair as alt
import matplotlib as mpl
import platform

def configure_matplotlib_fonts():
    """配置matplotlib使用系统中文字体"""
    # 对于不同操作系统设置不同的字体
    system = platform.system()
    
    if system == 'Windows':
        font_list = ['Microsoft YaHei', 'SimHei', 'SimSun']
    elif system == 'Darwin':  # macOS
        font_list = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Apple LiGothic']
    else:  # Linux等
        font_list = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Droid Sans Fallback']
    
    # 添加通用字体作为后备
    font_list.extend(['DejaVu Sans', 'Arial Unicode MS', 'sans-serif'])
    
    # 设置字体
    mpl.rcParams['font.sans-serif'] = font_list
    mpl.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    
    # 打印当前使用的字体族，用于调试
    print(f"当前使用的matplotlib字体族: {mpl.rcParams['font.sans-serif']}")
    
    # 返回成功信息
    return "matplotlib字体配置完成"

# 确保在导入时就配置好字体
configure_matplotlib_fonts()

def plot_training_progress(scores, avg_scores, epsilon_history=None):
    """
    绘制训练进度图表
    
    参数:
        scores (list): 每个回合的分数
        avg_scores (list): 移动平均分数
        epsilon_history (list, optional): epsilon值的历史记录
        
    返回:
        fig: matplotlib图表对象
    """
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # 绘制分数
    color = 'tab:blue'
    ax1.set_xlabel('回合数')
    ax1.set_ylabel('分数', color=color)
    ax1.plot(scores, alpha=0.3, color=color, label='回合分数')
    ax1.plot(avg_scores, color=color, label='平均分数(最近100回合)')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # 如果提供了epsilon历史记录，也绘制它
    if epsilon_history is not None:
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Epsilon', color=color)
        ax2.plot(epsilon_history, color=color, label='Epsilon')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim([0, 1.1])
    
    fig.tight_layout()
    lines1, labels1 = ax1.get_legend_handles_labels()
    
    if epsilon_history is not None:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    else:
        ax1.legend(lines1, labels1, loc='upper left')
    
    plt.title('DQN训练进度')
    
    return fig

def plot_loss_history(loss_history):
    """
    绘制损失历史图表
    
    参数:
        loss_history (list): 损失值历史记录
        
    返回:
        fig: matplotlib图表对象
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # 使用指数移动平均平滑损失曲线
    if len(loss_history) > 0:
        df = pd.DataFrame({'loss': loss_history})
        smoothed_loss = df['loss'].ewm(span=100).mean()
        
        ax.plot(loss_history, alpha=0.2, color='tab:blue', label='损失')
        ax.plot(smoothed_loss, color='tab:blue', label='平滑损失')
        
        ax.set_xlabel('训练步数')
        ax.set_ylabel('损失值')
        ax.set_title('DQN训练损失')
        ax.legend()
    
    return fig

def plot_q_values(states, q_network):
    """
    绘制Q值分布
    
    参数:
        states (ndarray): 状态数组
        q_network: Q网络模型
        
    返回:
        fig: matplotlib图表对象
    """
    q_values = q_network.predict(states, verbose=0)
    q_mean = np.mean(q_values, axis=0)
    q_std = np.std(q_values, axis=0)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.arange(len(q_mean))
    ax.bar(x, q_mean, yerr=q_std, align='center', alpha=0.7, ecolor='black', capsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(['向左', '向右'])
    ax.set_ylabel('Q值')
    ax.set_title('动作Q值分布')
    ax.yaxis.grid(True)
    
    return fig

def create_altair_training_chart(scores, avg_scores, epsilon_history=None):
    """
    使用Altair创建训练进度交互式图表
    
    参数:
        scores (list): 每个回合的分数
        avg_scores (list): 移动平均分数
        epsilon_history (list, optional): epsilon值的历史记录
        
    返回:
        chart: Altair图表对象
    """
    # 创建数据框
    data = pd.DataFrame({
        'Episode': range(len(scores)),
        'Score': scores,
        'Avg Score': avg_scores
    })
    
    # 转换为长格式，便于Altair处理
    data_melted = pd.melt(
        data, 
        id_vars=['Episode'], 
        value_vars=['Score', 'Avg Score'],
        var_name='Metric', 
        value_name='Value'
    )
    
    # 创建基本图表
    chart = alt.Chart(data_melted).mark_line().encode(
        x='Episode:Q',
        y='Value:Q',
        color='Metric:N',
        tooltip=['Episode:Q', 'Value:Q', 'Metric:N']
    ).properties(
        width=700,
        height=400,
        title='DQN训练进度'
    ).interactive()
    
    # 如果有epsilon历史记录，添加到图表中
    if epsilon_history is not None:
        epsilon_data = pd.DataFrame({
            'Episode': range(len(epsilon_history)),
            'Epsilon': epsilon_history
        })
        
        epsilon_chart = alt.Chart(epsilon_data).mark_line(color='red').encode(
            x='Episode:Q',
            y=alt.Y('Epsilon:Q', scale=alt.Scale(domain=[0, 1])),
            tooltip=['Episode:Q', 'Epsilon:Q']
        )
        
        # 创建双轴图表
        chart = alt.layer(chart, epsilon_chart).resolve_scale(y='independent')
    
    return chart

def create_animated_gif(frames, filename, fps=30):
    """
    从帧序列创建动画GIF
    
    参数:
        frames (list): 帧列表
        filename (str): 输出文件名
        fps (int): 每秒帧数
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    patch = plt.imshow(frames[0])
    plt.axis('off')
    
    def animate(i):
        patch.set_data(frames[i])
        return [patch]
    
    anim = FuncAnimation(
        fig, animate, frames=len(frames),
        interval=1000/fps, blit=True
    )
    
    anim.save(filename, writer='pillow', fps=fps)
    plt.close()

def visualize_state_action_values(q_values, action_descriptions):
    """
    可视化状态-动作值
    
    参数:
        q_values (ndarray): Q值数组
        action_descriptions (dict): 动作描述字典
        
    返回:
        fig: matplotlib图表对象
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    actions = list(action_descriptions.values())
    ax.bar(range(len(q_values)), q_values, tick_label=actions)
    ax.set_ylabel('Q值')
    ax.set_title('状态-动作值')
    
    # 标出最大Q值
    max_idx = np.argmax(q_values)
    max_value = q_values[max_idx]
    ax.get_children()[max_idx].set_color('g')
    
    ax.text(max_idx, max_value + 0.1, f'最优动作: {actions[max_idx]}', 
            ha='center', va='bottom', fontweight='bold')
    
    return fig

def create_experience_replay_figure():
    """创建经验回放示意图"""
    # 创建经验回放示意图
    fig, ax = plt.subplots(figsize=(10, 6))
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
    
    return fig

def create_target_network_figure():
    """创建目标网络示意图"""
    # 创建目标网络与Q网络示意图
    fig, ax = plt.subplots(figsize=(10, 6))
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
    
    return fig

def create_streamlit_tabs_for_concepts():
    """
    为DQN核心概念创建Streamlit标签页
    """
    tab1, tab2, tab3 = st.tabs(["经验回放", "目标网络", "ε-贪婪策略"])
    
    with tab1:
        st.markdown("""
        ### 经验回放 (Experience Replay)
        
        经验回放是DQN算法的关键创新之一，它将智能体的经验存储在一个回放缓冲区中，然后随机采样来训练网络。
        
        **主要优势:**
        - **打破数据相关性**：连续采样的状态通常高度相关，随机采样可以打破这种相关性
        - **提高数据效率**：每个经验可以被多次使用
        - **平滑数据分布**：减少学习过程中的波动
        
        **实现方式:**
        ```python
        class ReplayBuffer:
            def __init__(self, capacity):
                self.buffer = deque(maxlen=capacity)
                
            def add(self, state, action, reward, next_state, done):
                self.buffer.append((state, action, reward, next_state, done))
                
            def sample(self, batch_size):
                return random.sample(self.buffer, batch_size)
        ```
        """)
        
        # 使用matplotlib创建的图像
        fig = create_experience_replay_figure()
        st.pyplot(fig)
        st.caption("经验回放示意图")
    
    with tab2:
        st.markdown("""
        ### 目标网络 (Target Network)
        
        目标网络是DQN的另一个关键创新，用于提高训练的稳定性。
        
        **主要优势:**
        - **稳定目标**：通过使用单独的网络计算目标值，避免了移动目标问题
        - **减少相关性**：目标网络参数更新频率较低，减少了当前Q值和目标Q值之间的相关性
        - **防止震荡**：防止Q值估计的剧烈波动
        
        **实现方式:**
        ```python
        # 初始化目标网络和主网络
        self.q_network = create_model()
        self.target_network = create_model()
        
        # 定期将主网络权重复制到目标网络
        if self.t_step % update_target_every == 0:
            self.target_network.set_weights(self.q_network.get_weights())
        ```
        """)
        
        # 使用matplotlib创建的图像
        fig = create_target_network_figure()
        st.pyplot(fig)
        st.caption("目标网络示意图")
    
    with tab3:
        st.markdown("""
        ### ε-贪婪策略 (ε-greedy Policy)
        
        ε-贪婪策略平衡了探索（尝试新动作）和利用（选择已知的最佳动作）。
        
        **工作原理:**
        - 以概率 ε 选择随机动作（探索）
        - 以概率 1-ε 选择当前估计的最优动作（利用）
        - 随着训练的进行，ε 值通常会从接近1逐渐降低到接近0
        
        **实现方式:**
        ```python
        def act(self, state):
            if random.random() > self.epsilon:
                # 利用：选择Q值最大的动作
                q_values = self.q_network.predict(state)
                return np.argmax(q_values[0])
            else:
                # 探索：随机选择动作
                return random.choice([0, 1])
                
        # 每次学习后更新epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        ```
        """)
        
        # 创建epsilon衰减图表
        epsilon = 1.0
        epsilon_end = 0.01
        epsilon_decay = 0.995
        epsilon_values = []
        for i in range(500):
            epsilon_values.append(epsilon)
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        chart_data = pd.DataFrame({
            'Episode': range(len(epsilon_values)),
            'Epsilon': epsilon_values
        })
        
        st.line_chart(chart_data.set_index('Episode'))

def display_dqn_code_segments():
    """
    显示DQN关键代码段
    """
    with st.expander("DQN网络模型"):
        st.code("""
def create_dqn_model(state_size, action_size, hidden_size=64):
    model = keras.Sequential([
        layers.Dense(hidden_size, activation='relu', input_shape=(state_size,)),
        layers.Dense(hidden_size, activation='relu'),
        layers.Dense(action_size, activation='linear')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                 loss='mse')
    return model
        """, language="python")
    
    with st.expander("经验回放缓冲区"):
        st.code("""
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)
        
        return (np.array(states), 
                np.array(actions), 
                np.array(rewards), 
                np.array(next_states), 
                np.array(dones))
        """, language="python")
    
    with st.expander("ε-贪婪动作选择"):
        st.code("""
def act(self, state, eval_mode=False):
    state = np.reshape(state, [1, self.state_size])
    
    if eval_mode or random.random() > self.epsilon:
        # 利用：选择Q值最大的动作
        action_values = self.q_network.predict(state, verbose=0)
        return np.argmax(action_values[0])
    else:
        # 探索：随机选择动作
        return random.choice(np.arange(self.action_size))
        """, language="python")
    
    with st.expander("DQN学习过程"):
        st.code("""
def learn(self, experiences):
    states, actions, rewards, next_states, dones = experiences
    
    # 从目标网络中获取下一个状态的最大Q值
    target_q_values = self.target_network.predict(next_states, verbose=0)
    max_target_q = np.max(target_q_values, axis=1)
    
    # 计算目标Q值
    targets = rewards + (self.gamma * max_target_q * (1 - dones))
    
    # 获取当前预测的Q值并更新目标
    target_f = self.q_network.predict(states, verbose=0)
    for i, action in enumerate(actions):
        target_f[i][action] = targets[i]
    
    # 训练Q网络
    self.q_network.fit(states, target_f, epochs=1, verbose=0)
    
    # 更新epsilon
    self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        """, language="python") 