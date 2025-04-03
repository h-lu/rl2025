import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import torch
from matplotlib.colors import LinearSegmentedColormap
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

def smooth_curve(points, factor=0.9):
    """
    使用指数移动平均平滑曲线
    """
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def plot_training_results(rewards, losses=None, epsilons=None, window=10):
    """
    绘制训练结果
    
    参数:
        rewards: 每个回合的奖励
        losses: 每次更新的损失值
        epsilons: epsilon值的变化
        window: 移动平均窗口大小
    """
    st.subheader("训练结果")
    
    # 创建多个选项卡
    tabs = st.tabs(["奖励", "损失", "探索率"])
    
    # 奖励曲线
    with tabs[0]:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 原始奖励
        ax.plot(rewards, label='原始奖励', alpha=0.3, color='blue')
        
        # 计算移动平均
        if len(rewards) >= window:
            moving_avg = pd.Series(rewards).rolling(window=window).mean()
            ax.plot(moving_avg, label=f'{window}回合移动平均', color='red')
        
        # 计算平滑曲线
        smoothed = smooth_curve(rewards)
        ax.plot(smoothed, label='平滑曲线', color='green')
        
        ax.set_xlabel('回合')
        ax.set_ylabel('奖励')
        ax.set_title('训练过程中的回合奖励')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        
        st.pyplot(fig)
        
        # 统计信息
        col1, col2, col3 = st.columns(3)
        col1.metric("最终平均奖励", f"{np.mean(rewards[-100:]):.2f}")
        col2.metric("最高奖励", f"{np.max(rewards):.2f}")
        col3.metric("最终奖励", f"{rewards[-1]:.2f}")
    
    # 损失曲线
    with tabs[1]:
        if losses:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 原始损失
            ax.plot(losses, label='原始损失', alpha=0.3, color='blue')
            
            # 计算移动平均
            if len(losses) >= window:
                moving_avg = pd.Series(losses).rolling(window=window).mean()
                ax.plot(moving_avg, label=f'{window}次更新移动平均', color='red')
            
            # 计算平滑曲线
            smoothed = smooth_curve(losses)
            ax.plot(smoothed, label='平滑曲线', color='green')
            
            ax.set_xlabel('更新次数')
            ax.set_ylabel('损失')
            ax.set_title('训练过程中的损失变化')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
            
            st.pyplot(fig)
            
            # 统计信息
            col1, col2 = st.columns(2)
            col1.metric("最终平均损失", f"{np.mean(losses[-100:]):.4f}")
            col2.metric("最低损失", f"{np.min(losses):.4f}")
        else:
            st.info("没有损失数据可显示")
    
    # Epsilon曲线
    with tabs[2]:
        if epsilons:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.plot(epsilons, label='探索率', color='purple')
            
            ax.set_xlabel('更新次数')
            ax.set_ylabel('探索率')
            ax.set_title('训练过程中的探索率变化')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
            
            st.pyplot(fig)
            
            # 统计信息
            col1, col2 = st.columns(2)
            col1.metric("初始探索率", f"{epsilons[0]:.2f}")
            col2.metric("最终探索率", f"{epsilons[-1]:.2f}")
        else:
            st.info("没有探索率数据可显示")

def render_blackjack_state(state):
    """
    渲染21点游戏状态
    
    参数:
        state: (玩家点数, 庄家明牌点数, 是否有可用的A)
    """
    player_sum, dealer_card, usable_ace = state
    
    st.markdown(f"""
    ### 当前状态:
    - 玩家点数: **{player_sum}**
    - 庄家明牌: **{dealer_card}**
    - 可用的A: **{'是' if usable_ace else '否'}**
    """)

def visualize_q_values(agent, render_type='heatmap'):
    """
    可视化智能体学习到的价值函数
    
    参数:
        agent: DQN智能体
        render_type: 可视化类型，'heatmap' 或 'policy'
    """
    player_values = range(12, 22)  # 玩家可能的点数12-21
    dealer_values = range(1, 11)  # 庄家可能的明牌1-10
    has_usable_ace = [False, True]  # 是否有可用的A
    
    for ace in has_usable_ace:
        st.subheader(f"{'有可用A' if ace else '无可用A'}")
        
        if render_type == 'heatmap':
            # 准备Q值数据
            hit_q_values = np.zeros((len(player_values), len(dealer_values)))
            stick_q_values = np.zeros((len(player_values), len(dealer_values)))
            max_q_values = np.zeros((len(player_values), len(dealer_values)))
            
            for i, player in enumerate(player_values):
                for j, dealer in enumerate(dealer_values):
                    state = np.array([player, dealer, int(ace)], dtype=np.float32)
                    q_values = agent.q_network(torch.FloatTensor(state)).detach().numpy()
                    hit_q_values[i, j] = q_values[1]  # 要牌的Q值
                    stick_q_values[i, j] = q_values[0]  # 停牌的Q值
                    max_q_values[i, j] = max(q_values)
            
            # 创建热力图
            fig, axs = plt.subplots(1, 3, figsize=(18, 6))
            
            # 停牌Q值
            im0 = axs[0].imshow(stick_q_values, cmap='coolwarm')
            axs[0].set_title('停牌Q值')
            axs[0].set_xlabel('庄家明牌')
            axs[0].set_ylabel('玩家点数')
            axs[0].set_xticks(np.arange(len(dealer_values)))
            axs[0].set_yticks(np.arange(len(player_values)))
            axs[0].set_xticklabels(dealer_values)
            axs[0].set_yticklabels(player_values)
            fig.colorbar(im0, ax=axs[0])
            
            # 要牌Q值
            im1 = axs[1].imshow(hit_q_values, cmap='coolwarm')
            axs[1].set_title('要牌Q值')
            axs[1].set_xlabel('庄家明牌')
            axs[1].set_ylabel('玩家点数')
            axs[1].set_xticks(np.arange(len(dealer_values)))
            axs[1].set_yticks(np.arange(len(player_values)))
            axs[1].set_xticklabels(dealer_values)
            axs[1].set_yticklabels(player_values)
            fig.colorbar(im1, ax=axs[1])
            
            # 最大Q值
            im2 = axs[2].imshow(max_q_values, cmap='coolwarm')
            axs[2].set_title('最大Q值')
            axs[2].set_xlabel('庄家明牌')
            axs[2].set_ylabel('玩家点数')
            axs[2].set_xticks(np.arange(len(dealer_values)))
            axs[2].set_yticks(np.arange(len(player_values)))
            axs[2].set_xticklabels(dealer_values)
            axs[2].set_yticklabels(player_values)
            fig.colorbar(im2, ax=axs[2])
            
            plt.tight_layout()
            st.pyplot(fig)
            
        elif render_type == 'policy':
            # 准备策略数据
            action_values = np.zeros((len(player_values), len(dealer_values)))
            
            for i, player in enumerate(player_values):
                for j, dealer in enumerate(dealer_values):
                    state = np.array([player, dealer, int(ace)], dtype=np.float32)
                    q_values = agent.q_network(torch.FloatTensor(state)).detach().numpy()
                    action = np.argmax(q_values)  # 0: 停牌, 1: 要牌
                    action_values[i, j] = action
            
            # 创建自定义颜色映射
            colors = ['lightcoral', 'lightblue']  # 红色表示停牌，蓝色表示要牌
            cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=2)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(action_values, cmap=cmap, vmin=0, vmax=1)
            
            # 添加文字标注
            for i in range(len(player_values)):
                for j in range(len(dealer_values)):
                    text = "H" if action_values[i, j] == 1 else "S"  # H表示要牌，S表示停牌
                    ax.text(j, i, text, ha="center", va="center", color="black", fontsize=12, fontweight="bold")
            
            ax.set_title('最优策略 (S: 停牌, H: 要牌)')
            ax.set_xlabel('庄家明牌')
            ax.set_ylabel('玩家点数')
            ax.set_xticks(np.arange(len(dealer_values)))
            ax.set_yticks(np.arange(len(player_values)))
            ax.set_xticklabels(dealer_values)
            ax.set_yticklabels(player_values)
            
            # 添加网格线
            ax.set_xticks(np.arange(-0.5, len(dealer_values), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(player_values), 1), minor=True)
            ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
            
            plt.tight_layout()
            st.pyplot(fig)
