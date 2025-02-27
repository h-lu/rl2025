import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import plotly.graph_objects as go
import plotly.express as px
from matplotlib.patches import FancyArrowPatch

# 设置字体，确保中文显示正常
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei'] 
plt.rcParams['axes.unicode_minus'] = False

def plot_mdp_diagram():
    """
    绘制马尔可夫决策过程示意图
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 创建节点
    state_circle = plt.Circle((0.3, 0.5), 0.15, color='skyblue', alpha=0.6, label='State')
    next_state_circle = plt.Circle((0.7, 0.5), 0.15, color='skyblue', alpha=0.6)
    
    # 添加节点
    ax.add_patch(state_circle)
    ax.add_patch(next_state_circle)
    
    # 添加箭头
    arrow = FancyArrowPatch((0.45, 0.5), (0.55, 0.5), 
                         arrowstyle='->', 
                         mutation_scale=20, 
                         color='black')
    ax.add_patch(arrow)
    
    # 添加标签
    ax.text(0.3, 0.5, "$s_t$", ha='center', va='center', fontsize=14)
    ax.text(0.7, 0.5, "$s_{t+1}$", ha='center', va='center', fontsize=14)
    ax.text(0.5, 0.55, "动作 $a_t$", ha='center', va='bottom', fontsize=12)
    ax.text(0.5, 0.42, "奖励 $r_t$", ha='center', va='top', fontsize=12)
    
    # 添加环境和智能体框
    environment = plt.Rectangle((0.1, 0.2), 0.8, 0.6, fill=False, 
                            edgecolor='green', linestyle='--', linewidth=2)
    ax.add_patch(environment)
    ax.text(0.5, 0.15, "环境", ha='center', va='center', fontsize=14, color='green')
    
    agent = plt.Rectangle((0.25, 0.7), 0.5, 0.2, fill=False,
                       edgecolor='red', linestyle='--', linewidth=2)
    ax.add_patch(agent)
    ax.text(0.5, 0.8, "智能体", ha='center', va='center', fontsize=14, color='red')
    
    # 设置轴的范围和隐藏刻度
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # 添加标题
    ax.set_title("马尔可夫决策过程 (MDP)", fontsize=16)
    
    return fig

def plot_value_function(grid_size=5):
    """
    绘制一个简单的价值函数热图
    """
    # 创建一个示例价值函数
    np.random.seed(42)
    values = np.random.uniform(-1, 1, (grid_size, grid_size))
    values[grid_size//2, grid_size//2] = 2  # 目标位置
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(values, annot=True, fmt=".2f", cmap="coolwarm", 
                cbar_kws={'label': '状态价值 V(s)'}, ax=ax)
    
    # 设置标题和轴标签
    ax.set_title("状态价值函数热图示例", fontsize=14)
    ax.set_xlabel("列坐标", fontsize=12)
    ax.set_ylabel("行坐标", fontsize=12)
    
    return fig

def plot_exploration_exploitation():
    """
    绘制探索与利用的平衡图
    """
    epsilon_values = np.linspace(0, 1, 100)
    exploration = epsilon_values
    exploitation = 1 - epsilon_values
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(epsilon_values, exploration, 'r-', linewidth=2, label='探索')
    ax.plot(epsilon_values, exploitation, 'b-', linewidth=2, label='利用')
    ax.fill_between(epsilon_values, 0, exploration, color='red', alpha=0.2)
    ax.fill_between(epsilon_values, 0, exploitation, color='blue', alpha=0.2)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('$\epsilon$ 值', fontsize=12)
    ax.set_ylabel('概率', fontsize=12)
    ax.set_title('$\epsilon$-greedy 策略中探索与利用的平衡', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    return fig

def plot_grid_world(grid_map, agent_pos=None, target_pos=None):
    """
    可视化Grid World环境
    
    参数:
        grid_map: 网格地图数组
        agent_pos: 智能体位置(行,列)
        target_pos: 目标位置(行,列)
    """
    # 创建自定义的颜色映射
    cmap = ListedColormap(['white', 'black', 'gold', 'red'])
    
    # 创建网格地图的副本
    plot_grid = np.copy(grid_map)
    
    # 添加智能体和目标位置
    if agent_pos is not None:
        # 临时保存智能体位置的原始值
        original_value = plot_grid[agent_pos[0], agent_pos[1]]
        plot_grid[agent_pos[0], agent_pos[1]] = 3  # 标记为红色
    
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(plot_grid, cmap=cmap, cbar=False, 
                annot=False, square=True, linewidths=.5, ax=ax)
    
    # 添加标签
    for i in range(plot_grid.shape[0]):
        for j in range(plot_grid.shape[1]):
            cell_value = grid_map[i, j]
            
            # 根据单元格类型添加标签
            if cell_value == 0:  # 空白单元格
                pass
            elif cell_value == 1:  # 墙壁
                pass
            elif cell_value == 2:  # 目标
                ax.text(j + 0.5, i + 0.5, "G", ha='center', va='center', fontsize=20, 
                        weight='bold', color='darkred')
            
            # 如果是智能体位置
            if agent_pos is not None and i == agent_pos[0] and j == agent_pos[1]:
                ax.text(j + 0.5, i + 0.5, "A", ha='center', va='center', fontsize=20, 
                        weight='bold', color='blue')
    
    # 恢复智能体位置的原始值
    if agent_pos is not None:
        plot_grid[agent_pos[0], agent_pos[1]] = original_value
    
    # 设置轴标签
    ax.set_ylabel('行', fontsize=12)
    ax.set_xlabel('列', fontsize=12)
    ax.set_title('Grid World 环境', fontsize=14)
    
    # 反转y轴，使(0,0)位于左上角
    ax.invert_yaxis()
    
    return fig

def plot_q_values(q_table, grid_size=5):
    """
    可视化Q值表
    
    参数:
        q_table: 形状为 (state_count, action_count) 的Q值表
        grid_size: 网格大小
    """
    state_count, action_count = q_table.shape
    
    # 创建图形
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    
    # 方向标签
    action_labels = ['上', '右', '下', '左']
    
    # 遍历所有状态
    for i in range(grid_size):
        for j in range(grid_size):
            state_idx = i * grid_size + j
            
            # 计算每个方向的箭头长度，基于Q值
            q_values = q_table[state_idx]
            
            # 规范化Q值以便可视化
            if np.max(np.abs(q_values)) > 0:
                normalized_q = q_values / np.max(np.abs(q_values))
            else:
                normalized_q = np.zeros(action_count)
            
            ax = axes[i, j]
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            
            # 画出四个方向的箭头
            if normalized_q[0] != 0:  # 上
                ax.arrow(0, 0, 0, normalized_q[0], head_width=0.2, 
                        head_length=0.2, fc='blue', ec='blue', alpha=0.7)
                # 显示实际Q值
                ax.text(0, normalized_q[0]/2 + 0.1, f"{q_values[0]:.1f}", ha='center', fontsize=8)
            if normalized_q[1] != 0:  # 右
                ax.arrow(0, 0, normalized_q[1], 0, head_width=0.2, 
                        head_length=0.2, fc='red', ec='red', alpha=0.7)
                ax.text(normalized_q[1]/2 + 0.1, 0, f"{q_values[1]:.1f}", va='center', fontsize=8)
            if normalized_q[2] != 0:  # 下
                ax.arrow(0, 0, 0, -normalized_q[2], head_width=0.2, 
                        head_length=0.2, fc='green', ec='green', alpha=0.7)
                ax.text(0, -normalized_q[2]/2 - 0.1, f"{q_values[2]:.1f}", ha='center', fontsize=8)
            if normalized_q[3] != 0:  # 左
                ax.arrow(0, 0, -normalized_q[3], 0, head_width=0.2, 
                        head_length=0.2, fc='purple', ec='purple', alpha=0.7)
                ax.text(-normalized_q[3]/2 - 0.1, 0, f"{q_values[3]:.1f}", va='center', fontsize=8)
            
            # 显示最大Q值的方向
            if np.any(normalized_q != 0):
                best_action = np.argmax(q_values)
                ax.text(0, -0.8, f"最优动作: {action_labels[best_action]}", ha='center', fontsize=8, 
                       bbox=dict(facecolor='yellow', alpha=0.2))
            
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"状态 ({i},{j})", fontsize=10)
    
    # 添加图例
    legend_elements = [
        plt.Line2D([0], [0], color='blue', lw=2, label='上'),
        plt.Line2D([0], [0], color='red', lw=2, label='右'),
        plt.Line2D([0], [0], color='green', lw=2, label='下'),
        plt.Line2D([0], [0], color='purple', lw=2, label='左')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=4)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    fig.suptitle('Q值表可视化 - 箭头方向表示动作，箭头长度表示Q值大小', fontsize=16)
    
    return fig

def plot_learning_curve(rewards, window=10):
    """
    绘制学习曲线，展示智能体随时间的学习进度
    
    参数:
        rewards: 每个episode的奖励列表
        window: 移动平均窗口大小
    """
    # 计算移动平均
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    else:
        moving_avg = rewards
        
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制原始奖励
    ax.plot(rewards, 'b-', alpha=0.3, label='原始奖励')
    
    # 绘制移动平均
    if len(rewards) >= window:
        ax.plot(np.arange(window-1, len(rewards)), moving_avg, 'r-', linewidth=2,
                label=f'移动平均 (窗口={window})')
    
    # 添加趋势线
    if len(rewards) > 2:
        z = np.polyfit(range(len(rewards)), rewards, 1)
        p = np.poly1d(z)
        ax.plot(range(len(rewards)), p(range(len(rewards))), "g--", linewidth=1, 
                label=f"趋势线 (斜率={z[0]:.4f})")
    
    # 添加注释和解释
    if len(rewards) > 0:
        # 标记最大值和最小值
        max_reward = max(rewards)
        max_idx = rewards.index(max_reward)
        min_reward = min(rewards)
        min_idx = rewards.index(min_reward)
        
        ax.annotate(f"最大值: {max_reward:.2f}", 
                   xy=(max_idx, max_reward), xytext=(-20, 20),
                   textcoords="offset points", 
                   arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
        
        ax.annotate(f"最小值: {min_reward:.2f}", 
                   xy=(min_idx, min_reward), xytext=(-20, -20),
                   textcoords="offset points", 
                   arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    
    ax.set_xlabel('Episode 数', fontsize=12)
    ax.set_ylabel('累积奖励', fontsize=12)
    ax.set_title('强化学习过程中的学习曲线', fontsize=14)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 添加背景色带表示不同学习阶段
    if len(rewards) >= 30:
        ax.axvspan(0, len(rewards)//3, alpha=0.1, color='red', label='早期探索阶段')
        ax.axvspan(len(rewards)//3, 2*len(rewards)//3, alpha=0.1, color='yellow', label='过渡学习阶段')
        ax.axvspan(2*len(rewards)//3, len(rewards), alpha=0.1, color='green', label='策略收敛阶段')
        
        # 更新图例
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels, loc='best', fontsize=10)
    
    # 添加解释性文本
    plt.figtext(0.5, 0.01, 
                "学习曲线展示了智能体在训练过程中的表现变化。\n"
                "上升的曲线表示学习效果改善，平稳的曲线表示策略逐渐收敛。", 
                ha="center", fontsize=10, 
                bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
    
    return fig 