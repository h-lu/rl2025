import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.colors import LinearSegmentedColormap

# 添加matplotlib中文字体配置函数
def configure_matplotlib_fonts():
    """配置matplotlib使用系统中文字体"""
    # 对于不同操作系统设置不同的字体
    import platform
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

def create_q_table_df(states, actions, q_table):
    """
    创建Q表的DataFrame展示格式
    
    参数:
    - states: 状态列表
    - actions: 动作列表
    - q_table: Q值字典，键为(state, action)元组，值为Q值
    
    返回:
    - df: 格式化的DataFrame
    """
    data = []
    for state in states:
        row = {"状态": state}
        for action in actions:
            row[action] = q_table.get((state, action), 0.0)
        data.append(row)
    
    return pd.DataFrame(data)

def plot_learning_curve(rewards, title="学习曲线"):
    """
    绘制学习曲线图
    
    参数:
    - rewards: 奖励列表
    - title: 图表标题
    
    返回:
    - fig: matplotlib图表对象
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(rewards)
    ax.set_title(title)
    ax.set_xlabel("回合")
    ax.set_ylabel("奖励")
    ax.grid(True)
    
    # 添加移动平均线
    if len(rewards) > 10:
        window_size = min(50, len(rewards) // 5)
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        ax.plot(range(window_size-1, len(rewards)), moving_avg, 'r--', linewidth=2)
        ax.legend(['回合奖励', f'{window_size}回合移动平均'])
    
    return fig

def plot_heatmap(df, title="Q值热力图", color_scale="Blues"):
    """
    创建一个单一颜色由浅到深的热力图
    
    参数:
    - df: 包含Q值的DataFrame
    - title: 图表标题
    
    返回:
    - fig: plotly图表对象
    """
    fig = px.imshow(
        df.iloc[:, 1:].values,
        x=df.columns[1:],
        y=df['状态'],
        color_continuous_scale=color_scale,
        labels=dict(x="动作", y="状态", color="Q值"),
        title=title
    )
    return fig

def display_algorithm_step(step_name, code_snippet, explanation):
    """显示算法步骤、代码和解释"""
    st.markdown(f"### {step_name}")
    st.code(code_snippet, language="python")
    st.markdown(f"**解释**: {explanation}")

def interactive_parameter_section(param_name, min_val, max_val, default_val, step=0.1, description=""):
    """创建交互式参数调整组件"""
    st.markdown(f"#### {param_name}")
    if description:
        st.markdown(description)
    return st.slider(f"选择{param_name}值", min_val, max_val, default_val, step=step) 