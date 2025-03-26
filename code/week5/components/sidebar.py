"""
侧边栏组件模块

为Streamlit应用提供统一的侧边栏导航和配置选项
"""

import streamlit as st
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

def create_sidebar():
    """
    创建侧边栏导航和配置选项
    
    返回:
        str: 当前选择的页面
    """
    st.sidebar.title("DQN交互式课件")
    
    # 添加导航选项
    selected_page = st.sidebar.radio(
        "导航",
        config.SIDEBAR_OPTIONS
    )
    
    st.sidebar.divider()
    
    # 添加"关于"部分
    with st.sidebar.expander("关于", expanded=False):
        st.markdown("""
        ### 深度Q网络 (DQN) 交互式课件
        
        本课件旨在帮助你理解深度强化学习中的DQN算法。
        
        * 查看理论基础，了解DQN的核心概念
        * 研究代码实现，掌握算法的具体细节
        * 观看算法演示，直观理解DQN的工作原理
        * 进行交互式实验，探索不同参数的影响
        """)
    
    return selected_page

def create_dqn_params_config():
    """
    创建DQN参数配置选项
    
    返回:
        dict: 配置参数
    """
    st.sidebar.subheader("参数设置")
    
    params = {}
    
    # 核心DQN参数
    params['gamma'] = st.sidebar.slider(
        "折扣因子 (γ)", 
        min_value=0.8, 
        max_value=0.999, 
        value=config.GAMMA,
        step=0.01,
        help="控制未来奖励的重要性，值越大越重视长期奖励"
    )
    
    params['epsilon_start'] = st.sidebar.slider(
        "初始探索率 (ε_start)", 
        min_value=0.5, 
        max_value=1.0, 
        value=config.EPSILON_START,
        step=0.05,
        help="初始随机探索的概率"
    )
    
    params['epsilon_end'] = st.sidebar.slider(
        "最小探索率 (ε_end)", 
        min_value=0.01, 
        max_value=0.2, 
        value=config.EPSILON_END,
        step=0.01,
        help="探索率的最小值"
    )
    
    params['epsilon_decay'] = st.sidebar.slider(
        "探索率衰减 (ε_decay)", 
        min_value=0.9, 
        max_value=0.999, 
        value=config.EPSILON_DECAY,
        step=0.001,
        help="探索率的衰减系数，值越大衰减越慢"
    )
    
    # 网络和训练参数
    with st.sidebar.expander("高级参数", expanded=False):
        params['learning_rate'] = st.slider(
            "学习率", 
            min_value=0.0001, 
            max_value=0.01, 
            value=config.LEARNING_RATE,
            step=0.0001,
            format="%.4f",
            help="神经网络的学习率"
        )
        
        params['buffer_size'] = st.slider(
            "经验回放缓冲区大小", 
            min_value=1000, 
            max_value=100000, 
            value=config.BUFFER_SIZE,
            step=1000,
            help="经验回放缓冲区的容量"
        )
        
        params['batch_size'] = st.slider(
            "批次大小", 
            min_value=16, 
            max_value=256, 
            value=config.BATCH_SIZE,
            step=16,
            help="每次学习使用的经验样本数量"
        )
        
        params['update_target_every'] = st.slider(
            "目标网络更新频率", 
            min_value=5, 
            max_value=100, 
            value=config.UPDATE_TARGET_EVERY,
            step=5,
            help="多少步更新一次目标网络"
        )
        
        params['hidden_size'] = st.slider(
            "隐藏层大小", 
            min_value=32, 
            max_value=256, 
            value=config.HIDDEN_SIZE,
            step=32,
            help="神经网络隐藏层的节点数"
        )
    
    # 训练控制参数
    with st.sidebar.expander("训练设置", expanded=False):
        params['max_episodes'] = st.slider(
            "最大回合数", 
            min_value=100, 
            max_value=1000, 
            value=config.MAX_EPISODES,
            step=100,
            help="训练的最大回合数"
        )
        
        params['target_score'] = st.slider(
            "目标分数", 
            min_value=100, 
            max_value=500, 
            value=int(config.TARGET_SCORE),
            step=50,
            help="达到此分数后停止训练"
        )
    
    return params 