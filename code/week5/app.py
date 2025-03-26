"""
DQN算法交互式教程应用程序入口

整合所有模块，提供完整的DQN算法交互式学习体验
"""

import streamlit as st
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from components.sidebar import create_sidebar
from pages.theory import render_theory_page
from pages.implementation import render_implementation_page
from pages.demo import render_demo_page
from pages.interactive import render_interactive_page

def main():
    """主函数，设置页面布局并根据选择渲染对应页面"""
    # 设置页面配置
    st.set_page_config(
        page_title="DQN算法交互式教程",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 自定义CSS样式
    st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
        background-color: #f0f2f6;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4c9afe !important;
        color: white !important;
    }
    .streamlit-expanderHeader {
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 创建侧边栏
    selected_page = create_sidebar()
    
    # 渲染页面标题
    # 根据选择的页面渲染对应内容
    if selected_page == "理论基础":
        render_theory_page()
    elif selected_page == "代码实现":
        render_implementation_page()
    elif selected_page == "算法演示":
        render_demo_page()
    elif selected_page == "交互式实验":
        render_interactive_page()
    else:
        # 默认显示理论基础页面
        render_theory_page()

if __name__ == "__main__":
    main() 