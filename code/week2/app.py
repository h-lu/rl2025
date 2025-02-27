import streamlit as st
import base64
from pathlib import Path
import sys
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']  # macOS优先使用Arial Unicode MS
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 添加当前目录到路径，以便导入其他模块
sys.path.append(os.path.dirname(__file__))

# 导入各个页面模块
from pages import introduction, value_functions, exploration, grid_world, basic_exercises, advanced_exercises

# 设置页面配置
st.set_page_config(
    page_title="第二周：强化学习框架与迷宫环境",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 加载和显示图片的函数
def load_image_as_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# 自定义CSS，设置字体
st.markdown("""
<style>
    body {
        font-family: "Arial Unicode MS", "Microsoft YaHei", "STXihei", sans-serif;
    }
    .stMarkdown, .stText, .stButton, .stCheckbox, .stTitle, .stSubheader {
        font-family: "Arial Unicode MS", "Microsoft YaHei", "STXihei", sans-serif !important;
    }
</style>
""", unsafe_allow_html=True)

# 侧边栏标题
st.sidebar.title("第二周：强化学习框架与迷宫环境")

# 页面选项列表
page_options = [
    "课程介绍", 
    "马尔可夫决策过程", 
    "价值函数", 
    "探索与利用平衡", 
    "Grid World环境",
    "基础练习",
    "进阶练习"
]

# 选择页面
selection = st.sidebar.radio("选择内容", page_options)

# 页面映射
if selection in ["课程介绍", "马尔可夫决策过程"]:
    current_page = introduction
elif selection == "价值函数":
    current_page = value_functions
elif selection == "探索与利用平衡":
    current_page = exploration
elif selection == "Grid World环境":
    current_page = grid_world
elif selection == "基础练习":
    current_page = basic_exercises
else:  # 进阶练习
    current_page = advanced_exercises

# 侧边栏学习目标
with st.sidebar.expander("本周学习目标", expanded=False):
    st.markdown("""
    - 理解马尔可夫决策过程 (MDP) 的基本思想
    - 掌握策略 (Policy)、价值函数 (Value Function) 的概念
    - 理解探索 (Exploration) 与利用 (Exploitation) 的平衡
    - 学习使用 Gymnasium 库搭建迷宫环境 (Grid World)
    - 掌握使用 AI 辅助工具进行代码补全和修改
    """)

# 显示选定的页面
current_page.show()

# 页面底部
st.sidebar.markdown("---")
st.sidebar.info(
    "**强化学习 2024年春季学期**\n\n"
    "本课件供教学使用，请勿商用。"
) 