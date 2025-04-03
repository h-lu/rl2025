import streamlit as st
import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import platform
import numpy as np

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入各模块
from pages.intro import show_intro
from pages.neuron_basics import show_neuron_basics
from pages.feedforward_networks import show_feedforward_networks
from pages.training import show_training
from pages.deep_learning_intro import show_deep_learning_intro
from pages.simple_classification import show_simple_classification
from pages.nn_dqn_relation import show_nn_dqn_relation
from pages.exercises import show_exercises

# 导入练习模块
# from exercises.perceptron_exercise import perceptron_exercise
# from exercises.activation_functions_exercise import activation_functions_exercise
# from exercises.forward_propagation_exercise import forward_propagation_exercise
# from exercises.backpropagation_exercise import backpropagation_exercise
# from exercises.nn_dqn_exercise import nn_dqn_exercise

# 配置matplotlib支持中文
def configure_matplotlib_chinese_fonts():
    """配置matplotlib支持中文字体"""
    system = platform.system()
    
    # 直接设置字体，不使用复杂的尝试逻辑
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']  # 优先使用Mac上常见的Unicode字体
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    
    # Mac系统特殊设置
    if system == 'Darwin':
        # 尝试直接使用系统默认字体
        try:
            import matplotlib.font_manager as fm
            # 强制刷新字体缓存
            fm._rebuild()
            print("已刷新matplotlib字体缓存")
        except:
            pass

# 配置中文字体
configure_matplotlib_chinese_fonts()

# 设置页面配置
st.set_page_config(
    page_title="神经网络与深度学习入门",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 侧边栏导航
st.sidebar.title("神经网络与深度学习入门")
st.sidebar.markdown("为DQN学习做准备的90分钟课程")

# 导航选项
pages = {
    "1. 导论": show_intro,
    "2. 人工神经元基础": show_neuron_basics,
    "3. 神经网络架构": show_feedforward_networks,
    "4. 训练神经网络": show_training,
    "5. 深度学习要点": show_deep_learning_intro,
    "6. 实践案例": show_simple_classification,
    "7. 神经网络与DQN的关系": show_nn_dqn_relation,
    "8. 练习环节": show_exercises
}

# 选择页面
selection = st.sidebar.radio("导航", list(pages.keys()))

# 显示选中的页面
pages[selection]()

# 页脚
st.sidebar.markdown("---")
st.sidebar.info("© 2024 神经网络与DQN入门课程") 