import streamlit as st
from utils.visualization import nn_svg_to_html

def show_intro():
    """显示神经网络和深度学习导论页面"""
    st.title("神经网络与深度学习导论")
    
    st.markdown("""
    ## 欢迎学习神经网络与深度学习入门
    
    本课程旨在为后续学习**深度Q网络(DQN)**做准备，将帮助您理解神经网络的基本概念、架构和训练方法。
    
    ### 什么是神经网络？
    
    神经网络是受人脑结构启发的机器学习模型，由大量相互连接的处理单元（神经元）组成，能够从数据中学习并识别复杂模式。
    """)
    
    # 展示神经网络SVG图像
    st.markdown("""
    ### 典型的神经网络结构
    
    下图展示了一个多层神经网络的基本结构：
    """)
    
    # 使用自定义SVG图像
    st.markdown(nn_svg_to_html(), unsafe_allow_html=True)
    
    st.markdown("""
    ### 神经网络的主要特点

    - **自动特征学习**：能够自动从原始数据中提取有用特征
    - **非线性建模**：可以学习复杂的非线性关系
    - **分布式表示**：信息以分布式方式存储在网络权重中
    - **容错性**：对输入数据的噪声和缺失具有一定的鲁棒性
    
    ### 什么是深度学习？
    
    深度学习是神经网络的发展和扩展，特指具有多个隐藏层的神经网络模型。这些"深层"网络能够学习更复杂的特征层次，表示能力更强。
    
    ### 深度学习的关键突破
    
    - **更强大的计算能力**：GPU和专用硬件的发展
    - **大规模数据集**：互联网时代产生的海量数据
    - **改进的训练算法**：解决了深层网络的训练难题
    - **更好的正则化技术**：防止过拟合的新方法
    
    ### 为何学习神经网络对DQN很重要？
    
    深度Q网络(DQN)是强化学习和深度学习的结合：
    
    - DQN使用神经网络来近似Q函数
    - 理解神经网络的训练过程有助于理解DQN的稳定性问题
    - 神经网络的表达能力使DQN能处理高维状态空间
    """)
    
    # 课程大纲
    st.markdown("""
    ### 本课程大纲
    
    1. **人工神经元基础**：从单个神经元开始理解
    2. **神经网络架构**：前馈网络和层的概念
    3. **训练神经网络**：损失函数、梯度下降和反向传播
    4. **深度学习要点**：深层网络的特性
    5. **实践案例**：实现简单分类案例
    6. **神经网络与DQN的关系**：为下一步学习做准备
    7. **动手练习**：巩固所学知识
    """)
    
    # 学习目标
    st.markdown("""
    ### 学习目标
    
    完成本课程后，您将能够：
    
    - 理解神经网络的基本原理和组成部分
    - 掌握简单神经网络的训练过程
    - 了解深度学习的核心概念
    - 理解神经网络如何应用于DQN
    - 使用Python和PyTorch实现基础神经网络模型
    """)
    
    # 先修知识
    st.sidebar.markdown("""
    ### 先修知识
    
    本课程假设您已具备：
    
    - 基本的Python编程能力
    - 基础数学知识（线性代数、微积分）
    - 机器学习的基本概念
    """)
    
    # 交互提示
    st.sidebar.info("使用左侧导航栏浏览各个章节，每个章节都包含交互式示例和练习。") 