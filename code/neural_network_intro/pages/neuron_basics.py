import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils.visualization import plot_activation_functions, plot_neuron, fig_to_html

def show_neuron_basics():
    """显示人工神经元基础页面"""
    st.title("人工神经元基础")
    
    st.markdown("""
    ## 从生物神经元到人工神经元
    
    神经网络的基本构建块是**人工神经元**，其设计受到生物神经元的启发。
    """)
    
    # 展示生物神经元与人工神经元的对比
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 生物神经元")
        st.markdown("""
        生物神经元的主要组成部分：
        - **树突**：接收来自其他神经元的信号
        - **细胞体**：整合输入信号
        - **轴突**：传递输出信号到其他神经元
        """)
        st.image("https://upload.wikimedia.org/wikipedia/commons/1/10/Blausen_0657_MultipolarNeuron.png", 
                caption="生物神经元结构", width=300)
    
    with col2:
        st.markdown("### 人工神经元")
        st.markdown("""
        人工神经元的对应结构：
        - **输入与权重**：模拟树突接收的信号及其强度
        - **加权求和**：模拟细胞体的信号整合
        - **激活函数**：模拟触发动作电位的过程
        - **输出**：类似轴突传递的信号
        """)
        neuron_fig = plot_neuron()
        st.pyplot(neuron_fig)
    
    # 人工神经元的数学模型
    st.markdown("""
    ## 人工神经元的数学模型
    
    一个基本的人工神经元执行以下操作：
    
    1. 计算输入的加权和
    2. 添加偏置项
    3. 将结果传递给激活函数
    4. 输出激活函数的结果
    
    ### 数学表达式
    
    对于输入 $x_1, x_2, ..., x_n$，权重 $w_1, w_2, ..., w_n$，和偏置 $b$：
    
    $$z = w_1 x_1 + w_2 x_2 + ... + w_n x_n + b = \sum_{i=1}^{n} w_i x_i + b$$
    
    $$y = f(z)$$
    
    其中 $f$ 是激活函数，$y$ 是神经元的输出。
    """)
    
    # 激活函数
    st.markdown("""
    ## 激活函数
    
    激活函数引入非线性，使神经网络能够学习复杂的模式。以下是常见的激活函数：
    """)
    
    # 展示激活函数图像
    activation_fig = plot_activation_functions()
    st.pyplot(activation_fig)
    
    # 各种激活函数的特点
    with st.expander("各种激活函数的特点与应用"):
        st.markdown("""
        ### Sigmoid 函数
        
        $$f(x) = \\frac{1}{1 + e^{-x}}$$
        
        **特点**:
        - 输出范围在 (0, 1) 之间
        - 在深层网络中容易导致梯度消失
        - 历史上常用于二分类问题的输出层
        
        ### Tanh 函数
        
        $$f(x) = \\frac{e^x - e^{-x}}{e^x + e^{-x}}$$
        
        **特点**:
        - 输出范围在 (-1, 1) 之间
        - 零中心化，有助于优化
        - 同样存在梯度消失问题
        
        ### ReLU (Rectified Linear Unit)
        
        $$f(x) = max(0, x)$$
        
        **特点**:
        - 计算简单，训练速度快
        - 缓解了梯度消失问题
        - 可能导致"神经元死亡"问题
        - 现代深度学习中最常用的激活函数
        
        ### Leaky ReLU
        
        $$f(x) = max(\\alpha x, x), \\text{其中} \\alpha \\text{通常为} 0.01$$
        
        **特点**:
        - 解决了ReLU的"神经元死亡"问题
        - 保留了ReLU的大部分优点
        """)
    
    # 交互式单个神经元演示
    st.markdown("## 交互式神经元演示")
    
    st.markdown("""
    下面是一个简单的神经元模型，您可以调整输入值和权重，观察输出的变化。
    """)
    
    # 设置三个输入和权重的滑动条
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("输入值")
        x1 = st.slider("输入 x₁", -1.0, 1.0, 0.5, 0.1)
        x2 = st.slider("输入 x₂", -1.0, 1.0, -0.3, 0.1)
        x3 = st.slider("输入 x₃", -1.0, 1.0, 0.8, 0.1)
    
    with col2:
        st.subheader("权重值")
        w1 = st.slider("权重 w₁", -1.0, 1.0, 0.5, 0.1)
        w2 = st.slider("权重 w₂", -1.0, 1.0, 0.7, 0.1)
        w3 = st.slider("权重 w₃", -1.0, 1.0, -0.2, 0.1)
    
    # 偏置值
    bias = st.slider("偏置 b", -1.0, 1.0, 0.0, 0.1)
    
    # 激活函数选择
    activation = st.selectbox(
        "选择激活函数",
        ["Sigmoid", "Tanh", "ReLU", "Leaky ReLU", "Linear"]
    )
    
    # 计算神经元输出
    inputs = np.array([x1, x2, x3])
    weights = np.array([w1, w2, w3])
    
    z = np.dot(inputs, weights) + bias
    
    # 应用激活函数
    if activation == "Sigmoid":
        output = 1 / (1 + np.exp(-z))
        formula = f"y = sigmoid({z:.3f}) = 1 / (1 + e^(-{z:.3f}))"
    elif activation == "Tanh":
        output = np.tanh(z)
        formula = f"y = tanh({z:.3f})"
    elif activation == "ReLU":
        output = max(0, z)
        formula = f"y = ReLU({z:.3f}) = max(0, {z:.3f})"
    elif activation == "Leaky ReLU":
        alpha = 0.01
        output = max(alpha * z, z)
        formula = f"y = Leaky ReLU({z:.3f}) = max({alpha} * {z:.3f}, {z:.3f})"
    else:  # Linear
        output = z
        formula = f"y = {z:.3f}"
    
    # 显示结果
    st.markdown("### 计算结果")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **加权和**: $z = (w_1 \\cdot x_1) + (w_2 \\cdot x_2) + (w_3 \\cdot x_3) + b$
        
        $z = ({w1:.2f} \\cdot {x1:.2f}) + ({w2:.2f} \\cdot {x2:.2f}) + ({w3:.2f} \\cdot {x3:.2f}) + {bias:.2f} = {z:.3f}$
        """)
    
    with col2:
        st.markdown(f"""
        **激活后输出**: {formula}
        
        $y = {output:.5f}$
        """)
    
    # 可视化神经元的响应
    st.markdown("### 神经元响应可视化")
    
    # 创建一个网格来绘制激活函数曲线
    x_vals = np.linspace(-5, 5, 100)
    
    if activation == "Sigmoid":
        y_vals = 1 / (1 + np.exp(-x_vals))
        title = "Sigmoid 激活函数"
    elif activation == "Tanh":
        y_vals = np.tanh(x_vals)
        title = "Tanh 激活函数"
    elif activation == "ReLU":
        y_vals = np.maximum(0, x_vals)
        title = "ReLU 激活函数"
    elif activation == "Leaky ReLU":
        alpha = 0.01
        y_vals = np.where(x_vals > 0, x_vals, alpha * x_vals)
        title = "Leaky ReLU 激活函数"
    else:  # Linear
        y_vals = x_vals
        title = "线性激活函数"
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x_vals, y_vals)
    ax.axvline(x=z, color='r', linestyle='--')
    ax.axhline(y=output, color='r', linestyle='--')
    ax.plot(z, output, 'ro', markersize=8)
    ax.set_title(title)
    ax.grid(True)
    ax.set_xlabel("输入 (z)")
    ax.set_ylabel("输出 (y)")
    
    # 可视化当前点在曲线上的位置
    ax.annotate(f'({z:.2f}, {output:.2f})', 
                xy=(z, output), 
                xytext=(z+0.5, output+0.1),
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    st.pyplot(fig)
    
    # 感知器和神经元的区别
    with st.expander("感知器与现代神经元的区别"):
        st.markdown("""
        ### 感知器与现代神经元
        
        **感知器** 是最早的神经网络模型之一，由Frank Rosenblatt在1958年提出。
        
        **特点**:
        - 使用阶跃函数作为激活函数 $f(z) = 1 \\text{ if } z \\geq 0 \\text{ else } 0$
        - 只能学习线性可分的问题
        - 不能解决XOR等非线性问题
        
        **现代神经元** 使用连续可微的激活函数，如Sigmoid、ReLU等。
        
        **主要差异**:
        - 现代神经元使用连续可微激活函数，允许梯度下降优化
        - 多层神经元可以解决非线性问题
        - 现代模型支持前向和反向传播
        """)
    
    # 基础练习部分
    st.markdown("## 基础练习")
    
    st.markdown("""
    1. 尝试找出使神经元输出为0.5的输入值组合。
    2. 切换不同激活函数，观察相同输入下输出的变化。
    3. 思考为什么ReLU在现代神经网络中如此受欢迎。
    """)
    
    # 扩展学习资源
    st.sidebar.markdown("""
    ### 扩展资源
    
    - 3Blue1Brown: [但是什么是神经网络？](https://www.youtube.com/watch?v=aircAruvnKk)
    - 吴恩达深度学习课程中的神经网络基础
    - PyTorch文档: 神经网络模块
    """)
    
    # 小测验
    with st.expander("小测验: 检验您的理解"):
        st.markdown("""
        1. 在神经网络中，权重的作用是什么？
           - A. 存储神经元的中间状态
           - B. 决定各输入对输出的重要性
           - C. 控制网络的学习速率
           - D. 防止过拟合
        
        2. 为什么神经网络需要激活函数？
           - A. 增加模型参数数量
           - B. 简化计算过程
           - C. 引入非线性，增强表达能力
           - D. 加速训练过程
        
        3. 下列哪个激活函数的值域是(0,1)？
           - A. ReLU
           - B. Tanh
           - C. Sigmoid
           - D. Leaky ReLU
        
        **答案**: 1-B, 2-C, 3-C
        """)
        
    # 下一章预告
    st.markdown("""
    ## 下一章预告
    
    在下一章，我们将学习如何将这些神经元连接起来形成**神经网络**，以及网络的层次结构和前向传播过程。
    """) 