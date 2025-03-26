import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def activation_functions_exercise():
    """激活函数详细练习模块"""
    st.title("练习：激活函数详解")
    
    st.markdown("""
    ## 练习目标
    
    通过本练习，你将：
    1. 深入理解各种激活函数的特性和行为
    2. 实现并可视化常见激活函数及其梯度
    3. 分析不同激活函数对神经网络训练的影响
    4. 了解各激活函数的适用场景
    
    完成这些练习将帮助你更好地选择和使用神经网络中的激活函数。
    """)
    
    # 基础激活函数定义
    st.markdown("""
    ## 激活函数定义与实现
    
    在神经网络中，激活函数为网络引入非线性变换，使网络能够学习复杂的模式。
    下面是常见激活函数的实现：
    """)
    
    with st.expander("查看激活函数实现代码"):
        st.code("""
def sigmoid(x):
    \"\"\"Sigmoid激活函数：f(x) = 1 / (1 + e^(-x))\"\"\"
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    \"\"\"Sigmoid函数的导数：f'(x) = f(x) * (1 - f(x))\"\"\"
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    \"\"\"双曲正切函数：f(x) = (e^x - e^(-x)) / (e^x + e^(-x))\"\"\"
    return np.tanh(x)

def tanh_derivative(x):
    \"\"\"tanh函数的导数：f'(x) = 1 - f(x)^2\"\"\"
    t = np.tanh(x)
    return 1 - t**2

def relu(x):
    \"\"\"ReLU激活函数：f(x) = max(0, x)\"\"\"
    return np.maximum(0, x)

def relu_derivative(x):
    \"\"\"ReLU函数的导数：f'(x) = 1 if x > 0 else 0\"\"\"
    return np.where(x > 0, 1, 0)

def leaky_relu(x, alpha=0.01):
    \"\"\"Leaky ReLU激活函数：f(x) = x if x > 0 else alpha*x\"\"\"
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    \"\"\"Leaky ReLU函数的导数：f'(x) = 1 if x > 0 else alpha\"\"\"
    return np.where(x > 0, 1, alpha)

def elu(x, alpha=1.0):
    \"\"\"ELU激活函数：f(x) = x if x > 0 else alpha*(e^x - 1)\"\"\"
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def elu_derivative(x, alpha=1.0):
    \"\"\"ELU函数的导数：f'(x) = 1 if x > 0 else f(x) + alpha\"\"\"
    return np.where(x > 0, 1, alpha * np.exp(x))

def swish(x, beta=1.0):
    \"\"\"Swish激活函数：f(x) = x * sigmoid(beta*x)\"\"\"
    return x * sigmoid(beta * x)

def swish_derivative(x, beta=1.0):
    \"\"\"Swish函数的导数\"\"\"
    sig = sigmoid(beta * x)
    return beta * x * sig * (1 - sig) + sig

def mish(x):
    \"\"\"Mish激活函数：f(x) = x * tanh(softplus(x))\"\"\"
    softplus = np.log(1 + np.exp(x))
    return x * np.tanh(softplus)

def softmax(x):
    \"\"\"Softmax函数：将输入转换为概率分布\"\"\"
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        """)
    
    # 交互式激活函数可视化
    st.markdown("## 激活函数可视化")
    
    col1, col2 = st.columns(2)
    
    with col1:
        activation_choice = st.selectbox(
            "选择要可视化的激活函数",
            ["Sigmoid", "Tanh", "ReLU", "Leaky ReLU", "ELU", "Swish", "Mish"]
        )
    
    with col2:
        show_derivative = st.checkbox("显示导数", value=True)
    
    # 参数设置（针对特定激活函数）
    if activation_choice == "Leaky ReLU":
        alpha = st.slider("alpha参数", 0.0, 0.5, 0.01, 0.01)
    elif activation_choice == "ELU":
        alpha = st.slider("alpha参数", 0.1, 2.0, 1.0, 0.1)
    elif activation_choice == "Swish":
        beta = st.slider("beta参数", 0.1, 2.0, 1.0, 0.1)
    
    # 生成数据
    x = np.linspace(-5, 5, 1000)
    
    # 计算激活函数值
    if activation_choice == "Sigmoid":
        y = 1 / (1 + np.exp(-x))
        dy = y * (1 - y) if show_derivative else None
        title = "Sigmoid激活函数"
    elif activation_choice == "Tanh":
        y = np.tanh(x)
        dy = 1 - y**2 if show_derivative else None
        title = "Tanh激活函数"
    elif activation_choice == "ReLU":
        y = np.maximum(0, x)
        dy = np.where(x > 0, 1, 0) if show_derivative else None
        title = "ReLU激活函数"
    elif activation_choice == "Leaky ReLU":
        y = np.where(x > 0, x, alpha * x)
        dy = np.where(x > 0, 1, alpha) if show_derivative else None
        title = f"Leaky ReLU激活函数 (alpha={alpha})"
    elif activation_choice == "ELU":
        y = np.where(x > 0, x, alpha * (np.exp(x) - 1))
        dy = np.where(x > 0, 1, alpha * np.exp(x)) if show_derivative else None
        title = f"ELU激活函数 (alpha={alpha})"
    elif activation_choice == "Swish":
        sigmoid_x = 1 / (1 + np.exp(-beta * x))
        y = x * sigmoid_x
        if show_derivative:
            dy = beta * x * sigmoid_x * (1 - sigmoid_x) + sigmoid_x
        else:
            dy = None
        title = f"Swish激活函数 (beta={beta})"
    elif activation_choice == "Mish":
        softplus = np.log(1 + np.exp(x))
        tanh_softplus = np.tanh(softplus)
        y = x * tanh_softplus
        # Mish的导数比较复杂，这里使用近似值
        if show_derivative:
            omega = 4 * (x + 1) + 4 * np.exp(2*x) + np.exp(3*x) + np.exp(x) * (4*x + 6)
            delta = 2 * np.exp(x) + np.exp(2*x) + 2
            dy = np.exp(x) * omega / (delta**2)
        else:
            dy = None
        title = "Mish激活函数"
    
    # 绘制激活函数图
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, label=f"{activation_choice}函数")
    if show_derivative:
        ax.plot(x, dy, "--", label=f"{activation_choice}导数")
    
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax.set_title(title)
    ax.set_xlabel("输入 x")
    ax.set_ylabel("输出 f(x)")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    
    # 激活函数特性分析
    st.markdown("## 激活函数特性分析")
    
    # 各激活函数的特性比较表格
    st.markdown("### 激活函数特性比较")
    
    activation_data = {
        "激活函数": ["Sigmoid", "Tanh", "ReLU", "Leaky ReLU", "ELU", "Swish", "Mish"],
        "值域": ["(0, 1)", "(-1, 1)", "[0, +∞)", "(-∞, +∞)", "(-α, +∞)", "(-∞, +∞)", "(-∞, +∞)"],
        "饱和性": ["两侧饱和", "两侧饱和", "左侧饱和", "不饱和", "右侧不饱和", "左侧轻微饱和", "左侧轻微饱和"],
        "零中心性": ["否", "是", "否", "否", "近似", "近似", "近似"],
        "计算复杂度": ["中", "中", "低", "低", "高", "中", "高"],
        "梯度消失问题": ["严重", "有", "左侧有", "轻微", "轻微", "轻微", "轻微"],
        "常用于": ["输出层(二分类)", "隐藏层", "隐藏层", "隐藏层", "隐藏层", "隐藏层", "隐藏层"]
    }
    
    st.table(activation_data)
    
    # 激活函数选择指南
    st.markdown("""
    ### 激活函数选择指南
    
    1. **对于隐藏层**:
       - **推荐首选**: ReLU - 简单高效，训练速度快
       - **如果遇到"死亡ReLU"问题**: 尝试Leaky ReLU或ELU
       - **追求最新性能**: 尝试Swish或Mish (通常效果略好但计算成本更高)
    
    2. **对于输出层**:
       - **二分类问题**: Sigmoid
       - **多分类问题**: Softmax
       - **回归问题**: 线性激活(不使用激活函数)或轻微非线性(如tanh)
    """)
    
    # 练习：激活函数在前向传播中的应用
    st.markdown("""
    ## 练习：激活函数在前向传播中的应用
    
    下面的交互式演示展示了激活函数如何在神经网络的前向传播中发挥作用。
    """)
    
    # 简单神经元演示
    col1, col2 = st.columns(2)
    
    with col1:
        w1 = st.slider("权重 w₁", -2.0, 2.0, 0.5, 0.1)
        w2 = st.slider("权重 w₂", -2.0, 2.0, -0.5, 0.1)
        bias = st.slider("偏置 b", -2.0, 2.0, 0.0, 0.1)
    
    with col2:
        x1 = st.slider("输入 x₁", -2.0, 2.0, 1.0, 0.1)
        x2 = st.slider("输入 x₂", -2.0, 2.0, 0.5, 0.1)
        neuron_activation = st.selectbox(
            "神经元激活函数",
            ["ReLU", "Sigmoid", "Tanh", "Leaky ReLU"]
        )
    
    # 计算神经元输出
    z = w1 * x1 + w2 * x2 + bias
    
    if neuron_activation == "ReLU":
        output = max(0, z)
        activation_fn = "max(0, z)"
    elif neuron_activation == "Sigmoid":
        output = 1 / (1 + np.exp(-z))
        activation_fn = "1 / (1 + e^(-z))"
    elif neuron_activation == "Tanh":
        output = np.tanh(z)
        activation_fn = "tanh(z)"
    elif neuron_activation == "Leaky ReLU":
        output = z if z > 0 else 0.01 * z
        activation_fn = "z if z > 0 else 0.01*z"
    
    # 显示计算过程
    st.markdown(f"""
    ### 计算过程
    
    1. **线性组合**: z = w₁·x₁ + w₂·x₂ + b = {w1} · {x1} + {w2} · {x2} + {bias} = {z:.4f}
    
    2. **激活函数应用**: a = {neuron_activation}(z) = {activation_fn} = {output:.4f}
    """)
    
    # 可视化决策边界
    st.markdown("### 决策边界可视化")
    
    # 生成网格数据
    x1_range = np.linspace(-2, 2, 100)
    x2_range = np.linspace(-2, 2, 100)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    Z = w1 * X1 + w2 * X2 + bias
    
    # 应用激活函数
    if neuron_activation == "ReLU":
        A = np.maximum(0, Z)
    elif neuron_activation == "Sigmoid":
        A = 1 / (1 + np.exp(-Z))
    elif neuron_activation == "Tanh":
        A = np.tanh(Z)
    elif neuron_activation == "Leaky ReLU":
        A = np.where(Z > 0, Z, 0.01 * Z)
    
    # 绘制决策表面
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制输入点
    ax.scatter(x1, x2, color='red', s=100, zorder=10, label='输入点')
    
    # 绘制决策表面
    contour = ax.contourf(X1, X2, A, 20, cmap='viridis', alpha=0.7)
    fig.colorbar(contour, ax=ax, label=f'{neuron_activation}激活后的输出')
    
    # 绘制决策边界 (z=0的线)
    if w2 != 0:  # 避免除以零
        boundary_x2 = (-w1 * x1_range - bias) / w2
        ax.plot(x1_range, boundary_x2, 'k--', label='z=0线')
    
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_title(f'使用{neuron_activation}的神经元决策表面')
    ax.legend()
    ax.grid(alpha=0.3)
    
    st.pyplot(fig)
    
    # 梯度消失/爆炸问题
    st.markdown("""
    ## 激活函数与梯度消失/爆炸问题
    
    不同的激活函数对梯度传播有不同的影响。下面我们可以观察多层网络中的梯度变化。
    """)
    
    num_layers = st.slider("神经网络层数", 1, 50, 10)
    selected_activation = st.selectbox(
        "选择激活函数",
        ["Sigmoid", "Tanh", "ReLU", "Leaky ReLU", "ELU"],
        key="gradient_demo"
    )
    
    # 函数定义
    def apply_activation_and_gradient(x, activation):
        if activation == "Sigmoid":
            y = 1 / (1 + np.exp(-x))
            grad = y * (1 - y)
        elif activation == "Tanh":
            y = np.tanh(x)
            grad = 1 - y**2
        elif activation == "ReLU":
            y = np.maximum(0, x)
            grad = np.where(x > 0, 1, 0)
        elif activation == "Leaky ReLU":
            y = np.where(x > 0, x, 0.01 * x)
            grad = np.where(x > 0, 1, 0.01)
        elif activation == "ELU":
            y = np.where(x > 0, x, 1.0 * (np.exp(x) - 1))
            grad = np.where(x > 0, 1, 1.0 * np.exp(x))
        return y, grad
    
    # 模拟梯度传播
    initial_values = np.linspace(-2, 2, 100)
    gradients = np.ones_like(initial_values)  # 初始梯度为1
    
    # 计算每层的梯度变化
    layer_gradients = [gradients.copy()]
    
    for i in range(num_layers):
        # 模拟前向传播（简化）
        values = 1.0 * initial_values  # 简单的线性变换
        
        # 应用激活函数并获取梯度
        _, layer_grad = apply_activation_and_gradient(values, selected_activation)
        
        # 梯度传播（连乘）
        gradients = gradients * layer_grad
        layer_gradients.append(gradients.copy())
    
    # 可视化梯度变化
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 绘制不同层的梯度
    shown_layers = min(5, num_layers+1)  # 最多显示5层
    layer_indices = np.linspace(0, num_layers, shown_layers, dtype=int)
    
    for i, layer_idx in enumerate(layer_indices):
        if layer_idx < len(layer_gradients):
            ax.plot(initial_values, layer_gradients[layer_idx], 
                   label=f'第{layer_idx}层梯度')
    
    ax.set_xlabel('输入值')
    ax.set_ylabel('梯度值')
    ax.set_title(f'{selected_activation}激活函数在{num_layers}层网络中的梯度传播')
    ax.legend()
    ax.grid(alpha=0.3)
    
    st.pyplot(fig)
    
    # 梯度变化范围
    st.markdown("### 不同层的梯度幅度变化")
    
    # 计算每层梯度的平均幅度
    gradient_magnitudes = [np.mean(np.abs(g)) for g in layer_gradients]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(len(gradient_magnitudes)), gradient_magnitudes, 'o-')
    ax.set_xlabel('网络层数')
    ax.set_ylabel('梯度平均幅度')
    ax.set_title(f'{selected_activation}激活函数的梯度幅度变化')
    
    # 添加梯度消失/爆炸的参考线
    ax.axhline(y=0.001, color='r', linestyle='--', alpha=0.7, label='梯度消失阈值')
    ax.axhline(y=100, color='orange', linestyle='--', alpha=0.7, label='梯度爆炸阈值')
    
    ax.set_yscale('log')  # 使用对数尺度更好观察变化
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.3)
    
    st.pyplot(fig)
    
    # 思考问题
    st.markdown("""
    ## 思考问题
    
    1. 为什么ReLU激活函数在深度学习中如此流行？它相比Sigmoid和Tanh有哪些优势？
    
    2. "死亡ReLU"问题是什么？Leaky ReLU、PReLU和ELU如何解决这个问题？
    
    3. 在多层神经网络中，使用Sigmoid激活函数容易导致什么问题？为什么？
    
    4. 为什么输出层的激活函数选择通常与任务类型相关？
    
    5. 在实践中，如何选择合适的激活函数？有哪些经验法则？
    """)
    
    # 小结
    st.markdown("""
    ## 小结
    
    在本练习中，你：
    1. 实现并可视化了多种常见激活函数及其导数
    2. 了解了各激活函数的特性、优缺点和适用场景
    3. 观察了激活函数在神经元决策表面上的影响
    4. 探索了不同激活函数对梯度传播的影响及梯度消失/爆炸问题
    
    这些理解将帮助你更好地设计神经网络架构，选择适当的激活函数，并解决训练过程中可能遇到的问题。
    """)

if __name__ == "__main__":
    activation_functions_exercise() 