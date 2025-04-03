import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
from utils.data_utils import generate_regression_data
from utils.visualization_utils import setup_chinese_font

def optimizer_backprop_exercise():
    """优化器和反向传播练习页面"""
    st.title("练习：优化器与反向传播")
    
    # 设置中文字体
    setup_chinese_font()
    
    st.markdown("""
    ## 练习目标
    
    通过本练习，你将：
    1. 理解梯度下降的基本原理
    2. 学习神经网络中的反向传播算法
    3. 比较不同优化器的性能
    4. 观察优化过程的收敛行为
    
    完成这些练习将帮助你理解神经网络训练的核心机制。
    """)
    
    # 练习1：梯度下降基础
    st.markdown("""
    ## 练习1：梯度下降基础
    
    梯度下降是训练神经网络的基础算法。它通过计算损失函数相对于模型参数的梯度，
    并沿着梯度的反方向更新参数，从而最小化损失函数。
    """)
    
    st.markdown("""
    ### 梯度下降公式
    
    参数更新公式：
    
    $$\\theta_{new} = \\theta_{old} - \\alpha \\nabla J(\\theta)$$
    
    其中：
    - $\\theta$ 是模型参数
    - $\\alpha$ 是学习率
    - $\\nabla J(\\theta)$ 是损失函数 $J$ 关于 $\\theta$ 的梯度
    """)
    
    # 梯度下降可视化
    st.markdown("### 梯度下降可视化")
    
    # 一个简单的二次函数
    def f(x):
        return x**2 + 2
    
    def df(x):
        return 2*x
    
    # 用户可调整的参数
    learning_rate = st.slider("学习率", 0.01, 0.5, 0.1, 0.01)
    iterations = st.slider("迭代次数", 5, 20, 10)
    start_point = st.slider("起始点", -4.0, 4.0, 3.0, 0.1)
    
    # 梯度下降过程
    x_history = [start_point]
    x_current = start_point
    
    for _ in range(iterations):
        x_current = x_current - learning_rate * df(x_current)
        x_history.append(x_current)
    
    # 绘制函数和梯度下降过程
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制函数
    x = np.linspace(-5, 5, 100)
    y = f(x)
    ax.plot(x, y, 'b-', label='f(x) = x²+2')
    
    # 绘制梯度下降路径
    for i in range(len(x_history)-1):
        ax.plot(x_history[i], f(x_history[i]), 'ro')
        ax.arrow(x_history[i], f(x_history[i]), 
                x_history[i+1] - x_history[i], 
                f(x_history[i+1]) - f(x_history[i]),
                head_width=0.1, head_length=0.1, fc='r', ec='r')
    
    ax.plot(x_history[-1], f(x_history[-1]), 'ro')
    
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title('梯度下降优化过程')
    ax.legend()
    ax.grid(True)
    
    st.pyplot(fig)
    
    # 结果表格
    st.markdown("### 优化过程")
    
    df_history = pd.DataFrame({
        '迭代': list(range(len(x_history))),
        '参数值': [f"{x:.4f}" for x in x_history],
        '函数值': [f"{f(x):.4f}" for x in x_history],
        '梯度': [f"{df(x):.4f}" for x in x_history]
    })
    
    st.dataframe(df_history)
    
    # 练习2：简单神经网络的反向传播
    st.markdown("""
    ## 练习2：简单神经网络的反向传播
    
    反向传播是计算神经网络中梯度的高效算法。它首先进行前向传播计算输出，
    然后反向计算每一层的梯度。
    """)
    
    st.markdown("""
    ### 单层神经网络的反向传播
    
    考虑一个简单的单层神经网络：
    
    $$z = Wx + b$$
    $$a = \sigma(z)$$
    
    假设使用均方误差损失：
    
    $$L = \\frac{1}{2}(a - y)^2$$
    
    反向传播计算梯度的步骤：
    
    1. $\\frac{\\partial L}{\\partial a} = (a - y)$
    2. $\\frac{\\partial L}{\\partial z} = \\frac{\\partial L}{\\partial a} \\cdot \\frac{\\partial a}{\\partial z} = (a - y) \\cdot \sigma'(z)$
    3. $\\frac{\\partial L}{\\partial W} = \\frac{\\partial L}{\\partial z} \\cdot \\frac{\\partial z}{\\partial W} = (a - y) \\cdot \sigma'(z) \\cdot x$
    4. $\\frac{\\partial L}{\\partial b} = \\frac{\\partial L}{\\partial z} \\cdot \\frac{\\partial z}{\\partial b} = (a - y) \\cdot \sigma'(z)$
    """)
    
    # 可视化演示
    st.markdown("### 反向传播可视化演示")
    
    # 生成一个简单的回归数据集
    X, y_true = generate_regression_data(n_samples=50, noise=0.5)
    
    # 单层神经网络的参数
    W_init = st.slider("初始权重", -2.0, 2.0, 0.5, 0.1)
    b_init = st.slider("初始偏置", -2.0, 2.0, 0.0, 0.1)
    bp_learning_rate = st.slider("反向传播学习率", 0.01, 0.5, 0.1, 0.01, key="bp_lr")
    bp_iterations = st.slider("训练迭代次数", 10, 100, 50, 5, key="bp_iter")
    
    # 激活函数和其导数
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(x):
        s = sigmoid(x)
        return s * (1 - s)
    
    # 前向传播
    def forward(X, W, b):
        z = X * W + b
        a = sigmoid(z)
        return a, z
    
    # 损失函数
    def compute_loss(a, y):
        return 0.5 * np.mean((a - y) ** 2)
    
    # 反向传播训练过程
    W = W_init
    b = b_init
    loss_history = []
    W_history = []
    b_history = []
    
    for i in range(bp_iterations):
        # 前向传播
        a, z = forward(X, W, b)
        
        # 计算损失
        loss = compute_loss(a, y_true)
        loss_history.append(loss)
        W_history.append(W)
        b_history.append(b)
        
        # 计算梯度
        dL_da = a - y_true  # 损失对激活值的梯度
        dL_dz = dL_da * sigmoid_derivative(z)  # 链式法则
        dL_dW = np.mean(dL_dz * X)  # 损失对权重的梯度
        dL_db = np.mean(dL_dz)  # 损失对偏置的梯度
        
        # 更新参数
        W = W - bp_learning_rate * dL_dW
        b = b - bp_learning_rate * dL_db
    
    # 绘制训练结果
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 绘制数据和模型预测
    a_final, _ = forward(X, W, b)
    
    ax1.scatter(X, y_true, label='真实数据')
    ax1.plot(X, a_final, 'r-', label='模型预测')
    ax1.set_xlabel('输入 x')
    ax1.set_ylabel('输出 y')
    ax1.set_title('拟合结果')
    ax1.legend()
    ax1.grid(True)
    
    # 绘制损失变化
    ax2.plot(loss_history)
    ax2.set_xlabel('迭代次数')
    ax2.set_ylabel('损失')
    ax2.set_title('训练损失曲线')
    ax2.grid(True)
    
    st.pyplot(fig)
    
    st.markdown(f"""
    ### 训练结果
    
    **最终参数**:
    - 权重 W = {W:.4f}
    - 偏置 b = {b:.4f}
    
    **最终损失**: {loss_history[-1]:.4f}
    """)
    
    # 练习3：不同优化器比较
    st.markdown("""
    ## 练习3：不同优化器比较
    
    梯度下降有多种变体，如随机梯度下降(SGD)、带动量的SGD、RMSprop和Adam等。
    每种优化器都有其优缺点。
    """)
    
    st.markdown("""
    ### 常见优化器
    
    1. **SGD**：最基本的优化器，简单但可能收敛慢。
       $$\\theta_{t+1} = \\theta_t - \\alpha \\nabla J(\\theta_t)$$
    
    2. **带动量的SGD**：加入历史梯度信息，加速收敛。
       $$v_{t+1} = \\beta v_t + (1-\\beta)\\nabla J(\\theta_t)$$
       $$\\theta_{t+1} = \\theta_t - \\alpha v_{t+1}$$
    
    3. **Adam**：自适应学习率，结合了动量和RMSprop的优点。
       $$m_t = \\beta_1 m_{t-1} + (1-\\beta_1)\\nabla J(\\theta_t)$$
       $$v_t = \\beta_2 v_{t-1} + (1-\\beta_2)(\\nabla J(\\theta_t))^2$$
       $$\\hat{m}_t = \\frac{m_t}{1-\\beta_1^t}, \\hat{v}_t = \\frac{v_t}{1-\\beta_2^t}$$
       $$\\theta_{t+1} = \\theta_t - \\alpha \\frac{\\hat{m}_t}{\\sqrt{\\hat{v}_t} + \\epsilon}$$
    """)
    
    # 优化器可视化
    st.markdown("### 优化器性能比较")
    
    # 定义一个简单的2D函数进行优化
    def func_2d(x, y):
        return x**2 + 5*y**2 + 0.1 * np.sin(x*10) + 0.1 * np.cos(y*10)
    
    def grad_func_2d(x, y):
        dx = 2*x + 0.1*10*np.cos(x*10)
        dy = 10*y - 0.1*10*np.sin(y*10)
        return np.array([dx, dy])
    
    # 用户选择优化器
    optimizer = st.selectbox(
        "选择优化器",
        ["SGD", "带动量的SGD", "Adam"]
    )
    
    # 优化器参数
    opt_learning_rate = st.slider("优化器学习率", 0.01, 0.3, 0.1, 0.01, key="opt_lr")
    opt_iterations = st.slider("优化迭代次数", 10, 100, 30, 5, key="opt_iter")
    
    # 初始点
    start_x = st.slider("初始x", -2.0, 2.0, 1.5, 0.1)
    start_y = st.slider("初始y", -2.0, 2.0, 1.5, 0.1)
    
    # 不同优化器的实现
    def sgd(grad_func, start_pos, lr, iterations):
        pos = np.array(start_pos)
        path = [pos.copy()]
        
        for _ in range(iterations):
            grad = grad_func(pos[0], pos[1])
            pos = pos - lr * grad
            path.append(pos.copy())
            
        return np.array(path)
    
    def sgd_momentum(grad_func, start_pos, lr, iterations, beta=0.9):
        pos = np.array(start_pos)
        path = [pos.copy()]
        velocity = np.zeros_like(pos)
        
        for _ in range(iterations):
            grad = grad_func(pos[0], pos[1])
            velocity = beta * velocity + (1 - beta) * grad
            pos = pos - lr * velocity
            path.append(pos.copy())
            
        return np.array(path)
    
    def adam(grad_func, start_pos, lr, iterations, beta1=0.9, beta2=0.999, epsilon=1e-8):
        pos = np.array(start_pos)
        path = [pos.copy()]
        m = np.zeros_like(pos)
        v = np.zeros_like(pos)
        
        for t in range(1, iterations + 1):
            grad = grad_func(pos[0], pos[1])
            
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad**2
            
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            
            pos = pos - lr * m_hat / (np.sqrt(v_hat) + epsilon)
            path.append(pos.copy())
            
        return np.array(path)
    
    # 运行选定的优化器
    if optimizer == "SGD":
        path = sgd(grad_func_2d, [start_x, start_y], opt_learning_rate, opt_iterations)
    elif optimizer == "带动量的SGD":
        path = sgd_momentum(grad_func_2d, [start_x, start_y], opt_learning_rate, opt_iterations)
    else:  # Adam
        path = adam(grad_func_2d, [start_x, start_y], opt_learning_rate, opt_iterations)
    
    # 绘制结果
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 生成网格数据
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    Z = func_2d(X, Y)
    
    # 绘制等高线
    contour = ax.contour(X, Y, Z, 20, cmap='viridis', alpha=0.6)
    ax.clabel(contour, inline=True, fontsize=8)
    
    # 绘制优化路径
    ax.plot(path[:, 0], path[:, 1], 'ro-', linewidth=1.5, markersize=3)
    ax.plot(path[0, 0], path[0, 1], 'go', markersize=6, label='起始点')
    ax.plot(path[-1, 0], path[-1, 1], 'bo', markersize=6, label='终点')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'{optimizer}优化路径')
    ax.legend()
    ax.grid(True)
    
    st.pyplot(fig)
    
    # 显示优化结果
    st.markdown(f"""
    ### 优化结果
    
    **起始点**: ({start_x:.2f}, {start_y:.2f})
    
    **终点**: ({path[-1, 0]:.4f}, {path[-1, 1]:.4f})
    
    **终点函数值**: {func_2d(path[-1, 0], path[-1, 1]):.6f}
    
    **迭代次数**: {opt_iterations}
    """)
    
    # 思考题
    st.markdown("""
    ## 思考题
    
    1. 学习率过大或过小会有什么影响？
    
    2. 为什么需要不同的优化器？各有什么优缺点？
    
    3. 如何判断训练过程中是否出现了梯度消失或梯度爆炸问题？
    
    4. 深度神经网络中的反向传播算法可能面临什么计算挑战？
    """)
    
    # 思考题提示
    with st.expander("查看思考题提示"):
        st.markdown("""
        **思考题提示**：
        
        1. **学习率影响**:
           - 过大：可能导致振荡或发散，错过最优解
           - 过小：收敛速度慢，可能陷入局部最优
        
        2. **不同优化器比较**:
           - SGD: 简单但收敛慢，容易陷入局部最优
           - 带动量的SGD: 加速收敛，更容易跳出局部最优
           - Adam: 自适应学习率，适应不同参数的更新需求，通常效果最好但计算开销大
        
        3. **梯度问题判断**:
           - 梯度消失：深层网络中梯度接近0，参数几乎不更新
           - 梯度爆炸：梯度值非常大，参数更新剧烈，损失可能突然增大
        
        4. **深度网络反向传播挑战**:
           - 计算量大，特别是大规模网络
           - 梯度消失/爆炸问题
           - 需要存储中间激活值占用内存
           - 并行计算挑战
        """)
    
    # 小结
    st.markdown("""
    ## 小结
    
    在本练习中，你：
    1. 理解了梯度下降的基本原理
    2. 学习了神经网络中的反向传播算法
    3. 比较了不同优化器的性能
    4. 观察了优化过程的收敛行为
    
    这些是深度学习训练的核心概念，对于理解神经网络如何学习至关重要。
    """)

if __name__ == "__main__":
    optimizer_backprop_exercise() 