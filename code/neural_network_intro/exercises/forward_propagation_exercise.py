import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils.visualization_utils import plot_activation_functions

def forward_propagation_exercise():
    """前向传播练习页面"""
    st.title("练习：前向传播")
    
    st.markdown("""
    ## 练习目标
    
    通过本练习，你将：
    1. 理解神经网络中的前向传播过程
    2. 手动计算简单神经网络的前向传播
    3. 学习矩阵化实现前向传播
    4. 观察不同激活函数的影响
    
    完成这些练习将帮助你理解神经网络如何从输入计算输出。
    """)
    
    # 练习1：单个神经元的前向传播
    st.markdown("""
    ## 练习1：单个神经元的前向传播
    
    首先，让我们观察单个神经元是如何计算其输出的。
    """)
    
    st.markdown("""
    ### 单个神经元的计算
    
    神经元计算公式：
    
    $$z = w_1 x_1 + w_2 x_2 + ... + w_n x_n + b$$
    
    $$a = \sigma(z)$$
    
    其中：
    - $w_i$ 是权重
    - $x_i$ 是输入
    - $b$ 是偏置
    - $\sigma$ 是激活函数
    - $z$ 是加权和
    - $a$ 是神经元输出
    """)
    
    # 单个神经元计算演示
    st.markdown("### 交互式单神经元计算")
    
    col1, col2 = st.columns(2)
    
    with col1:
        x1 = st.slider("输入 x₁", -10.0, 10.0, 2.0, 0.1)
        x2 = st.slider("输入 x₂", -10.0, 10.0, -1.0, 0.1)
        w1 = st.slider("权重 w₁", -2.0, 2.0, 0.5, 0.1)
        w2 = st.slider("权重 w₂", -2.0, 2.0, -0.5, 0.1)
    
    with col2:
        b = st.slider("偏置 b", -10.0, 10.0, 1.0, 0.1)
        activation = st.selectbox(
            "激活函数",
            ["Sigmoid", "ReLU", "Tanh", "线性"]
        )
    
    # 计算加权和
    z = w1 * x1 + w2 * x2 + b
    
    # 应用激活函数
    if activation == "Sigmoid":
        a = 1 / (1 + np.exp(-z))
        formula = "σ(z) = 1 / (1 + e^(-z))"
    elif activation == "ReLU":
        a = max(0, z)
        formula = "σ(z) = max(0, z)"
    elif activation == "Tanh":
        a = np.tanh(z)
        formula = "σ(z) = tanh(z)"
    else:  # 线性
        a = z
        formula = "σ(z) = z"
    
    # 显示计算过程和结果
    st.markdown(f"""
    ### 计算过程
    
    1. **计算加权和**:
       $z = w_1 x_1 + w_2 x_2 + b$
       $z = ({w1}) \\times ({x1}) + ({w2}) \\times ({x2}) + ({b})$
       $z = {w1 * x1} + ({w2 * x2}) + ({b})$
       $z = {z}$
    
    2. **应用激活函数** ({activation}):
       ${formula}$
       $a = {a}$
    
    **神经元输出**: ${a:.4f}$
    """)
    
    # 可视化激活函数
    st.markdown("### 激活函数曲线")
    
    fig = plot_activation_functions(z_value=z)
    st.pyplot(fig)
    
    # 练习2：单层前向传播
    st.markdown("""
    ## 练习2：单层前向传播
    
    现在，让我们考虑一个包含多个神经元的单层网络。
    """)
    
    st.markdown("""
    ### 矩阵形式的前向传播
    
    对于一层包含多个神经元的网络，我们可以使用矩阵运算来高效计算：
    
    $$Z = XW^T + b$$
    $$A = \sigma(Z)$$
    
    其中：
    - $X$ 是输入矩阵（每行是一个样本，每列是一个特征）
    - $W$ 是权重矩阵（每行对应一个神经元的权重）
    - $b$ 是偏置向量
    - $Z$ 是加权和矩阵
    - $A$ 是激活值矩阵
    - $\sigma$ 是激活函数（按元素应用）
    """)
    
    # 单层网络前向传播演示
    st.markdown("### 交互式单层网络计算")
    
    # 设置参数
    n_inputs = st.slider("输入维度", 2, 5, 3)
    n_neurons = st.slider("神经元数量", 1, 5, 2)
    
    # 生成随机输入、权重和偏置
    np.random.seed(42)  # 固定随机种子以便重现结果
    
    X = np.random.randn(1, n_inputs)  # 单个样本
    W = np.random.randn(n_neurons, n_inputs)
    b = np.random.randn(n_neurons)
    
    # 显示输入
    st.markdown("#### 输入向量")
    X_df = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(n_inputs)])
    st.write(X_df)
    
    # 显示权重矩阵
    st.markdown("#### 权重矩阵")
    W_df = pd.DataFrame(W, 
                       index=[f"神经元 {i+1}" for i in range(n_neurons)],
                       columns=[f"w{i+1}" for i in range(n_inputs)])
    st.write(W_df)
    
    # 显示偏置向量
    st.markdown("#### 偏置向量")
    b_df = pd.DataFrame(b.reshape(1, -1), 
                        columns=[f"b{i+1}" for i in range(n_neurons)])
    st.write(b_df)
    
    # 选择激活函数
    layer_activation = st.selectbox(
        "层激活函数",
        ["Sigmoid", "ReLU", "Tanh", "线性"],
        key="layer_activation"
    )
    
    # 计算前向传播
    Z = np.dot(X, W.T) + b
    
    # 应用激活函数
    if layer_activation == "Sigmoid":
        A = 1 / (1 + np.exp(-Z))
    elif layer_activation == "ReLU":
        A = np.maximum(0, Z)
    elif layer_activation == "Tanh":
        A = np.tanh(Z)
    else:  # 线性
        A = Z
    
    # 显示计算结果
    st.markdown("#### 加权和")
    Z_df = pd.DataFrame(Z, columns=[f"z{i+1}" for i in range(n_neurons)])
    st.write(Z_df)
    
    st.markdown("#### 激活值（输出）")
    A_df = pd.DataFrame(A, columns=[f"a{i+1}" for i in range(n_neurons)])
    st.write(A_df)
    
    # 可视化单层网络
    st.markdown("### 单层网络可视化")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制神经元和连接
    input_pos = [(0, i) for i in range(n_inputs)]
    neuron_pos = [(1, i) for i in range(n_neurons)]
    
    # 绘制输入节点
    for i, pos in enumerate(input_pos):
        circle = plt.Circle(pos, 0.2, color='skyblue', alpha=0.8)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], f"x{i+1}\n{X[0, i]:.2f}", ha='center', va='center', fontsize=9)
    
    # 绘制神经元
    for i, pos in enumerate(neuron_pos):
        circle = plt.Circle(pos, 0.2, color='lightgreen', alpha=0.8)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], f"a{i+1}\n{A[0, i]:.2f}", ha='center', va='center', fontsize=9)
    
    # 绘制连接
    for i, in_pos in enumerate(input_pos):
        for j, neuron_pos in enumerate(neuron_pos):
            # 基于权重决定线条颜色和粗细
            weight = W[j, i]
            width = abs(weight) * 2
            color = 'red' if weight < 0 else 'blue'
            alpha = min(abs(weight), 1.0)
            
            ax.plot([in_pos[0] + 0.2, neuron_pos[0] - 0.2], 
                    [in_pos[1], neuron_pos[1]], 
                    color=color, alpha=alpha, linewidth=width)
            
            # 在连接线中间添加权重标签
            mid_x = (in_pos[0] + 0.2 + neuron_pos[0] - 0.2) / 2
            mid_y = (in_pos[1] + neuron_pos[1]) / 2
            ax.text(mid_x, mid_y, f"{weight:.2f}", ha='center', va='center', 
                   fontsize=8, bbox=dict(facecolor='white', alpha=0.7))
    
    # 标记偏置
    for i, pos in enumerate(neuron_pos):
        bias = b[i]
        ax.text(pos[0], pos[1] + 0.3, f"b{i+1}={bias:.2f}", ha='center', va='center', 
               fontsize=8, bbox=dict(facecolor='lightyellow', alpha=0.7))
    
    # 设置图表
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, n_inputs-0.5 if n_inputs > n_neurons else n_neurons-0.5)
    ax.set_title(f"单层神经网络 ({layer_activation}激活)")
    ax.set_aspect('equal')
    ax.axis('off')
    
    st.pyplot(fig)
    
    # 练习3：多层前向传播
    st.markdown("""
    ## 练习3：多层前向传播
    
    最后，让我们实现一个具有多层的神经网络的前向传播。
    """)
    
    st.markdown("""
    ### 多层网络的前向传播
    
    对于多层神经网络，前向传播是一个递归过程：
    
    1. 第一层接收原始输入 $X$
    2. 每一层的输出成为下一层的输入
    3. 最后一层产生网络的最终输出
    
    每一层的计算过程为：
    
    $$Z^{[l]} = A^{[l-1]} W^{[l]T} + b^{[l]}$$
    $$A^{[l]} = \sigma^{[l]}(Z^{[l]})$$
    
    其中 $l$ 是层索引，$A^{[0]} = X$
    """)
    
    # 多层网络演示
    st.markdown("### 交互式多层网络计算")
    
    # 定义网络结构
    st.markdown("#### 定义网络结构")
    
    layers = []
    n_layers = st.slider("网络层数", 2, 4, 3)
    
    # 第一层输入维度
    n_inputs_multi = st.slider("输入维度", 2, 5, 3, key="multi_input")
    
    # 每层神经元数量
    for i in range(n_layers):
        if i == n_layers - 1:
            layer_name = "输出层"
        else:
            layer_name = f"隐藏层 {i+1}"
        
        n_neurons_layer = st.slider(
            f"{layer_name}神经元数量", 
            1, 5, 
            2 if i < n_layers - 1 else 1
        )
        
        activation_layer = st.selectbox(
            f"{layer_name}激活函数",
            ["Sigmoid", "ReLU", "Tanh", "线性"],
            index=1 if i < n_layers - 1 else 0,
            key=f"act_{i}"
        )
        
        layers.append({
            "name": layer_name,
            "neurons": n_neurons_layer,
            "activation": activation_layer
        })
    
    # 生成随机输入和参数
    st.markdown("#### 随机生成网络参数")
    
    # 固定随机种子
    np.random.seed(42)
    
    # 生成输入
    X_multi = np.random.randn(1, n_inputs_multi)
    
    # 初始化参数字典
    parameters = {}
    
    # 生成每层的权重和偏置
    prev_layer_size = n_inputs_multi
    for i, layer in enumerate(layers):
        layer_idx = i + 1
        n_neurons_layer = layer["neurons"]
        
        parameters[f"W{layer_idx}"] = np.random.randn(n_neurons_layer, prev_layer_size) * 0.1
        parameters[f"b{layer_idx}"] = np.random.randn(n_neurons_layer) * 0.1
        
        prev_layer_size = n_neurons_layer
    
    if st.checkbox("显示网络参数"):
        for i, layer in enumerate(layers):
            layer_idx = i + 1
            st.markdown(f"##### {layer['name']}参数")
            
            st.markdown(f"W{layer_idx}:")
            st.write(parameters[f"W{layer_idx}"])
            
            st.markdown(f"b{layer_idx}:")
            st.write(parameters[f"b{layer_idx}"])
    
    # 前向传播计算
    st.markdown("#### 前向传播计算")
    
    # 存储每一层的值
    cache = {"A0": X_multi}
    
    # 显示输入
    st.markdown("输入:")
    st.write(pd.DataFrame(X_multi, columns=[f"x{i+1}" for i in range(n_inputs_multi)]))
    
    # 对每一层进行前向传播
    for i, layer in enumerate(layers):
        layer_idx = i + 1
        W = parameters[f"W{layer_idx}"]
        b = parameters[f"b{layer_idx}"]
        A_prev = cache[f"A{layer_idx-1}"]
        
        # 计算加权和
        Z = np.dot(A_prev, W.T) + b
        cache[f"Z{layer_idx}"] = Z
        
        # 应用激活函数
        activation = layer["activation"]
        if activation == "Sigmoid":
            A = 1 / (1 + np.exp(-Z))
        elif activation == "ReLU":
            A = np.maximum(0, Z)
        elif activation == "Tanh":
            A = np.tanh(Z)
        else:  # 线性
            A = Z
        
        cache[f"A{layer_idx}"] = A
        
        # 显示该层计算结果
        st.markdown(f"##### {layer['name']}输出 (使用{activation}激活)")
        st.write(pd.DataFrame(
            cache[f"A{layer_idx}"], 
            columns=[f"a{j+1}" for j in range(layer["neurons"])]
        ))
    
    # 获取最终输出
    output = cache[f"A{n_layers}"]
    
    # 可视化多层网络
    st.markdown("### 多层网络可视化")
    
    # 创建网络结构的可视化
    layer_sizes = [n_inputs_multi] + [layer["neurons"] for layer in layers]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 计算每一层中的最大神经元数量，以便设置图的纵向尺寸
    max_neurons = max(layer_sizes)
    
    # 存储所有神经元的位置
    neuron_positions = {}
    
    # 绘制每一层的神经元
    for l, layer_size in enumerate(layer_sizes):
        for n in range(layer_size):
            # 计算神经元位置，使每层居中
            x = l
            y = n - (layer_size - 1) / 2
            
            # 存储位置
            neuron_positions[(l, n)] = (x, y)
            
            # 设置节点颜色和标签
            if l == 0:
                color = 'skyblue'
                label = f"x{n+1}\n{X_multi[0, n]:.2f}"
            elif l == len(layer_sizes) - 1:
                color = 'salmon'
                label = f"y{n+1}\n{output[0, n]:.2f}"
            else:
                color = 'lightgreen'
                label = f"h{l}{n+1}\n{cache[f'A{l}'][0, n]:.2f}"
            
            # 绘制神经元
            circle = plt.Circle((x, y), 0.2, color=color, alpha=0.8)
            ax.add_patch(circle)
            ax.text(x, y, label, ha='center', va='center', fontsize=9)
    
    # 绘制层间连接
    for l in range(len(layer_sizes) - 1):
        for i in range(layer_sizes[l]):
            for j in range(layer_sizes[l + 1]):
                # 获取源节点和目标节点位置
                source_pos = neuron_positions[(l, i)]
                target_pos = neuron_positions[(l + 1, j)]
                
                # 获取权重并计算连接线的粗细和颜色
                weight = parameters[f"W{l+1}"][j, i]
                width = abs(weight) * 2
                color = 'red' if weight < 0 else 'blue'
                alpha = min(abs(weight), 1.0)
                
                # 绘制连接线
                ax.plot([source_pos[0] + 0.2, target_pos[0] - 0.2], 
                        [source_pos[1], target_pos[1]], 
                        color=color, alpha=alpha, linewidth=width)
    
    # 设置图表
    ax.set_xlim(-0.5, len(layer_sizes) - 0.5)
    ax.set_ylim(-max_neurons/2 - 0.5, max_neurons/2 + 0.5)
    
    # 添加层标签
    layer_names = ["输入层"] + [f"隐藏层 {i+1}" if i < len(layers) - 1 else "输出层" for i in range(len(layers))]
    for i, name in enumerate(layer_names):
        ax.text(i, max_neurons/2 + 0.3, name, ha='center', va='center', fontsize=11)
    
    ax.set_title("多层神经网络前向传播")
    ax.axis('off')
    
    st.pyplot(fig)
    
    # 练习4：编写前向传播函数
    st.markdown("""
    ## 练习4：编写前向传播函数
    
    尝试将上述前向传播过程封装成可复用的函数。
    """)
    
    # 前向传播函数模板
    st.code("""
    def forward_propagation(X, parameters, activations):
        '''
        实现多层神经网络的前向传播
        
        参数:
            X: 输入数据，形状为 (m, n_x)
            parameters: 包含所有参数的字典 {"W1": W1, "b1": b1, ...}
            activations: 每一层使用的激活函数列表 ["relu", "sigmoid", ...]
            
        返回:
            AL: 最后一层的激活值（模型的输出）
            caches: 包含每一层的线性和激活缓存的列表
        '''
        caches = []
        A = X
        L = len(parameters) // 2  # 参数字典包含 W1,b1,W2,b2,...
        
        # 实现前向传播
        for l in range(1, L + 1):
            A_prev = A
            
            # 获取当前层的参数
            W = parameters['W' + str(l)]
            b = parameters['b' + str(l)]
            
            # 计算线性部分
            Z = np.dot(A_prev, W.T) + b
            
            # 获取当前层的激活函数
            activation = activations[l-1]
            
            # 应用激活函数
            if activation == "sigmoid":
                A = 1 / (1 + np.exp(-Z))
            elif activation == "relu":
                A = np.maximum(0, Z)
            elif activation == "tanh":
                A = np.tanh(Z)
            else:  # 线性
                A = Z
            
            # 保存缓存用于反向传播
            cache = {"Z": Z, "A": A}
            caches.append(cache)
        
        return A, caches
    """)
    
    # 思考题
    st.markdown("""
    ## 思考题
    
    1. 为什么在深度神经网络中使用矩阵化的前向传播比循环计算每个神经元的输出更高效？
    
    2. 不同的激活函数对模型的学习能力有何影响？
    
    3. 在前向传播过程中，我们需要保存哪些中间值用于后续的反向传播？为什么？
    
    4. 隐藏层的数量和每层神经元的数量如何影响模型的表达能力和计算复杂度？
    """)
    
    with st.expander("查看思考题提示"):
        st.markdown("""
        **思考题提示**：
        
        1. **矩阵运算效率**:
           - 矩阵运算可以并行处理
           - 现代硬件（如GPU）针对矩阵运算做了优化
           - 使用优化的线性代数库（如BLAS）可以大大提高计算速度
        
        2. **激活函数影响**:
           - Sigmoid/Tanh: 容易导致梯度消失问题
           - ReLU: 解决梯度消失，但可能导致神经元"死亡"
           - 不同的激活函数适合不同的问题域和网络层
        
        3. **保存的中间值**:
           - 需要保存每一层的激活值和加权和
           - 这些值在反向传播中用于计算梯度
           - 没有这些缓存值，反向传播将无法进行
        
        4. **网络结构影响**:
           - 更多的层和神经元增加了模型的表达能力
           - 计算复杂度随层数和神经元数量增加而增加
           - 需要在表达能力和计算效率之间找到平衡
        """)
    
    # 关键概念总结
    st.markdown("""
    ## 关键概念总结
    
    1. **前向传播是神经网络的核心计算过程**，将输入数据转换为预测输出。
    
    2. **单个神经元**执行加权和与非线性激活两步操作。
    
    3. **矩阵形式**使神经网络计算高效可并行化。
    
    4. **多层网络**通过层叠转换提取更复杂的特征。
    
    5. **激活函数**引入非线性，使网络能学习复杂模式。
    
    6. **缓存中间值**对于后续的反向传播和参数更新至关重要。
    """)

# 导入额外需要的库
import pandas as pd

if __name__ == "__main__":
    forward_propagation_exercise() 