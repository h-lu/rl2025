import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils.visualization import plot_network_layers
from utils.model_utils import forward_pass_np
import torch
import torch.nn as nn

def sigmoid(x):
    """Sigmoid激活函数"""
    return 1 / (1 + np.exp(-x))

def relu(x):
    """ReLU激活函数"""
    return np.maximum(0, x)

def tanh(x):
    """Tanh激活函数"""
    return np.tanh(x)

def weight_to_color(weight):
    """将权重值转换为颜色
    正权重: 蓝色
    负权重: 红色
    权重接近0: 灰色
    """
    if weight > 0:
        return f'rgba(0, 0, 255, {min(abs(weight), 1)})'
    else:
        return f'rgba(255, 0, 0, {min(abs(weight), 1)})'

def weight_to_width(weight):
    """将权重值转换为线宽
    权重绝对值越大，线越粗
    """
    return 1 + 2 * min(abs(weight), 1)

def show_feedforward_networks():
    """显示前馈神经网络架构页面"""
    st.title("神经网络架构")
    
    st.markdown("""
    ## 前馈神经网络
    
    前馈神经网络（Feedforward Neural Network）是最基本的深度学习模型架构，特点是信息只向前流动，没有循环或反馈连接。
    """)
    
    # 神经网络基本架构
    st.markdown("""
    ### 神经网络层次结构
    
    典型的前馈神经网络由以下几个部分组成：
    
    1. **输入层**：接收外部数据
    2. **隐藏层**：处理信息（可以有多个）
    3. **输出层**：产生最终结果
    
    每一层由多个神经元组成，层与层之间全连接。
    """)
    
    # 可视化神经网络结构
    st.markdown("### 神经网络结构可视化")
    
    # 允许用户通过滑块调整网络结构
    col1, col2, col3 = st.columns(3)
    
    with col1:
        input_size = st.slider("输入层神经元数量", 1, 10, 3)
    
    with col2:
        hidden_sizes = st.multiselect(
            "隐藏层配置 (每层神经元数量)",
            options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            default=[4, 3]
        )
    
    with col3:
        output_size = st.slider("输出层神经元数量", 1, 10, 2)
    
    # 如果没有选择隐藏层，提供默认值
    if not hidden_sizes:
        hidden_sizes = [4]
    
    # 生成网络结构
    layer_sizes = [input_size] + hidden_sizes + [output_size]
    
    # 可视化网络结构
    network_fig = plot_network_layers(layer_sizes)
    st.pyplot(network_fig)
    
    # 层的类型和特点
    st.markdown("""
    ## 神经网络层的类型和特点
    
    ### 全连接层（Dense/Linear层）
    
    全连接层是最基本的神经网络层类型，其中一层的每个神经元与下一层的所有神经元相连。
    
    **数学表示**：
    
    对于输入向量 $\\mathbf{x}$，权重矩阵 $\\mathbf{W}$，和偏置向量 $\\mathbf{b}$：
    
    $$\\mathbf{z} = \\mathbf{W} \\mathbf{x} + \\mathbf{b}$$
    
    然后应用激活函数：
    
    $$\\mathbf{a} = f(\\mathbf{z})$$
    """)
    
    # 神经网络的矩阵表示
    with st.expander("神经网络的矩阵表示"):
        st.markdown("""
        ### 矩阵形式的网络计算
        
        对于一个具有 $n$ 个输入，$m$ 个输出的全连接层，我们可以用矩阵乘法表示：
        
        $$
        \\begin{bmatrix} 
        z_1 \\\\ 
        z_2 \\\\ 
        \\vdots \\\\ 
        z_m 
        \\end{bmatrix} = 
        \\begin{bmatrix} 
        w_{11} & w_{12} & \\cdots & w_{1n} \\\\ 
        w_{21} & w_{22} & \\cdots & w_{2n} \\\\ 
        \\vdots & \\vdots & \\ddots & \\vdots \\\\ 
        w_{m1} & w_{m2} & \\cdots & w_{mn} 
        \\end{bmatrix}
        \\begin{bmatrix} 
        x_1 \\\\ 
        x_2 \\\\ 
        \\vdots \\\\ 
        x_n 
        \\end{bmatrix} + 
        \\begin{bmatrix} 
        b_1 \\\\ 
        b_2 \\\\ 
        \\vdots \\\\ 
        b_m 
        \\end{bmatrix}
        $$
        
        使用矩阵乘法可以大大加速神经网络的计算，特别是在GPU上。
        """)
    
    # 前向传播过程
    st.markdown("""
    ## 前向传播
    
    前向传播（Forward Propagation）是神经网络处理输入数据、生成预测的过程。
    
    ### 前向传播步骤
    
    1. 输入数据通过输入层
    2. 数据在每一层进行线性变换（加权和+偏置）
    3. 应用激活函数引入非线性
    4. 将结果传递给下一层
    5. 输出层产生最终结果
    """)
    
    # 交互式前向传播演示
    st.markdown("## 交互式前向传播演示")
    
    # 使用一个简单的小型网络进行演示
    st.markdown("""
    下面是一个简单的神经网络前向传播演示。您可以输入数据，查看数据如何通过网络传播并产生输出。
    
    我们将使用一个具有以下结构的小型网络：
    - 输入层：2个神经元
    - 隐藏层：3个神经元
    - 输出层：2个神经元
    """)
    
    # 用户输入
    col1, col2 = st.columns(2)
    
    with col1:
        x1 = st.slider("输入 x₁", -1.0, 1.0, 0.5, 0.1)
    
    with col2:
        x2 = st.slider("输入 x₂", -1.0, 1.0, -0.2, 0.1)
    
    # 选择激活函数
    activation = st.selectbox(
        "选择激活函数",
        ["Sigmoid", "ReLU", "Tanh"]
    )
    
    # 设置网络参数（为简化演示，使用固定权重）
    # 权重和偏置
    weights = [
        np.array([[0.1, 0.2],
                  [0.3, -0.1],
                  [-0.2, 0.4]]),  # 输入层到隐藏层
        np.array([[0.5, -0.3, 0.2],
                  [-0.1, 0.2, 0.4]])   # 隐藏层到输出层
    ]
    
    biases = [
        np.array([0.1, 0.2, -0.1]),  # 隐藏层偏置
        np.array([0.0, 0.3])         # 输出层偏置
    ]
    
    # 模拟前向传播
    input_data = np.array([x1, x2])
    
    # 使用函数进行前向传播
    activations, layer_inputs = forward_pass_np(
        input_data, weights, biases, 
        activation_func=activation.lower()
    )
    
    # 显示前向传播的结果
    st.markdown("### 前向传播过程")
    
    # 输入层
    st.markdown("**输入层**")
    st.markdown(f"输入: $\\mathbf{{x}} = [{x1:.2f}, {x2:.2f}]$")
    
    # 隐藏层
    st.markdown("**隐藏层**")
    z_hidden = layer_inputs[0]
    a_hidden = activations[1]
    
    # 显示计算细节
    st.markdown(f"""
    线性变换:
    
    $\\mathbf{{z}}^{{[1]}}_1 = (0.1 \\times {x1:.2f}) + (0.2 \\times {x2:.2f}) + 0.1 = {z_hidden[0][0]:.4f}$
    
    $\\mathbf{{z}}^{{[1]}}_2 = (0.3 \\times {x1:.2f}) + (-0.1 \\times {x2:.2f}) + 0.2 = {z_hidden[0][1]:.4f}$
    
    $\\mathbf{{z}}^{{[1]}}_3 = (-0.2 \\times {x1:.2f}) + (0.4 \\times {x2:.2f}) - 0.1 = {z_hidden[0][2]:.4f}$
    """)
    
    # 应用激活函数
    st.markdown(f"""
    应用{activation}激活函数:
    
    $\\mathbf{{a}}^{{[1]}}_1 = {activation}({z_hidden[0][0]:.4f}) = {a_hidden[0][0]:.4f}$
    
    $\\mathbf{{a}}^{{[1]}}_2 = {activation}({z_hidden[0][1]:.4f}) = {a_hidden[0][1]:.4f}$
    
    $\\mathbf{{a}}^{{[1]}}_3 = {activation}({z_hidden[0][2]:.4f}) = {a_hidden[0][2]:.4f}$
    """)
    
    # 输出层
    st.markdown("**输出层**")
    z_output = layer_inputs[1]
    a_output = activations[2]
    
    # 显示计算细节
    st.markdown(f"""
    线性变换:
    
    $\\mathbf{{z}}^{{[2]}}_1 = (0.5 \\times {a_hidden[0][0]:.4f}) + (-0.3 \\times {a_hidden[0][1]:.4f}) + (0.2 \\times {a_hidden[0][2]:.4f}) + 0.0 = {z_output[0][0]:.4f}$
    
    $\\mathbf{{z}}^{{[2]}}_2 = (-0.1 \\times {a_hidden[0][0]:.4f}) + (0.2 \\times {a_hidden[0][1]:.4f}) + (0.4 \\times {a_hidden[0][2]:.4f}) + 0.3 = {z_output[0][1]:.4f}$
    """)
    
    # 应用激活函数
    st.markdown(f"""
    应用{activation}激活函数:
    
    $\\mathbf{{a}}^{{[2]}}_1 = {activation}({z_output[0][0]:.4f}) = {a_output[0][0]:.4f}$
    
    $\\mathbf{{a}}^{{[2]}}_2 = {activation}({z_output[0][1]:.4f}) = {a_output[0][1]:.4f}$
    """)
    
    # 最终输出
    st.markdown("**最终输出**")
    st.markdown(f"输出: $\\mathbf{{y}} = [{a_output[0][0]:.4f}, {a_output[0][1]:.4f}]$")
    
    # 可视化前向传播过程
    st.markdown("### 可视化前向传播")
    
    # 创建图像
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 节点位置
    layer_positions = [0, 1, 2]  # x坐标
    
    # 输入层节点
    input_y = [0.25, 0.75]
    
    # 隐藏层节点
    hidden_y = [0.167, 0.5, 0.833]
    
    # 输出层节点
    output_y = [0.33, 0.67]
    
    # 绘制节点
    # 输入层
    for i, y in enumerate(input_y):
        ax.add_patch(plt.Circle((layer_positions[0], y), 0.05, color='lightblue'))
        ax.text(layer_positions[0], y, f"{activations[0][i] if i < len(activations[0]) else 0:.2f}", ha='center', va='center')
    
    # 隐藏层
    for i, y in enumerate(hidden_y):
        ax.add_patch(plt.Circle((layer_positions[1], y), 0.05, color='lightgreen'))
        ax.text(layer_positions[1], y, f"{a_hidden[0][i] if i < len(a_hidden[0]) else 0:.2f}", ha='center', va='center')
    
    # 输出层
    for i, y in enumerate(output_y):
        ax.add_patch(plt.Circle((layer_positions[2], y), 0.05, color='salmon'))
        ax.text(layer_positions[2], y, f"{a_output[0][i] if i < len(a_output[0]) else 0:.2f}", ha='center', va='center')
    
    # 绘制连接线（权重）
    for i, y_in in enumerate(input_y):
        for j, y_hid in enumerate(hidden_y):
            weight = weights[0][j][i]
            ax.plot([layer_positions[0], layer_positions[1]], [y_in, y_hid], 
                   color=weight_to_color(weight), lw=weight_to_width(weight), alpha=0.7)
            ax.text((layer_positions[0] + layer_positions[1])/2, 
                   (y_in + y_hid)/2, f"{weight:.1f}", ha='center', va='center',
                   bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, boxstyle='round,pad=0.1'))
    
    for i, y_hid in enumerate(hidden_y):
        for j, y_out in enumerate(output_y):
            if i < len(a_hidden[0]) and j < len(a_output[0]):  # 确保索引有效
                weight = weights[1][j][i]
                ax.plot([layer_positions[1], layer_positions[2]], [y_hid, y_out], 
                      color=weight_to_color(weight), lw=weight_to_width(weight), alpha=0.7)
                ax.text((layer_positions[1] + layer_positions[2])/2, 
                      (y_hid + y_out)/2, f"{weight:.1f}", ha='center', va='center',
                      bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, boxstyle='round,pad=0.1'))
    
    # 添加标签
    ax.text(layer_positions[0], 1.05, "输入层", ha='center', va='bottom')
    ax.text(layer_positions[1], 1.05, "隐藏层", ha='center', va='bottom')
    ax.text(layer_positions[2], 1.05, "输出层", ha='center', va='bottom')
    
    # 设置图表范围和标签
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0, 1)
    ax.set_xticks(layer_positions)
    ax.set_xticklabels(['输入层', '隐藏层', '输出层'])
    ax.set_yticks([])
    ax.set_title('神经网络前向传播可视化')
    
    st.pyplot(fig)
    
    # PyTorch实现
    st.markdown("## PyTorch实现示例")
    
    with st.expander("查看PyTorch代码"):
        st.code("""
# 定义一个简单的前馈神经网络
class SimpleNetwork(nn.Module):
    def __init__(self):
        super(SimpleNetwork, self).__init__()
        # 输入层到隐藏层
        self.fc1 = nn.Linear(2, 3)
        # 隐藏层到输出层
        self.fc2 = nn.Linear(3, 2)
        # 激活函数
        self.activation = nn.ReLU()  # 可以根据需要更改
        
    def forward(self, x):
        # 输入层到隐藏层
        hidden = self.fc1(x)
        hidden = self.activation(hidden)
        
        # 隐藏层到输出层
        output = self.fc2(hidden)
        
        return output

# 创建模型
model = SimpleNetwork()

# 准备输入数据
inputs = torch.tensor([[0.5, -0.2]], dtype=torch.float32)

# 前向传播
outputs = model(inputs)
print(f"输出: {outputs}")
        """, language="python")
    
    # 使用PyTorch实现前向传播
    st.markdown("### 使用PyTorch执行前向传播")
    
    # 创建一个与上面相同结构的网络
    class SimpleNetwork(nn.Module):
        def __init__(self, activation='relu'):
            super(SimpleNetwork, self).__init__()
            # 输入层到隐藏层
            self.fc1 = nn.Linear(2, 3)
            # 隐藏层到输出层
            self.fc2 = nn.Linear(3, 2)
            
            # 设置激活函数
            if activation.lower() == 'relu':
                self.activation = nn.ReLU()
            elif activation.lower() == 'sigmoid':
                self.activation = nn.Sigmoid()
            elif activation.lower() == 'tanh':
                self.activation = nn.Tanh()
            else:
                self.activation = nn.ReLU()
        
        def forward(self, x):
            # 输入层到隐藏层
            hidden = self.fc1(x)
            hidden = self.activation(hidden)
            
            # 隐藏层到输出层
            output = self.fc2(hidden)
            output = self.activation(output)
            
            return output
    
    # 创建PyTorch网络
    torch_model = SimpleNetwork(activation.lower())
    
    # 手动设置权重和偏置，使其与前面的数值计算一致
    with torch.no_grad():
        torch_model.fc1.weight.copy_(torch.tensor(weights[0], dtype=torch.float32))
        torch_model.fc1.bias.copy_(torch.tensor(biases[0], dtype=torch.float32))
        torch_model.fc2.weight.copy_(torch.tensor(weights[1], dtype=torch.float32))
        torch_model.fc2.bias.copy_(torch.tensor(biases[1], dtype=torch.float32))
    
    # 执行前向传播
    input_tensor = torch.tensor([x1, x2], dtype=torch.float32)
    output_tensor = torch_model(input_tensor)
    
    st.markdown(f"""
    **PyTorch输出**：
    
    输入：[{x1:.2f}, {x2:.2f}]
    
    输出：[{output_tensor[0].item():.4f}, {output_tensor[1].item():.4f}]
    """)
    
    # 神经网络架构的分类
    st.markdown("""
    ## 神经网络架构的多样性
    
    除了基本的全连接前馈网络，还有许多其他类型的神经网络架构，每种都适合特定类型的问题：
    
    ### 常见网络架构
    
    - **卷积神经网络 (CNN)**：适用于图像和空间数据
    - **循环神经网络 (RNN)**：适用于序列和时间序列数据
    - **长短期记忆网络 (LSTM)**：RNN的一种变体，解决长序列依赖问题
    - **自编码器 (Autoencoder)**：用于数据压缩和特征学习
    - **生成对抗网络 (GAN)**：用于生成新数据
    
    本课程专注于前馈网络，因为它是DQN算法使用的基础架构。
    """)
    
    # 基础练习
    st.markdown("## 基础练习")
    
    st.markdown("""
    1. 尝试不同的输入值，观察网络输出的变化。
    2. 切换不同的激活函数，比较结果差异。
    3. 思考：如果所有激活函数都是线性的，多层神经网络能否表达复杂函数？为什么？
    """)
    
    # 扩展资源
    st.sidebar.markdown("""
    ### 扩展资源
    
    - PyTorch官方教程：nn.Module和构建网络
    - 斯坦福CS231n：卷积神经网络
    - 《Deep Learning》(Goodfellow, Bengio, Courville)第6章
    """)
    
    # 小测验
    with st.expander("小测验：检验您的理解"):
        st.markdown("""
        1. 前馈神经网络的特点是什么？
           - A. 包含反馈连接
           - B. 信息只向前流动
           - C. 每层只有一个神经元
           - D. 不使用激活函数
        
        2. 在神经网络的矩阵表示中，权重矩阵W的形状通常是：
           - A. [输入维度 × 输出维度]
           - B. [输出维度 × 输入维度]
           - C. [输入维度 × 输入维度]
           - D. [输出维度 × 输出维度]
        
        3. 前向传播的主要步骤是：
           - A. 计算损失、更新权重
           - B. 线性变换、应用激活函数
           - C. 初始化权重、定义网络结构
           - D. 计算梯度、应用正则化
        
        **答案**: 1-B, 2-B, 3-B
        """)
    
    # 下一章预告
    st.markdown("""
    ## 下一章预告
    
    在下一章，我们将探讨如何**训练神经网络**，包括损失函数、梯度下降算法和反向传播过程。
    """) 