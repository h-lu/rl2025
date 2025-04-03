import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import io
import base64
from matplotlib.colors import to_rgba
import networkx as nx

def plot_activation_functions():
    """绘制常见激活函数图像"""
    x = np.linspace(-5, 5, 100)
    
    # 定义激活函数
    sigmoid = 1 / (1 + np.exp(-x))
    tanh = np.tanh(x)
    relu = np.maximum(0, x)
    leaky_relu = np.where(x > 0, x, 0.1 * x)
    
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    
    # Sigmoid
    ax[0].plot(x, sigmoid)
    ax[0].set_title('Sigmoid')
    ax[0].grid(True)
    ax[0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax[0].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Tanh
    ax[1].plot(x, tanh)
    ax[1].set_title('Tanh')
    ax[1].grid(True)
    ax[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax[1].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # ReLU
    ax[2].plot(x, relu)
    ax[2].set_title('ReLU')
    ax[2].grid(True)
    ax[2].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax[2].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Leaky ReLU
    ax[3].plot(x, leaky_relu)
    ax[3].set_title('Leaky ReLU')
    ax[3].grid(True)
    ax[3].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax[3].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_neuron():
    """绘制单个神经元结构图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 输入
    inputs = [(1, 4), (1, 3), (1, 2)]
    weights = [0.5, -0.3, 0.8]
    
    # 神经元
    neuron_pos = (4, 3)
    
    # 绘制输入
    for i, (pos, label) in enumerate(zip(inputs, ['x₁', 'x₂', 'x₃'])):
        circle = plt.Circle(pos, 0.3, color='skyblue', ec='blue')
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], label, ha='center', va='center', fontsize=12)
        
        # 连接线
        ax.arrow(pos[0]+0.3, pos[1], neuron_pos[0]-pos[0]-0.6, neuron_pos[1]-pos[1], 
                head_width=0.1, head_length=0.1, fc='black', ec='black', 
                length_includes_head=True)
        
        # 权重标签
        mid_x = (pos[0] + neuron_pos[0]) / 2
        mid_y = (pos[1] + neuron_pos[1]) / 2
        ax.text(mid_x, mid_y, f'w{i+1}={weights[i]}', ha='center', va='center', 
                fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    
    # 绘制神经元
    neuron = plt.Circle(neuron_pos, 0.3, color='salmon', ec='red')
    ax.add_patch(neuron)
    ax.text(neuron_pos[0], neuron_pos[1], 'Σ', ha='center', va='center', fontsize=14)
    
    # 输出
    output_pos = (7, 3)
    ax.arrow(neuron_pos[0]+0.3, neuron_pos[1], output_pos[0]-neuron_pos[0]-0.6, 0, 
            head_width=0.1, head_length=0.1, fc='black', ec='black', 
            length_includes_head=True)
    ax.text((neuron_pos[0]+output_pos[0])/2, neuron_pos[1]+0.2, 'f', ha='center', 
            va='center', fontsize=12)
    
    output = plt.Circle(output_pos, 0.3, color='lightgreen', ec='green')
    ax.add_patch(output)
    ax.text(output_pos[0], output_pos[1], 'y', ha='center', va='center', fontsize=12)
    
    # 添加偏置
    bias_pos = (2.5, 1.5)
    bias = plt.Circle(bias_pos, 0.3, color='lightyellow', ec='orange')
    ax.add_patch(bias)
    ax.text(bias_pos[0], bias_pos[1], 'b=1', ha='center', va='center', fontsize=12)
    
    ax.arrow(bias_pos[0]+0.2, bias_pos[1]+0.2, neuron_pos[0]-bias_pos[0]-0.4, 
            neuron_pos[1]-bias_pos[1]-0.4, head_width=0.1, head_length=0.1, 
            fc='black', ec='black', length_includes_head=True)
    
    ax.set_xlim(0, 8)
    ax.set_ylim(1, 5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    return fig

def plot_network_layers(layer_sizes, title="前馈神经网络结构"):
    """绘制神经网络层次结构图"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    n_layers = len(layer_sizes)
    layer_positions = np.linspace(1, 9, n_layers)
    
    # 颜色定义
    node_colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3']
    
    for i, (layer_pos, layer_size) in enumerate(zip(layer_positions, layer_sizes)):
        layer_type = "输入层" if i == 0 else "输出层" if i == n_layers-1 else f"隐藏层 {i}"
        
        # 绘制每个神经元
        y_positions = np.linspace(0, 10, layer_size+2)[1:-1]
        
        for j, y in enumerate(y_positions):
            node = plt.Circle((layer_pos, y), 0.3, color=node_colors[min(i, len(node_colors)-1)], 
                             alpha=0.8, ec='black')
            ax.add_patch(node)
            
            if i == 0:
                ax.text(layer_pos-0.4, y, f'x{j+1}', ha='right', va='center')
            elif i == n_layers-1:
                ax.text(layer_pos+0.4, y, f'y{j+1}', ha='left', va='center')
        
        # 层标签
        ax.text(layer_pos, 11, layer_type, ha='center', va='center', fontsize=12, 
               bbox=dict(facecolor='white', alpha=0.7))
        
        # 连接到下一层
        if i < n_layers - 1:
            next_y_positions = np.linspace(0, 10, layer_sizes[i+1]+2)[1:-1]
            for y1 in y_positions:
                for y2 in next_y_positions:
                    ax.plot([layer_pos+0.3, layer_positions[i+1]-0.3], [y1, y2], 
                           'k-', alpha=0.1)
    
    ax.set_xlim(0, 10)
    ax.set_ylim(-1, 12)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.title(title, fontsize=14)
    plt.tight_layout()
    return fig

def plot_gradient_descent():
    """绘制梯度下降优化过程"""
    # 创建函数和网格
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + (Y-1)**2  # 简单的二次函数，最小值在(0,1)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制等高线
    contours = ax.contour(X, Y, Z, levels=np.logspace(-0.5, 3, 15), cmap='viridis')
    ax.clabel(contours, inline=True, fontsize=8)
    
    # 梯度下降路径点
    path_x = [-1.8, -1.3, -0.7, -0.3, -0.1, 0]
    path_y = [2.5, 2.0, 1.6, 1.3, 1.1, 1]
    
    # 绘制路径
    ax.plot(path_x, path_y, 'ro-', linewidth=2, markersize=8)
    
    # 标记起点和终点
    ax.plot(path_x[0], path_y[0], 'bo', markersize=10, label='起点')
    ax.plot(path_x[-1], path_y[-1], 'go', markersize=10, label='全局最小值')
    
    # 添加梯度箭头
    for i in range(len(path_x) - 1):
        dx = path_x[i+1] - path_x[i]
        dy = path_y[i+1] - path_y[i]
        ax.arrow(path_x[i], path_y[i], dx*0.8, dy*0.8, head_width=0.1, 
                head_length=0.1, fc='blue', ec='blue', length_includes_head=True)
    
    ax.set_xlabel('参数 1')
    ax.set_ylabel('参数 2')
    ax.set_title('梯度下降优化过程')
    ax.legend()
    
    plt.tight_layout()
    return fig

def plot_loss_curves():
    """绘制训练过程中的损失曲线"""
    epochs = np.arange(1, 101)
    
    # 生成训练集和验证集的损失曲线
    train_loss = 2 * np.exp(-0.03 * epochs) + 0.2 + np.random.normal(0, 0.05, len(epochs))
    val_loss = 2 * np.exp(-0.02 * epochs) + 0.3 + np.random.normal(0, 0.08, len(epochs))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(epochs, train_loss, 'b-', label='训练损失')
    ax.plot(epochs, val_loss, 'r-', label='验证损失')
    
    ax.axhline(y=0.3, color='g', linestyle='--', alpha=0.7, label='理想损失')
    
    # 标记过拟合的区域
    overfitting_start = 70
    ax.axvspan(overfitting_start, 100, alpha=0.2, color='red')
    ax.text(85, 1.0, '过拟合区域', ha='center', va='center', fontsize=12)
    
    ax.set_xlabel('训练轮次 (Epochs)')
    ax.set_ylabel('损失值')
    ax.set_title('神经网络训练过程中的损失曲线')
    ax.grid(True)
    ax.legend()
    
    plt.tight_layout()
    return fig

def nn_svg_to_html():
    """生成交互式神经网络结构SVG"""
    # 使用预定义的SVG图像作为字符串
    svg_code = '''
    <svg width="600" height="400" xmlns="http://www.w3.org/2000/svg">
      <!-- 输入层 -->
      <circle cx="100" cy="100" r="20" fill="#66c2a5" />
      <circle cx="100" cy="200" r="20" fill="#66c2a5" />
      <circle cx="100" cy="300" r="20" fill="#66c2a5" />
      
      <!-- 隐藏层1 -->
      <circle cx="250" cy="80" r="20" fill="#fc8d62" />
      <circle cx="250" cy="160" r="20" fill="#fc8d62" />
      <circle cx="250" cy="240" r="20" fill="#fc8d62" />
      <circle cx="250" cy="320" r="20" fill="#fc8d62" />
      
      <!-- 隐藏层2 -->
      <circle cx="400" cy="120" r="20" fill="#8da0cb" />
      <circle cx="400" cy="200" r="20" fill="#8da0cb" />
      <circle cx="400" cy="280" r="20" fill="#8da0cb" />
      
      <!-- 输出层 -->
      <circle cx="550" cy="150" r="20" fill="#e78ac3" />
      <circle cx="550" cy="250" r="20" fill="#e78ac3" />
      
      <!-- 连接线 - 输入到隐藏层1 -->
      <line x1="120" y1="100" x2="230" y2="80" stroke="black" stroke-opacity="0.2" />
      <line x1="120" y1="100" x2="230" y2="160" stroke="black" stroke-opacity="0.2" />
      <line x1="120" y1="100" x2="230" y2="240" stroke="black" stroke-opacity="0.2" />
      <line x1="120" y1="100" x2="230" y2="320" stroke="black" stroke-opacity="0.2" />
      
      <line x1="120" y1="200" x2="230" y2="80" stroke="black" stroke-opacity="0.2" />
      <line x1="120" y1="200" x2="230" y2="160" stroke="black" stroke-opacity="0.2" />
      <line x1="120" y1="200" x2="230" y2="240" stroke="black" stroke-opacity="0.2" />
      <line x1="120" y1="200" x2="230" y2="320" stroke="black" stroke-opacity="0.2" />
      
      <line x1="120" y1="300" x2="230" y2="80" stroke="black" stroke-opacity="0.2" />
      <line x1="120" y1="300" x2="230" y2="160" stroke="black" stroke-opacity="0.2" />
      <line x1="120" y1="300" x2="230" y2="240" stroke="black" stroke-opacity="0.2" />
      <line x1="120" y1="300" x2="230" y2="320" stroke="black" stroke-opacity="0.2" />
      
      <!-- 连接线 - 隐藏层1到隐藏层2 -->
      <line x1="270" y1="80" x2="380" y2="120" stroke="black" stroke-opacity="0.2" />
      <line x1="270" y1="80" x2="380" y2="200" stroke="black" stroke-opacity="0.2" />
      <line x1="270" y1="80" x2="380" y2="280" stroke="black" stroke-opacity="0.2" />
      
      <line x1="270" y1="160" x2="380" y2="120" stroke="black" stroke-opacity="0.2" />
      <line x1="270" y1="160" x2="380" y2="200" stroke="black" stroke-opacity="0.2" />
      <line x1="270" y1="160" x2="380" y2="280" stroke="black" stroke-opacity="0.2" />
      
      <line x1="270" y1="240" x2="380" y2="120" stroke="black" stroke-opacity="0.2" />
      <line x1="270" y1="240" x2="380" y2="200" stroke="black" stroke-opacity="0.2" />
      <line x1="270" y1="240" x2="380" y2="280" stroke="black" stroke-opacity="0.2" />
      
      <line x1="270" y1="320" x2="380" y2="120" stroke="black" stroke-opacity="0.2" />
      <line x1="270" y1="320" x2="380" y2="200" stroke="black" stroke-opacity="0.2" />
      <line x1="270" y1="320" x2="380" y2="280" stroke="black" stroke-opacity="0.2" />
      
      <!-- 连接线 - 隐藏层2到输出层 -->
      <line x1="420" y1="120" x2="530" y2="150" stroke="black" stroke-opacity="0.2" />
      <line x1="420" y1="120" x2="530" y2="250" stroke="black" stroke-opacity="0.2" />
      
      <line x1="420" y1="200" x2="530" y2="150" stroke="black" stroke-opacity="0.2" />
      <line x1="420" y1="200" x2="530" y2="250" stroke="black" stroke-opacity="0.2" />
      
      <line x1="420" y1="280" x2="530" y2="150" stroke="black" stroke-opacity="0.2" />
      <line x1="420" y1="280" x2="530" y2="250" stroke="black" stroke-opacity="0.2" />
      
      <!-- 层标签 -->
      <text x="100" y="30" font-family="Arial" font-size="16" text-anchor="middle">输入层</text>
      <text x="250" y="30" font-family="Arial" font-size="16" text-anchor="middle">隐藏层 1</text>
      <text x="400" y="30" font-family="Arial" font-size="16" text-anchor="middle">隐藏层 2</text>
      <text x="550" y="30" font-family="Arial" font-size="16" text-anchor="middle">输出层</text>
      
      <!-- 节点标签 -->
      <text x="70" y="100" font-family="Arial" font-size="12" text-anchor="middle">x₁</text>
      <text x="70" y="200" font-family="Arial" font-size="12" text-anchor="middle">x₂</text>
      <text x="70" y="300" font-family="Arial" font-size="12" text-anchor="middle">x₃</text>
      
      <text x="580" y="150" font-family="Arial" font-size="12" text-anchor="middle">y₁</text>
      <text x="580" y="250" font-family="Arial" font-size="12" text-anchor="middle">y₂</text>
    </svg>
    '''
    
    return svg_code

def fig_to_html(fig):
    """将matplotlib图形转换为HTML以便嵌入到Streamlit中"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    data = base64.b64encode(buf.getbuffer()).decode("utf8")
    return f'<img src="data:image/png;base64,{data}" style="width:100%"/>' 