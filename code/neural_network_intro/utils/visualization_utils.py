import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import platform

def setup_chinese_font():
    """设置中文字体，针对不同操作系统"""
    system = platform.system()
    
    if system == 'Windows':
        font = 'SimHei'  # 黑体
    elif system == 'Darwin':  # macOS
        font = 'Arial Unicode MS'  # 或 'PingFang SC'
    elif system == 'Linux':
        font = 'WenQuanYi Micro Hei'  # 文泉驿微米黑
    else:
        font = 'DejaVu Sans'  # 默认字体
    
    # 设置matplotlib字体
    plt.rcParams['font.sans-serif'] = [font]
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def plot_activation_functions(z_value=None):
    """绘制常见激活函数图像，可选高亮特定z值处的点
    
    参数:
        z_value: 可选，要在激活函数图上高亮显示的点的x坐标
    
    返回:
        matplotlib图像对象
    """
    # 设置中文字体
    setup_chinese_font()
    
    x = np.linspace(-5, 5, 100)
    
    # 定义激活函数
    sigmoid = 1 / (1 + np.exp(-x))
    tanh = np.tanh(x)
    relu = np.maximum(0, x)
    linear = x
    
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    
    # Sigmoid
    ax[0, 0].plot(x, sigmoid)
    ax[0, 0].set_title('Sigmoid激活函数')
    ax[0, 0].grid(True)
    ax[0, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax[0, 0].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax[0, 0].set_xlabel('z')
    ax[0, 0].set_ylabel('σ(z)')
    
    # Tanh
    ax[0, 1].plot(x, tanh)
    ax[0, 1].set_title('Tanh激活函数')
    ax[0, 1].grid(True)
    ax[0, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax[0, 1].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax[0, 1].set_xlabel('z')
    ax[0, 1].set_ylabel('tanh(z)')
    
    # ReLU
    ax[1, 0].plot(x, relu)
    ax[1, 0].set_title('ReLU激活函数')
    ax[1, 0].grid(True)
    ax[1, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax[1, 0].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax[1, 0].set_xlabel('z')
    ax[1, 0].set_ylabel('max(0, z)')
    
    # Linear
    ax[1, 1].plot(x, linear)
    ax[1, 1].set_title('线性激活函数')
    ax[1, 1].grid(True)
    ax[1, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax[1, 1].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax[1, 1].set_xlabel('z')
    ax[1, 1].set_ylabel('z')
    
    # 如果提供了z值，在所有图中高亮该点
    if z_value is not None:
        # 确保z值在我们的x范围内
        if -5 <= z_value <= 5:
            # 计算各激活函数在z_value处的值
            sigmoid_value = 1 / (1 + np.exp(-z_value))
            tanh_value = np.tanh(z_value)
            relu_value = max(0, z_value)
            linear_value = z_value
            
            # 在每个图中高亮显示该点
            ax[0, 0].plot(z_value, sigmoid_value, 'ro')
            ax[0, 0].annotate(f'({z_value:.2f}, {sigmoid_value:.2f})', 
                          xy=(z_value, sigmoid_value), 
                          xytext=(z_value + 0.5, sigmoid_value + 0.1),
                          arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
            
            ax[0, 1].plot(z_value, tanh_value, 'ro')
            ax[0, 1].annotate(f'({z_value:.2f}, {tanh_value:.2f})', 
                          xy=(z_value, tanh_value), 
                          xytext=(z_value + 0.5, tanh_value + 0.1),
                          arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
            
            ax[1, 0].plot(z_value, relu_value, 'ro')
            ax[1, 0].annotate(f'({z_value:.2f}, {relu_value:.2f})', 
                          xy=(z_value, relu_value), 
                          xytext=(z_value + 0.5, relu_value + 0.1),
                          arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
            
            ax[1, 1].plot(z_value, linear_value, 'ro')
            ax[1, 1].annotate(f'({z_value:.2f}, {linear_value:.2f})', 
                          xy=(z_value, linear_value), 
                          xytext=(z_value + 0.5, linear_value + 0.1),
                          arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    plt.tight_layout()
    return fig 