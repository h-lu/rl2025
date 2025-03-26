import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import streamlit as st

def sigmoid(x):
    """Sigmoid激活函数"""
    return 1 / (1 + np.exp(-x))

def relu(x):
    """ReLU激活函数"""
    return np.maximum(0, x)

def forward_pass_np(x, weights, biases, activation_func='sigmoid'):
    """使用numpy实现简单的前向传播"""
    activations = []
    layer_inputs = []
    a = x
    activations.append(a)
    
    for i in range(len(weights)):
        # 线性变换
        # 确保维度匹配 - 如果a是一维的，将其转为二维以便于矩阵乘法
        if len(a.shape) == 1:
            a = a.reshape(1, -1)  # 转为(1, n)形状的行向量
            
        # 确保权重矩阵的维度正确 - 输入维度应该等于前一层的输出维度
        # 对于feedforward_networks.py中的例子，权重矩阵的形状是(output_dim, input_dim)
        # 所以需要转置以便能够进行正确的矩阵乘法: a(1,input_dim) @ weights.T(input_dim,output_dim)
        z = np.dot(a, weights[i].T) + biases[i]
        layer_inputs.append(z)
        
        # 应用激活函数
        if activation_func == 'sigmoid':
            a = sigmoid(z)
        elif activation_func == 'relu':
            a = relu(z)
        else:
            a = z  # 线性激活或输出层
        
        activations.append(a)
    
    return activations, layer_inputs

class SimpleNN(nn.Module):
    """简单的PyTorch神经网络模型"""
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu'):
        super(SimpleNN, self).__init__()
        
        self.layers = nn.ModuleList()
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
    
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation(layer(x))
        
        # 输出层不使用激活函数
        x = self.layers[-1](x)
        return x

def create_simple_classification_data(n_samples=100, noise=0.1):
    """创建一个简单的二分类数据集"""
    # 生成圆形分布的数据
    np.random.seed(42)
    
    # 第一类 - 内圆
    n1 = n_samples // 2
    r1 = 0.5 + noise * np.random.randn(n1)
    theta1 = 2 * np.pi * np.random.rand(n1)
    X1 = np.vstack([r1 * np.cos(theta1), r1 * np.sin(theta1)]).T
    
    # 第二类 - 外圆
    n2 = n_samples - n1
    r2 = 2.0 + noise * np.random.randn(n2)
    theta2 = 2 * np.pi * np.random.rand(n2)
    X2 = np.vstack([r2 * np.cos(theta2), r2 * np.sin(theta2)]).T
    
    # 合并数据
    X = np.vstack([X1, X2])
    y = np.hstack([np.zeros(n1), np.ones(n2)])
    
    # 随机打乱数据
    indices = np.random.permutation(n_samples)
    X, y = X[indices], y[indices]
    
    return X, y

def plot_classification_data(X, y):
    """绘制分类数据"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 绘制两个类别的散点图
    colors = ['#1f77b4', '#ff7f0e']
    for i, color in enumerate(colors):
        mask = (y == i)
        ax.scatter(X[mask, 0], X[mask, 1], c=color, label=f'类别 {i}', alpha=0.7)
    
    ax.set_xlabel('特征 1')
    ax.set_ylabel('特征 2')
    ax.set_title('二分类数据')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def create_model_and_train(X, y, hidden_layers=[4], epochs=100, learning_rate=0.01):
    """创建并训练简单的分类模型"""
    # 转换为PyTorch张量
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y.reshape(-1, 1))
    
    # 创建数据集和数据加载器
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # 创建模型
    input_size = X.shape[1]
    model = SimpleNN(input_size, hidden_layers, 1, activation='relu')
    
    # 定义损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 存储训练过程
    losses = []
    
    # 训练模型
    for epoch in range(epochs):
        epoch_loss = 0
        for inputs, targets in dataloader:
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        
        # 打印训练进度
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
    
    return model, losses

def plot_decision_boundary(X, y, model):
    """绘制决策边界"""
    # 创建网格
    h = 0.02  # 网格步长
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # 将网格点转换为PyTorch张量
    grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    
    # 预测
    with torch.no_grad():
        Z = torch.sigmoid(model(grid)).numpy().reshape(xx.shape)
    
    # 绘制决策边界和数据点
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制决策边界
    ax.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.3)
    
    # 绘制数据点
    for i, color in enumerate(['#1f77b4', '#ff7f0e']):
        mask = (y == i)
        ax.scatter(X[mask, 0], X[mask, 1], c=color, label=f'类别 {i}', 
                  edgecolors='k', alpha=0.7)
    
    ax.set_xlabel('特征 1')
    ax.set_ylabel('特征 2')
    ax.set_title('神经网络决策边界')
    ax.legend()
    
    return fig

def plot_training_loss(losses):
    """绘制训练损失曲线"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(losses) + 1)
    ax.plot(epochs, losses, 'b-')
    ax.set_xlabel('训练轮次')
    ax.set_ylabel('损失')
    ax.set_title('训练损失曲线')
    ax.grid(True)
    
    return fig 