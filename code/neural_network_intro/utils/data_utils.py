import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def generate_linearly_separable_data(n_samples=100, noise=0.1):
    """生成线性可分的二分类数据集"""
    np.random.seed(42)
    
    # 生成两个类的中心点
    center1 = np.array([1, 1])
    center2 = np.array([-1, -1])
    
    # 为每个类生成点
    n_class1 = n_samples // 2
    n_class2 = n_samples - n_class1
    
    # 第一个类的数据点
    X1 = np.random.randn(n_class1, 2) * noise + center1
    # 第二个类的数据点
    X2 = np.random.randn(n_class2, 2) * noise + center2
    
    # 合并数据
    X = np.vstack([X1, X2])
    y = np.hstack([np.ones(n_class1), np.zeros(n_class2)])
    
    # 打乱数据顺序
    shuffle_idx = np.random.permutation(n_samples)
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    return X, y

def generate_data_for_dqn_demo(n_samples=1000, noise=0.1):
    """生成用于DQN演示的简单数据集"""
    # 创建一个简单的环境状态数据
    np.random.seed(42)
    
    # 生成状态特征
    states = np.random.rand(n_samples, 4) * 2 - 1  # 4维状态空间, 值范围[-1, 1]
    
    # 为每个状态计算最优动作的Q值 (简化模型)
    q_values = np.zeros((n_samples, 2))  # 假设有2个可能的动作
    
    # 简单规则: 如果状态的平均值为正, 动作0更好; 否则动作1更好
    state_means = states.mean(axis=1)
    
    # 添加一些非线性关系和噪声
    for i in range(n_samples):
        if states[i, 0] * states[i, 1] > 0:  # 非线性条件
            q_values[i, 0] = 0.8 + states[i, 0] * 0.2 + np.random.normal(0, noise)
            q_values[i, 1] = 0.2 + states[i, 1] * 0.1 + np.random.normal(0, noise)
        else:
            q_values[i, 0] = 0.3 + states[i, 2] * 0.1 + np.random.normal(0, noise)
            q_values[i, 1] = 0.7 + states[i, 3] * 0.2 + np.random.normal(0, noise)
    
    # 创建数据集
    X = states
    y = np.argmax(q_values, axis=1)  # 最优动作
    
    return X, y, q_values

def plot_classification_results(X, y, y_pred=None):
    """绘制分类结果的2D投影"""
    # 使用PCA对4维数据进行降维可视化
    from sklearn.decomposition import PCA
    
    # 降至2维
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制原始标签
    colors = ['#1f77b4', '#ff7f0e']
    for i, color in enumerate(colors):
        mask = (y == i)
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, 
                  label=f'类别 {i} (真实)', alpha=0.5, marker='o')
    
    # 如果有预测标签, 也绘制出来
    if y_pred is not None:
        for i, color in enumerate(['#1f77b4', '#ff7f0e']):
            mask = (y_pred == i)
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], edgecolors=color, 
                      facecolors='none', label=f'类别 {i} (预测)', alpha=0.7, marker='s')
    
    # 计算准确率
    if y_pred is not None:
        accuracy = np.mean(y == y_pred)
        ax.set_title(f'分类结果 (准确率: {accuracy:.2f})')
    else:
        ax.set_title('数据分布')
    
    ax.set_xlabel('主成分 1')
    ax.set_ylabel('主成分 2')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def generate_regression_data(n_samples=100, noise=0.3):
    """生成简单的回归数据集"""
    np.random.seed(42)
    
    # 生成输入特征
    X = np.random.rand(n_samples, 1) * 4 - 2  # 范围为 [-2, 2]
    
    # 生成目标值 (非线性函数加噪声)
    y = np.sin(X) + X**2 / 4 + np.random.normal(0, noise, size=X.shape)
    
    return X.flatten(), y.flatten()

def plot_regression_data(X, y, y_pred=None):
    """绘制回归数据和预测结果"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制原始数据点
    ax.scatter(X, y, color='blue', alpha=0.6, label='真实数据')
    
    if y_pred is not None:
        # 排序以便绘制平滑曲线
        idx = np.argsort(X.ravel())
        ax.plot(X[idx], y_pred[idx], 'r-', linewidth=2, label='模型预测')
        
        # 计算均方误差
        mse = np.mean((y - y_pred)**2)
        ax.set_title(f'回归结果 (MSE: {mse:.4f})')
    else:
        ax.set_title('回归数据')
    
    ax.set_xlabel('输入 X')
    ax.set_ylabel('输出 y')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def create_dqn_example_data():
    """创建DQN计算示例数据"""
    np.random.seed(42)
    
    # 创建示例状态、动作、奖励和下一状态
    states = [
        np.array([0.1, 0.2, 0.3, 0.4]),
        np.array([0.5, -0.3, 0.2, -0.1]),
        np.array([-0.2, 0.4, -0.5, 0.1]),
        np.array([0.3, 0.3, 0.3, 0.3])
    ]
    
    actions = [0, 1, 0, 1]
    rewards = [1.0, -0.5, 0.0, 2.0]
    
    # 创建下一状态
    next_states = [
        np.array([0.15, 0.25, 0.35, 0.45]),
        np.array([0.45, -0.25, 0.15, -0.05]),
        np.array([-0.15, 0.45, -0.45, 0.15]),
        np.array([0.35, 0.35, 0.35, 0.35])
    ]
    
    # 为每个状态生成Q值
    q_values = []
    for i in range(4):
        # 创建2个动作的Q值
        q = np.zeros(2)
        q[actions[i]] = rewards[i] + 0.5  # 当前动作的Q值稍高一些
        q[1 - actions[i]] = rewards[i] * 0.8  # 另一个动作的Q值稍低
        q_values.append(q)
    
    # 计算目标Q值
    gamma = 0.9
    target_q = []
    for i in range(4):
        # 复制当前Q值
        tq = q_values[i].copy()
        # 更新所选动作的Q值
        tq[actions[i]] = rewards[i] + gamma * np.max(q_values[i])
        target_q.append(tq)
    
    # 创建示例数据集
    return {
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'next_states': next_states,
        'q_values': q_values,
        'target_q': target_q
    }

def generate_synthetic_dataset(dataset_type='moons', n_samples=200, noise=0.2):
    """生成各种类型的合成数据集"""
    if dataset_type == 'moons':
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    elif dataset_type == 'circles':
        X, y = make_circles(n_samples=n_samples, noise=noise, random_state=42, factor=0.5)
    elif dataset_type == 'xor':
        # 创建XOR数据
        np.random.seed(42)
        X = np.random.rand(n_samples, 2) * 2 - 1
        y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(np.int32)
    else:
        raise ValueError(f"不支持的数据集类型: {dataset_type}")
    
    # 标准化特征
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test 