import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from utils.model_utils import create_simple_classification_data, plot_classification_data, create_model_and_train, plot_decision_boundary, plot_training_loss
from utils.data_utils import generate_synthetic_dataset

def show_simple_classification():
    """显示简单分类案例页面"""
    st.title("实践案例：简单分类问题")
    
    st.markdown("""
    ## 神经网络实战：二分类问题
    
    在本节中，我们将通过一个简单的二分类问题，演示神经网络的训练和使用过程。这个实例将帮助我们巩固前面学习的概念。
    """)
    
    # 数据集选择
    st.markdown("### 选择数据集")
    
    dataset_type = st.selectbox(
        "选择数据集类型",
        ["环形数据", "月牙形数据", "XOR数据", "自定义圆形数据"]
    )
    
    # 数据集参数
    col1, col2 = st.columns(2)
    
    with col1:
        n_samples = st.slider("样本数量", 50, 500, 200, 50)
    
    with col2:
        noise = st.slider("噪声水平", 0.0, 0.5, 0.1, 0.05)
    
    # 生成数据
    if dataset_type == "环形数据":
        X_train, X_test, y_train, y_test = generate_synthetic_dataset('circles', n_samples=n_samples, noise=noise)
        title = "环形数据集"
    elif dataset_type == "月牙形数据":
        X_train, X_test, y_train, y_test = generate_synthetic_dataset('moons', n_samples=n_samples, noise=noise)
        title = "月牙形数据集"
    elif dataset_type == "XOR数据":
        X_train, X_test, y_train, y_test = generate_synthetic_dataset('xor', n_samples=n_samples, noise=noise)
        title = "XOR数据集"
    else:  # 自定义圆形数据
        X, y = create_simple_classification_data(n_samples=n_samples, noise=noise)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        title = "自定义圆形数据集"
    
    # 显示数据集
    st.markdown(f"### 数据可视化: {title}")
    
    # 绘制训练数据
    fig = plt.figure(figsize=(10, 4))
    
    ax1 = fig.add_subplot(121)
    for i, color in enumerate(['#1f77b4', '#ff7f0e']):
        mask = (y_train == i)
        ax1.scatter(X_train[mask, 0], X_train[mask, 1], c=color, 
                   label=f'类别 {i}', alpha=0.7)
    ax1.set_title("训练集")
    ax1.set_xlabel("特征 1")
    ax1.set_ylabel("特征 2")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(122)
    for i, color in enumerate(['#1f77b4', '#ff7f0e']):
        mask = (y_test == i)
        ax2.scatter(X_test[mask, 0], X_test[mask, 1], c=color, 
                   label=f'类别 {i}', alpha=0.7)
    ax2.set_title("测试集")
    ax2.set_xlabel("特征 1")
    ax2.set_ylabel("特征 2")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("""
    上图展示了我们的二分类数据。不同颜色代表不同类别，左图是训练集，右图是测试集。
    
    这些数据不是线性可分的，意味着简单的线性模型无法很好地区分两个类别。这正是神经网络可以发挥作用的地方。
    """)
    
    # 神经网络模型设计
    st.markdown("## 神经网络模型设计")
    
    st.markdown("""
    现在，我们将设计一个神经网络来解决这个分类问题。您可以调整网络的结构和训练参数。
    """)
    
    # 网络参数选择
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 网络结构")
        
        hidden_layer_sizes = []
        num_layers = st.slider("隐藏层数量", 1, 3, 1)
        
        for i in range(num_layers):
            layer_size = st.slider(f"隐藏层 {i+1} 神经元数量", 2, 32, 8, 2)
            hidden_layer_sizes.append(layer_size)
        
        activation = st.selectbox(
            "激活函数",
            ["ReLU", "Sigmoid", "Tanh"]
        )
    
    with col2:
        st.markdown("### 训练参数")
        
        learning_rate = st.select_slider(
            "学习率",
            options=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
            value=0.01
        )
        
        epochs = st.slider("训练轮次", 10, 500, 100, 10)
        
        batch_size = st.select_slider(
            "批量大小",
            options=[4, 8, 16, 32, 64],
            value=16
        )
        
        optimizer_name = st.selectbox(
            "优化器",
            ["Adam", "SGD", "RMSprop"]
        )
    
    # 模型架构可视化
    st.markdown("### 模型架构")
    
    # 创建输出字符串
    layer_sizes = [2] + hidden_layer_sizes + [1]
    architecture_str = f"Input(2) → "
    
    for i, size in enumerate(hidden_layer_sizes):
        architecture_str += f"Dense({size}) → {activation} → "
    
    architecture_str += "Dense(1) → Sigmoid"
    
    st.code(architecture_str)
    
    # PyTorch模型定义
    model_code = f"""
class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.layers = nn.ModuleList()
        
        # 检查是否有隐藏层
        if len(hidden_layer_sizes) > 0:
            # 输入层到第一个隐藏层
            self.layers.append(nn.Linear(2, hidden_layer_sizes[0]))
            
            # 隐藏层之间的连接
            for i in range(len(hidden_layer_sizes) - 1):
                self.layers.append(nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i+1]))
            
            # 最后一个隐藏层到输出层
            self.layers.append(nn.Linear(hidden_layer_sizes[-1], 1))
        else:
            # 如果没有隐藏层，直接从输入到输出
            self.layers.append(nn.Linear(2, 1))
        
        # 激活函数
        self.activation = nn.{'ReLU' if activation == 'ReLU' else 'Sigmoid' if activation == 'Sigmoid' else 'Tanh'}()
        self.sigmoid = nn.Sigmoid()  # 用于输出层
        
    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = self.activation(x)
        
        x = self.layers[-1](x)
        x = self.sigmoid(x)
        return x
"""
    
    with st.expander("查看PyTorch模型代码"):
        st.code(model_code, language="python")
    
    # 训练模型
    if st.button("训练模型"):
        with st.spinner("模型训练中..."):
            # 转换为PyTorch张量
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.FloatTensor(y_train.reshape(-1, 1))
            
            # 创建数据集和数据加载器
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            # 创建模型
            class BinaryClassifier(nn.Module):
                def __init__(self):
                    super(BinaryClassifier, self).__init__()
                    self.layers = nn.ModuleList()
                    
                    # 检查是否有隐藏层
                    if len(hidden_layer_sizes) > 0:
                        # 输入层到第一个隐藏层
                        self.layers.append(nn.Linear(2, hidden_layer_sizes[0]))
                        
                        # 隐藏层之间的连接
                        for i in range(len(hidden_layer_sizes) - 1):
                            self.layers.append(nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i+1]))
                        
                        # 最后一个隐藏层到输出层
                        self.layers.append(nn.Linear(hidden_layer_sizes[-1], 1))
                    else:
                        # 如果没有隐藏层，直接从输入到输出
                        self.layers.append(nn.Linear(2, 1))
                    
                    # 激活函数
                    if activation == "ReLU":
                        self.activation = nn.ReLU()
                    elif activation == "Sigmoid":
                        self.activation = nn.Sigmoid()
                    else:  # Tanh
                        self.activation = nn.Tanh()
                    
                    self.sigmoid = nn.Sigmoid()  # 用于输出层
                    
                def forward(self, x):
                    for i in range(len(self.layers) - 1):
                        x = self.layers[i](x)
                        x = self.activation(x)
                    
                    x = self.layers[-1](x)
                    x = self.sigmoid(x)
                    return x
            
            model = BinaryClassifier()
            
            # 定义损失函数和优化器
            criterion = nn.BCELoss()
            
            if optimizer_name == "Adam":
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            elif optimizer_name == "SGD":
                optimizer = optim.SGD(model.parameters(), lr=learning_rate)
            else:  # RMSprop
                optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
            
            # 训练历史
            train_losses = []
            
            # 训练模型
            for epoch in range(epochs):
                model.train()
                epoch_loss = 0
                
                for batch_X, batch_y in train_loader:
                    # 前向传播
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    # 反向传播和优化
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                # 记录平均损失
                avg_loss = epoch_loss / len(train_loader)
                train_losses.append(avg_loss)
                
                # 每10轮打印一次进度
                if (epoch + 1) % 10 == 0:
                    st.text(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
            
            # 绘制训练损失曲线
            st.markdown("### 训练损失曲线")
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(range(1, epochs + 1), train_losses)
            ax.set_xlabel("训练轮次")
            ax.set_ylabel("损失")
            ax.set_title("训练损失曲线")
            ax.grid(True)
            st.pyplot(fig)
            
            # 评估模型
            st.markdown("### 模型评估")
            
            # 转换测试数据
            X_test_tensor = torch.FloatTensor(X_test)
            
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test_tensor)
                predicted = (test_outputs.numpy() > 0.5).astype(np.int32).reshape(-1)
                
                # 计算准确率
                accuracy = np.mean(predicted == y_test)
                st.markdown(f"**测试集准确率**: {accuracy:.2%}")
            
            # 绘制决策边界
            st.markdown("### 决策边界可视化")
            
            # 创建网格
            h = 0.02
            x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
            y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            
            # 对网格点进行预测
            grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
            with torch.no_grad():
                Z = model(grid).numpy()
            Z = (Z > 0.5).astype(np.int32).reshape(xx.shape)
            
            # 绘制决策边界和数据点
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 训练集结果
            ax1.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.3)
            for i, color in enumerate(['#1f77b4', '#ff7f0e']):
                mask = (y_train == i)
                ax1.scatter(X_train[mask, 0], X_train[mask, 1], c=color, 
                          label=f'类别 {i}', edgecolors='k', alpha=0.7)
            ax1.set_xlim(xx.min(), xx.max())
            ax1.set_ylim(yy.min(), yy.max())
            ax1.set_title("训练集上的决策边界")
            ax1.set_xlabel("特征 1")
            ax1.set_ylabel("特征 2")
            ax1.legend()
            
            # 测试集结果
            ax2.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.3)
            for i, color in enumerate(['#1f77b4', '#ff7f0e']):
                mask_true = (y_test == i)
                mask_correct = (predicted == y_test) & mask_true
                mask_incorrect = (predicted != y_test) & mask_true
                
                # 正确分类的点
                ax2.scatter(X_test[mask_correct, 0], X_test[mask_correct, 1], 
                           c=color, marker='o', edgecolors='k', alpha=0.7)
                
                # 错误分类的点
                ax2.scatter(X_test[mask_incorrect, 0], X_test[mask_incorrect, 1], 
                           c=color, marker='X', edgecolors='r', alpha=0.7)
            
            ax2.set_xlim(xx.min(), xx.max())
            ax2.set_ylim(yy.min(), yy.max())
            ax2.set_title("测试集上的决策边界 (X: 分类错误)")
            ax2.set_xlabel("特征 1")
            ax2.set_ylabel("特征 2")
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # 显示网络参数
            st.markdown("### 网络参数")
            
            total_params = sum(p.numel() for p in model.parameters())
            st.markdown(f"**参数总数**: {total_params}")
            
            # 展示一些模型参数
            with st.expander("查看模型参数"):
                for name, param in model.named_parameters():
                    st.markdown(f"**{name}**")
                    st.code(str(param.data.numpy()))
    
    else:
        st.markdown("点击'训练模型'按钮开始训练过程。")
    
    # 调整参数对结果的影响
    st.markdown("""
    ## 参数调整对结果的影响
    
    通过调整神经网络的架构和训练参数，您可以观察到不同的效果：
    
    1. **隐藏层数量和大小**：
       - 更多或更大的隐藏层可以提高模型的表示能力
       - 但过于复杂的模型可能导致过拟合
       
    2. **激活函数**：
       - ReLU通常训练更快，适合深层网络
       - Sigmoid和Tanh在某些分类问题上可能有优势
       
    3. **学习率**：
       - 太大可能导致训练不稳定
       - 太小可能导致收敛太慢
       
    4. **批量大小**：
       - 较小的批量带来更多的参数更新和噪声
       - 较大的批量可能提供更准确但更少的梯度更新
       
    5. **优化器**：
       - Adam通常收敛更快，但可能过度拟合
       - SGD可能泛化性更好，但收敛较慢
       - RMSprop在非平稳目标上表现良好
    """)
    
    # 基础练习
    st.markdown("## 基础练习")
    
    st.markdown("""
    1. 尝试不同的网络结构和参数组合，观察对训练过程和结果的影响。
    2. 比较不同数据集上的表现差异，思考为什么某些数据集更难分类？
    3. 对于难分类的情况（如XOR问题），测试增加网络复杂度是否有帮助。
    """)
    
    # 扩展练习
    with st.expander("扩展练习"):
        st.markdown("""
        1. **实现正则化**：
           - 尝试添加L2正则化（权重衰减）并观察其对过拟合的影响
           - 实现Dropout并比较结果
           
        2. **学习率调度**：
           - 实现学习率衰减策略
           - 比较固定学习率和衰减学习率的效果
           
        3. **交叉验证**：
           - 使用K折交叉验证评估模型性能
           - 分析不同模型超参数的稳定性
           
        4. **可视化特征**：
           - 提取并可视化隐藏层学到的特征
           - 观察神经网络如何转换输入空间
        """)
    
    # 与DQN的联系
    st.markdown("""
    ## 与DQN的联系
    
    本节中的分类示例与DQN有着密切联系：
    
    1. **函数近似**：
       - 分类模型学习输入到类别概率的映射
       - DQN学习状态到Q值的映射
       
    2. **训练过程**：
       - 两者都使用梯度下降优化
       - 两者都可能面临过拟合和欠拟合问题
       
    3. **架构选择**：
       - 网络复杂度需要匹配问题复杂度
       - 激活函数和层设计的原则类似
       
    4. **超参数敏感性**：
       - 学习率、批量大小等参数在两种情况下都很重要
       - 需要通过实验找到最佳配置
    
    主要区别在于，DQN处理的是强化学习问题，训练目标不是固定的，而是随着训练过程不断变化。
    """)
    
    # 侧边栏资源
    st.sidebar.markdown("""
    ### 深入了解资源
    
    - PyTorch官方教程：[分类器训练](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
    - 交互式ML演示：[TensorFlow Playground](https://playground.tensorflow.org/)
    - 神经网络可视化：[CNN Explainer](https://poloclub.github.io/cnn-explainer/)
    """)
    
    # 下一章预告
    st.markdown("""
    ## 下一章预告
    
    在下一章，我们将探讨**神经网络与DQN的关系**，介绍如何将神经网络用于强化学习，构建深度Q网络。
    """) 