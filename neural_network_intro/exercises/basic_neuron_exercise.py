import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils.data_utils import generate_linearly_separable_data

def basic_neuron_exercise():
    """基础神经元练习页面"""
    st.title("练习：基础神经元")
    
    st.markdown("""
    ## 练习目标
    
    通过本练习，你将：
    1. 理解单个神经元的工作原理
    2. 实现不同的激活函数
    3. 观察单个神经元的决策边界
    
    完成这些练习将帮助你建立对神经网络基础构建块的直观理解。
    """)
    
    # 练习1：实现激活函数
    st.markdown("""
    ## 练习1：激活函数实现
    
    在下面的代码中，尝试实现以下激活函数：
    1. Sigmoid
    2. ReLU
    3. Tanh
    4. Leaky ReLU
    
    然后观察它们的曲线图形。
    """)
    
    # 显示代码模板
    st.code("""
    def sigmoid(x):
        # 请实现sigmoid函数
        return 1 / (1 + np.exp(-x))
    
    def relu(x):
        # 请实现ReLU函数
        return np.maximum(0, x)
    
    def tanh(x):
        # 请实现tanh函数
        return np.tanh(x)
    
    def leaky_relu(x, alpha=0.01):
        # 请实现Leaky ReLU函数
        return np.maximum(alpha * x, x)
    """)
    
    # 交互式激活函数可视化
    st.markdown("### 激活函数可视化")
    
    activation_choice = st.selectbox(
        "选择要可视化的激活函数",
        ["Sigmoid", "ReLU", "Tanh", "Leaky ReLU"]
    )
    
    x = np.linspace(-10, 10, 1000)
    
    if activation_choice == "Sigmoid":
        y = 1 / (1 + np.exp(-x))
        title = "Sigmoid激活函数"
    elif activation_choice == "ReLU":
        y = np.maximum(0, x)
        title = "ReLU激活函数"
    elif activation_choice == "Tanh":
        y = np.tanh(x)
        title = "Tanh激活函数"
    else:  # Leaky ReLU
        alpha = st.slider("Leaky ReLU的alpha参数", 0.0, 0.3, 0.01, 0.01)
        y = np.where(x > 0, x, alpha * x)
        title = f"Leaky ReLU激活函数 (alpha={alpha})"
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax.set_title(title)
    ax.set_xlabel("输入 x")
    ax.set_ylabel("输出 f(x)")
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    
    # 练习2：单个神经元的决策边界
    st.markdown("""
    ## 练习2：单个神经元的决策边界
    
    单个神经元可以用于二分类问题，特别是线性可分的数据。
    
    在这个练习中，你将观察一个单一神经元（感知器）如何形成决策边界。
    """)
    
    # 生成线性可分的数据
    n_samples = st.slider("样本数量", 10, 200, 100)
    noise = st.slider("噪声水平", 0.0, 1.0, 0.1)
    
    X, y = generate_linearly_separable_data(n_samples=n_samples, noise=noise)
    
    # 可视化数据
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X[y==0, 0], X[y==0, 1], c='blue', label='类别 0')
    ax.scatter(X[y==1, 0], X[y==1, 1], c='red', label='类别 1')
    ax.set_title("线性可分数据")
    ax.set_xlabel("特征 1")
    ax.set_ylabel("特征 2")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    
    # 感知器实现代码模板
    st.markdown("### 感知器实现")
    
    st.code("""
    class Perceptron:
        def __init__(self, learning_rate=0.01, n_iterations=100):
            self.learning_rate = learning_rate
            self.n_iterations = n_iterations
            self.weights = None
            self.bias = None
        
        def fit(self, X, y):
            n_samples, n_features = X.shape
            self.weights = np.zeros(n_features)
            self.bias = 0
            
            for _ in range(self.n_iterations):
                for idx, x_i in enumerate(X):
                    linear_output = np.dot(x_i, self.weights) + self.bias
                    y_predicted = 1 if linear_output >= 0 else 0
                    
                    # 更新规则
                    if y[idx] != y_predicted:
                        self.weights += self.learning_rate * (y[idx] - y_predicted) * x_i
                        self.bias += self.learning_rate * (y[idx] - y_predicted)
        
        def predict(self, X):
            linear_output = np.dot(X, self.weights) + self.bias
            return np.where(linear_output >= 0, 1, 0)
    """)
    
    # 展示训练感知器和决策边界
    st.markdown("### 交互式感知器训练演示")
    
    learning_rate = st.slider("学习率", 0.001, 0.1, 0.01, 0.001)
    n_iterations = st.slider("迭代次数", 10, 1000, 100)
    
    # 简单的感知器实现(为了演示)
    class SimplePerceptron:
        def __init__(self, learning_rate=0.01, n_iterations=100):
            self.learning_rate = learning_rate
            self.n_iterations = n_iterations
            self.weights = None
            self.bias = None
            self.training_history = []
        
        def fit(self, X, y):
            n_samples, n_features = X.shape
            self.weights = np.zeros(n_features)
            self.bias = 0
            
            for _ in range(self.n_iterations):
                for idx, x_i in enumerate(X):
                    linear_output = np.dot(x_i, self.weights) + self.bias
                    y_predicted = 1 if linear_output >= 0 else 0
                    
                    # 更新规则
                    if y[idx] != y_predicted:
                        self.weights += self.learning_rate * (y[idx] - y_predicted) * x_i
                        self.bias += self.learning_rate * (y[idx] - y_predicted)
                
                # 记录这一轮迭代后的权重和偏置
                self.training_history.append((self.weights.copy(), self.bias))
        
        def predict(self, X):
            linear_output = np.dot(X, self.weights) + self.bias
            return np.where(linear_output >= 0, 1, 0)
    
    # 训练感知器
    perceptron = SimplePerceptron(learning_rate=learning_rate, n_iterations=n_iterations)
    perceptron.fit(X, y)
    
    # 可视化决策边界
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制原始数据点
    ax.scatter(X[y==0, 0], X[y==0, 1], c='blue', label='类别 0')
    ax.scatter(X[y==1, 0], X[y==1, 1], c='red', label='类别 1')
    
    # 绘制决策边界
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    xx = np.linspace(x_min, x_max, 100)
    
    if perceptron.weights[1] != 0:
        # 计算决策边界线: w1*x1 + w2*x2 + b = 0 => x2 = (-w1*x1 - b)/w2
        yy = (-perceptron.weights[0] * xx - perceptron.bias) / perceptron.weights[1]
        ax.plot(xx, yy, 'k-', label='决策边界')
    
    ax.set_title("感知器的决策边界")
    ax.set_xlabel("特征 1")
    ax.set_ylabel("特征 2")
    ax.legend()
    ax.grid(alpha=0.3)
    
    st.pyplot(fig)
    
    # 感知器参数
    st.markdown(f"""
    ### 感知器训练结果
    
    **最终权重**：[{perceptron.weights[0]:.4f}, {perceptron.weights[1]:.4f}]
    
    **最终偏置**：{perceptron.bias:.4f}
    
    **决策边界方程**：{perceptron.weights[0]:.4f} * x₁ + {perceptron.weights[1]:.4f} * x₂ + {perceptron.bias:.4f} = 0
    """)
    
    # 练习3：超参数调整实验
    st.markdown("""
    ## 练习3：超参数调整
    
    尝试调整学习率和迭代次数，观察它们如何影响感知器的训练和决策边界。
    
    **思考问题**：
    1. 较高的学习率会带来什么影响？
    2. 增加迭代次数总是有益的吗？
    3. 当数据不是线性可分时，感知器会如何表现？
    """)
    
    # 思考题答案区
    with st.expander("查看思考题提示"):
        st.markdown("""
        **思考题提示**：
        
        1. **较高学习率的影响**：
           - 学习率过高可能导致在最优解附近震荡
           - 可能跳过全局最优解
           - 在极端情况下可能导致权重发散
        
        2. **增加迭代次数**：
           - 对于线性可分数据，一旦找到适当的决策边界，额外的迭代可能没有显著改进
           - 可能导致过拟合，特别是在有噪声的数据上
           - 计算成本随迭代次数线性增加
        
        3. **非线性可分数据**：
           - 感知器无法收敛到一个稳定的解
           - 权重可能持续变化但永远无法找到完美的决策边界
           - 这正是为什么我们需要更复杂的神经网络结构的原因之一
        """)
    
    # 进阶练习
    st.markdown("""
    ## 进阶练习
    
    作为进一步的挑战，尝试以下练习：
    
    1. 修改代码以实现带有sigmoid激活函数的神经元（而不是感知器的阶跃函数）
    2. 实现一个带有批量更新的感知器（而不是样本逐个更新）
    3. 创建一个非线性可分的数据集，并观察感知器在这种情况下的表现
    """)
    
    # 小结
    st.markdown("""
    ## 小结
    
    在本练习中，你：
    1. 实现了不同的激活函数并可视化了它们
    2. 理解了单个神经元（感知器）如何形成决策边界
    3. 探索了学习率和迭代次数等超参数的影响
    
    这些是理解更复杂神经网络必不可少的基础知识。
    """)

if __name__ == "__main__":
    basic_neuron_exercise() 