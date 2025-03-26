import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils.visualization import plot_gradient_descent, plot_loss_curves

def show_training():
    """显示神经网络训练原理页面"""
    st.title("训练神经网络")
    
    st.markdown("""
    ## 神经网络的训练过程
    
    神经网络的训练目标是**调整权重和偏置**，使网络输出尽可能接近目标值。这个过程包括以下关键要素：
    
    1. 损失函数
    2. 梯度下降优化
    3. 反向传播算法
    """)
    
    # 损失函数部分
    st.markdown("""
    ## 损失函数
    
    损失函数（Loss Function）衡量神经网络预测值与真实值之间的差距，为网络提供优化方向。
    """)
    
    # 常见损失函数
    with st.expander("常见损失函数"):
        st.markdown("""
        ### 均方误差 (MSE)
        
        适用于回归问题。
        
        $$\\mathcal{L}_{MSE} = \\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2$$
        
        ### 交叉熵损失 (Cross-Entropy Loss)
        
        适用于分类问题。
        
        **二分类**：
        
        $$\\mathcal{L}_{BCE} = -\\frac{1}{n} \\sum_{i=1}^{n} [y_i \\log(\\hat{y}_i) + (1 - y_i) \\log(1 - \\hat{y}_i)]$$
        
        **多分类**：
        
        $$\\mathcal{L}_{CE} = -\\frac{1}{n} \\sum_{i=1}^{n} \\sum_{j=1}^{m} y_{ij} \\log(\\hat{y}_{ij})$$
        
        其中 $y_{ij}$ 是样本 $i$ 属于类别 $j$ 的真实概率（通常是0或1），$\\hat{y}_{ij}$ 是预测概率。
        
        ### Huber损失
        
        结合了MSE和平均绝对误差(MAE)的优点，对异常值更鲁棒。
        
        $$
        \\mathcal{L}_{Huber} = 
        \\begin{cases} 
            \\frac{1}{2}(y - \\hat{y})^2 & \\text{if } |y - \\hat{y}| \\leq \\delta \\\\ 
            \\delta(|y - \\hat{y}| - \\frac{\\delta}{2}) & \\text{otherwise} 
        \\end{cases}
        $$
        
        ### 为什么DQN通常使用MSE或Huber损失？
        
        在DQN中，我们试图预测Q值，这是一个连续值，因此使用回归损失函数：
        
        - MSE在Q值预测偏差不大时效果好
        - Huber损失在Q值波动较大时更稳定
        - 这两种损失函数都允许神经网络逐渐调整其预测，适合强化学习的渐进学习特性
        """)
    
    # 梯度下降部分
    st.markdown("""
    ## 梯度下降优化
    
    梯度下降（Gradient Descent）是最常用的神经网络优化算法，通过沿着损失函数的负梯度方向调整参数，逐步减小损失。
    
    ### 梯度下降基本原理
    
    参数更新规则：
    
    $$\\theta_{new} = \\theta_{old} - \\alpha \\nabla_\\theta \\mathcal{L}(\\theta)$$
    
    其中：
    - $\\theta$ 是模型参数（权重和偏置）
    - $\\alpha$ 是学习率
    - $\\nabla_\\theta \\mathcal{L}(\\theta)$ 是损失函数关于参数的梯度
    """)
    
    # 可视化梯度下降过程
    st.markdown("### 梯度下降过程可视化")
    
    # 展示梯度下降图
    gradient_descent_fig = plot_gradient_descent()
    st.pyplot(gradient_descent_fig)
    
    # 梯度下降变体
    with st.expander("梯度下降的变体"):
        st.markdown("""
        ### 批量梯度下降 (Batch Gradient Descent)
        
        使用**所有**训练样本计算梯度，然后更新参数。
        
        **优点**：梯度估计准确，收敛稳定
        **缺点**：计算成本高，内存消耗大
        
        ### 随机梯度下降 (Stochastic Gradient Descent, SGD)
        
        每次使用**单个**随机样本计算梯度并更新参数。
        
        **优点**：计算快速，可能跳出局部最小值
        **缺点**：梯度估计噪声大，收敛不稳定
        
        ### 小批量梯度下降 (Mini-batch Gradient Descent)
        
        使用**小批量**样本计算梯度并更新参数，是最常用的方法。
        
        **优点**：结合了前两种方法的优势，计算效率和稳定性的良好平衡
        **缺点**：需要合理选择批量大小
        
        ### 改进的优化算法
        
        - **动量法 (Momentum)**：添加动量项加速收敛
        - **AdaGrad**：自适应学习率，适用于稀疏数据
        - **RMSProp**：解决AdaGrad学习率递减过快的问题
        - **Adam**：结合动量和RMSProp的优点，最常用的优化器之一
        """)
    
    # 学习率的影响
    st.markdown("### 学习率的影响")
    
    # 创建学习率演示
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**学习率过小**")
        lr_small_fig, ax = plt.subplots(figsize=(4, 3))
        x = np.linspace(-2, 2, 100)
        y = x**2
        ax.plot(x, y)
        
        # 模拟小学习率的更新
        x_points = [-1.5, -1.3, -1.1, -0.9, -0.7, -0.5, -0.3, -0.1]
        y_points = [x_p**2 for x_p in x_points]
        
        ax.plot(x_points, y_points, 'ro-', markersize=6)
        ax.set_title("学习率过小")
        ax.set_xlabel("参数值")
        ax.set_ylabel("损失")
        st.pyplot(lr_small_fig)
        st.markdown("收敛太慢")
    
    with col2:
        st.markdown("**学习率适中**")
        lr_good_fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(x, y)
        
        # 模拟合适学习率的更新
        x_points = [-1.5, -0.9, -0.5, -0.1, 0.0]
        y_points = [x_p**2 for x_p in x_points]
        
        ax.plot(x_points, y_points, 'go-', markersize=6)
        ax.set_title("学习率适中")
        ax.set_xlabel("参数值")
        ax.set_ylabel("损失")
        st.pyplot(lr_good_fig)
        st.markdown("快速收敛到最小值")
    
    with col3:
        st.markdown("**学习率过大**")
        lr_large_fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(x, y)
        
        # 模拟大学习率的更新
        x_points = [-1.5, 1.0, -2.0, 2.5, -3.0]
        y_points = [x_p**2 for x_p in x_points]
        
        ax.plot(x_points, y_points, 'bo-', markersize=6)
        ax.set_title("学习率过大")
        ax.set_xlabel("参数值")
        ax.set_ylabel("损失")
        st.pyplot(lr_large_fig)
        st.markdown("不收敛，在最小值附近震荡或发散")
    
    # 反向传播部分
    st.markdown("""
    ## 反向传播算法
    
    反向传播（Backpropagation）是高效计算神经网络中所有参数梯度的算法，基于链式法则。
    
    ### 反向传播基本原理
    
    1. **前向传播**：计算每层的激活值和输出
    2. **计算输出层误差**：比较预测值与真实值
    3. **反向传播误差**：从输出层向输入层逐层计算每个参数的梯度
    4. **更新参数**：使用梯度下降更新权重和偏置
    """)
    
    # 链式法则
    with st.expander("链式法则与反向传播"):
        st.markdown("""
        ### 链式法则在反向传播中的应用
        
        链式法则是微积分中的基本原理，用于计算复合函数的导数：
        
        $$\\frac{d}{dx}f(g(x)) = \\frac{df}{dg} \\cdot \\frac{dg}{dx}$$
        
        在神经网络中，我们通过链式法则计算损失函数关于各层参数的梯度。
        
        例如，计算损失函数 $\\mathcal{L}$ 关于某层权重 $W^{[l]}$ 的梯度：
        
        $$\\frac{\\partial \\mathcal{L}}{\\partial W^{[l]}} = \\frac{\\partial \\mathcal{L}}{\\partial a^{[l]}} \\cdot \\frac{\\partial a^{[l]}}{\\partial z^{[l]}} \\cdot \\frac{\\partial z^{[l]}}{\\partial W^{[l]}}$$
        
        其中：
        - $a^{[l]}$ 是第 $l$ 层的激活值
        - $z^{[l]}$ 是第 $l$ 层的线性输出
        
        反向传播的高效之处在于它避免了重复计算，通过从后向前传递梯度，复用已计算的中间结果。
        """)
    
    # 训练过程可视化
    st.markdown("### 训练过程可视化")
    
    # 绘制损失曲线
    loss_curves_fig = plot_loss_curves()
    st.pyplot(loss_curves_fig)
    
    # 训练中的挑战
    st.markdown("""
    ## 训练神经网络的挑战
    
    ### 1. 过拟合与欠拟合
    
    - **过拟合**：模型在训练数据上表现很好，但泛化能力差
    - **欠拟合**：模型既不能拟合训练数据，也不能泛化
    
    ### 2. 梯度消失与梯度爆炸
    
    - **梯度消失**：梯度变得非常小，导致参数几乎不更新
    - **梯度爆炸**：梯度变得非常大，导致参数更新过度
    
    ### 3. 局部最小值与鞍点
    
    复杂的损失函数表面可能包含多个局部最小值和鞍点，使优化变得困难。
    """)
    
    # 解决训练挑战的技术
    with st.expander("解决训练挑战的技术"):
        st.markdown("""
        ### 解决过拟合
        
        - **正则化**：L1/L2正则化，添加惩罚项限制权重大小
        - **Dropout**：随机关闭一定比例的神经元
        - **早停**：在验证损失开始上升时停止训练
        - **数据增强**：通过变换增加训练数据多样性
        
        ### 解决梯度问题
        
        - **权重初始化**：使用特定的初始化方法（如He初始化、Xavier初始化）
        - **批标准化**：在每一层输出上应用标准化，加速训练
        - **梯度裁剪**：限制梯度的最大范数
        - **使用ReLU等缓解梯度消失的激活函数**
        
        ### 优化技术
        
        - **学习率调度**：逐渐减小学习率
        - **高级优化器**：Adam, RMSProp等
        - **批量归一化**：减少内部协变量偏移
        """)
    
    # 训练循环
    st.markdown("""
    ## 训练循环
    
    神经网络的典型训练循环包括：
    
    ```python
    # 训练循环
    for epoch in range(num_epochs):
        # 遍历小批量数据
        for batch_X, batch_y in data_loader:
            # 前向传播
            predictions = model(batch_X)
            
            # 计算损失
            loss = loss_function(predictions, batch_y)
            
            # 反向传播
            optimizer.zero_grad()  # 清除之前的梯度
            loss.backward()        # 计算当前梯度
            
            # 更新参数
            optimizer.step()
            
        # 每个epoch结束后评估
        if epoch % eval_interval == 0:
            validate_model()
    ```
    """)
    
    # 实际训练模型
    st.markdown("## 交互式训练演示")
    
    st.markdown("""
    以下是一个简单的交互式演示，您可以调整参数看它们如何影响训练过程。
    
    注意：此演示是使用预先生成的数据进行的简化模拟，实际训练过程会更复杂。
    """)
    
    # 模拟训练过程
    col1, col2 = st.columns(2)
    
    with col1:
        learning_rate = st.slider("学习率", 0.001, 0.5, 0.01, 0.001, format="%.3f")
        batch_size = st.slider("批量大小", 1, 64, 16, 1)
    
    with col2:
        epochs = st.slider("训练轮次", 10, 200, 100, 10)
        dropout_rate = st.slider("Dropout比例", 0.0, 0.5, 0.2, 0.1)
    
    # 模拟不同参数对训练过程的影响
    np.random.seed(42)
    
    # 根据参数生成模拟训练曲线
    epoch_range = np.arange(1, epochs + 1)
    
    # 学习率影响收敛速度和稳定性
    lr_factor = np.clip(1.0 / (learning_rate * 10), 0.5, 5.0)
    
    # 批量大小影响噪声程度
    noise_factor = np.clip(1.0 / (batch_size ** 0.5), 0.1, 1.0)
    
    # Dropout影响泛化能力（验证损失）
    dropout_factor = np.clip(1.0 + dropout_rate * 2, 1.0, 2.0)
    
    # 模拟训练曲线
    base_curve = 2 * np.exp(-0.01 * lr_factor * epoch_range) + 0.2
    train_loss = base_curve + np.random.normal(0, 0.05 * noise_factor, epochs)
    
    # 验证曲线 - 加入过拟合效果
    inflection_point = int(epochs * (0.3 + 0.4 * dropout_rate))  # Dropout延迟过拟合
    val_base = np.copy(base_curve)
    val_base[inflection_point:] += np.linspace(0, 0.5 / dropout_factor, epochs - inflection_point)
    val_loss = val_base + np.random.normal(0, 0.08 * noise_factor, epochs)
    
    # 绘制训练过程
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(epoch_range, train_loss, 'b-', label='训练损失')
    ax.plot(epoch_range, val_loss, 'r-', label='验证损失')
    
    # 标记最佳点
    best_epoch = np.argmin(val_loss)
    ax.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, label='最佳模型')
    
    # 标记过拟合区域
    if best_epoch < epochs - 1:
        ax.axvspan(best_epoch, epochs, alpha=0.2, color='red')
        ax.text((best_epoch + epochs) / 2, 0.5, '过拟合区域', ha='center', va='center', fontsize=12)
    
    ax.set_xlabel('训练轮次 (Epochs)')
    ax.set_ylabel('损失值')
    ax.set_title(f'模拟训练过程 (学习率={learning_rate}, 批量大小={batch_size}, Dropout={dropout_rate})')
    ax.grid(True)
    ax.legend()
    
    st.pyplot(fig)
    
    # 训练结果解释
    st.markdown(f"""
    ### 训练结果分析
    
    - **收敛速度**：损失下降的速率受学习率影响最大
    - **曲线波动**：波动程度受批量大小影响，批量越小波动越大
    - **过拟合时间**：Dropout率越高，过拟合出现得越晚
    - **最佳模型**：在第 {best_epoch} 轮达到最低验证损失
    
    在实际训练中，您通常会保存验证损失最低的模型，并使用早停技术避免过度训练。
    """)
    
    # DQN中的训练特点
    st.markdown("""
    ## DQN中的训练特点
    
    DQN训练与标准神经网络训练有一些关键区别：
    
    1. **目标是移动的**：训练目标是基于当前的Q值估计，会随着训练而变化
    2. **经验回放**：使用存储的经验样本进行批量训练，减少样本相关性
    3. **目标网络**：使用单独的网络生成训练目标，提高稳定性
    4. **时序差分学习**：使用贝尔曼方程和引导学习，而不是直接监督学习
    
    这些特点使DQN训练比传统神经网络更加复杂，但基本的梯度下降和反向传播原理仍然适用。
    """)
    
    # 基础练习
    st.markdown("## 基础练习")
    
    st.markdown("""
    1. 尝试调整不同的参数组合，观察训练曲线的变化。
    2. 思考为什么梯度下降需要正确的学习率？
    3. 反向传播算法如何应用链式法则？试着写出一个简单网络的梯度计算步骤。
    """)
    
    # 延伸阅读
    st.sidebar.markdown("""
    ### 延伸阅读
    
    - 反向传播算法详解：[3Blue1Brown神经网络系列](https://www.youtube.com/watch?v=Ilg3gGewQ5U)
    - 优化算法对比：[An overview of gradient descent optimization algorithms](https://ruder.io/optimizing-gradient-descent/)
    - 深度学习中的正则化：[Deep Learning Book, Chapter 7](https://www.deeplearningbook.org/)
    """)
    
    # 小测验
    with st.expander("小测验：检验您的理解"):
        st.markdown("""
        1. 梯度下降算法的基本思想是什么？
           - A. 沿着梯度方向调整参数
           - B. 沿着负梯度方向调整参数
           - C. 随机调整参数直到损失降低
           - D. 计算所有可能的参数组合并选择最优解
        
        2. 以下哪种损失函数通常用于回归问题？
           - A. 交叉熵损失
           - B. 均方误差 (MSE)
           - C. 折页损失 (Hinge loss)
           - D. KL散度
        
        3. 小批量梯度下降与批量梯度下降相比，主要优势是：
           - A. 总能找到全局最优解
           - B. 不需要计算梯度
           - C. 兼顾计算效率和收敛稳定性
           - D. 不需要设置学习率
           
        4. 哪种现象表明你的学习率可能设置得过大？
           - A. 损失函数缓慢稳定下降
           - B. 损失函数几乎不变
           - C. 损失函数剧烈波动或增加
           - D. 训练过程非常缓慢
        
        **答案**: 1-B, 2-B, 3-C, 4-C
        """)
        
    # 下一章预告
    st.markdown("""
    ## 下一章预告
    
    在下一章，我们将深入探讨**深度学习要点**，包括深层网络架构、优化技巧以及深度学习在各领域的应用。
    """) 