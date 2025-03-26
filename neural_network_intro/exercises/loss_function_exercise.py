import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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

def loss_function_exercise():
    """损失函数练习页面"""
    # 设置中文字体
    setup_chinese_font()
    
    st.title("练习：损失函数")
    
    st.markdown("""
    ## 练习目标
    
    通过本练习，你将：
    1. 理解各种常见损失函数的特点和使用场景
    2. 实现不同的损失函数及其梯度
    3. 可视化不同损失函数的行为
    4. 了解损失函数对模型训练的影响
    
    完成这些练习将帮助你理解损失函数在神经网络训练中的重要作用。
    """)
    
    # 练习1：常见损失函数
    st.markdown("""
    ## 练习1：常见损失函数
    
    神经网络训练中使用的几种常见损失函数：
    
    1. **均方误差 (MSE)**：回归问题的标准损失
    2. **二元交叉熵 (BCE)**：二分类问题的标准损失
    3. **分类交叉熵 (CCE)**：多分类问题的标准损失
    4. **Huber损失**：对异常值更稳健的回归损失
    """)
    
    # 损失函数代码实现
    st.markdown("""
    ### 损失函数实现
    
    以下是几种常见损失函数的实现代码：
    """)
    
    st.code("""
def mean_squared_error(y_true, y_pred):
    \"\"\"
    均方误差损失函数
    
    参数:
        y_true: 真实标签
        y_pred: 预测值
        
    返回:
        mse: 均方误差
    \"\"\"
    return np.mean((y_true - y_pred) ** 2)

def binary_crossentropy(y_true, y_pred):
    \"\"\"
    二元交叉熵损失函数
    
    参数:
        y_true: 真实标签 (0 或 1)
        y_pred: 预测概率 (0到1之间)
        
    返回:
        bce: 二元交叉熵损失
    \"\"\"
    # 防止log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def categorical_crossentropy(y_true, y_pred):
    \"\"\"
    分类交叉熵损失函数
    
    参数:
        y_true: one-hot编码的真实标签
        y_pred: 预测概率分布
        
    返回:
        cce: 分类交叉熵损失
    \"\"\"
    # 防止log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1.0)
    
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

def huber_loss(y_true, y_pred, delta=1.0):
    \"\"\"
    Huber损失函数
    
    参数:
        y_true: 真实标签
        y_pred: 预测值
        delta: 阈值参数
        
    返回:
        huber: Huber损失
    \"\"\"
    error = y_true - y_pred
    abs_error = np.abs(error)
    
    quadratic = np.minimum(abs_error, delta)
    linear = abs_error - quadratic
    
    return np.mean(0.5 * quadratic ** 2 + delta * linear)
""")
    
    # 练习2：损失函数可视化
    st.markdown("""
    ## 练习2：损失函数可视化
    
    通过可视化不同损失函数的行为，我们可以更好地理解它们的特性。
    """)
    
    # 选择要可视化的损失函数
    loss_function = st.selectbox(
        "选择要可视化的损失函数",
        ["均方误差 (MSE)", "二元交叉熵 (BCE)", "Huber损失"]
    )
    
    # 生成数据点
    y_true = 0.5
    y_preds = np.linspace(0, 1, 1000)
    
    # 计算损失
    if loss_function == "均方误差 (MSE)":
        losses = (y_true - y_preds) ** 2
        title = "均方误差 (MSE)"
        y_label = "MSE损失"
    elif loss_function == "二元交叉熵 (BCE)":
        epsilon = 1e-15
        y_preds_clipped = np.clip(y_preds, epsilon, 1 - epsilon)
        losses = -(y_true * np.log(y_preds_clipped) + (1 - y_true) * np.log(1 - y_preds_clipped))
        title = "二元交叉熵 (BCE)"
        y_label = "BCE损失"
    else:  # Huber损失
        delta = st.slider("Huber损失的delta参数", 0.1, 2.0, 1.0, 0.1)
        error = y_true - y_preds
        abs_error = np.abs(error)
        quadratic = np.minimum(abs_error, delta)
        linear = abs_error - quadratic
        losses = 0.5 * quadratic ** 2 + delta * linear
        title = f"Huber损失 (delta={delta})"
        y_label = "Huber损失"
    
    # 绘制损失函数曲线
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(y_preds, losses)
    
    # 标记真实值
    ax.axvline(x=y_true, color='r', linestyle='--', alpha=0.5, label=f'真实值 = {y_true}')
    
    # 标记最小值
    min_idx = np.argmin(losses)
    min_pred = y_preds[min_idx]
    min_loss = losses[min_idx]
    ax.plot(min_pred, min_loss, 'ro', label=f'最小损失点: ({min_pred:.2f}, {min_loss:.4f})')
    
    ax.set_title(title)
    ax.set_xlabel("预测值")
    ax.set_ylabel(y_label)
    ax.legend()
    ax.grid(alpha=0.3)
    
    st.pyplot(fig)
    
    # 关于最小值的说明
    st.markdown(f"""
    **观察**：对于{loss_function}，当预测值为{min_pred:.4f}时，损失达到最小值{min_loss:.6f}。
    
    {'对于均方误差和Huber损失，预测值等于真实值时损失最小。' if loss_function != '二元交叉熵 (BCE)' else '二元交叉熵在预测值趋近真实值时损失最小，但由于数值稳定性的约束，可能在图上看不到完全为零的损失。'}
    """)
    
    # 练习3：不同损失函数的比较
    st.markdown("""
    ## 练习3：不同损失函数的比较
    
    让我们比较不同损失函数对同一组数据的处理方式。
    """)
    
    # 设置比较场景
    st.markdown("### 异常值对损失函数的影响")
    
    # 滑动条控制异常值的幅度
    outlier_value = st.slider("异常值幅度", 1.0, 10.0, 5.0, 0.5)
    
    # 生成带有异常值的样本
    np.random.seed(42)
    n_samples = 20
    y_true_normal = np.random.normal(0, 1, n_samples-1)
    y_true_with_outlier = np.append(y_true_normal, outlier_value)
    
    # 计算不同的预测情况下的损失
    y_preds = np.linspace(-1, outlier_value+1, 100)
    
    mse_losses = []
    huber_losses = []
    
    for pred in y_preds:
        y_pred = np.full_like(y_true_with_outlier, pred)
        
        # MSE
        mse = np.mean((y_true_with_outlier - y_pred) ** 2)
        mse_losses.append(mse)
        
        # Huber
        delta = 1.0
        error = y_true_with_outlier - y_pred
        abs_error = np.abs(error)
        quadratic = np.minimum(abs_error, delta)
        linear = abs_error - quadratic
        huber = np.mean(0.5 * quadratic ** 2 + delta * linear)
        huber_losses.append(huber)
    
    # 绘制比较图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(y_preds, mse_losses, label='MSE损失')
    ax.plot(y_preds, huber_losses, label='Huber损失 (delta=1.0)')
    
    # 标记正常数据的均值
    normal_mean = np.mean(y_true_normal)
    ax.axvline(x=normal_mean, color='g', linestyle='--', alpha=0.5, 
               label=f'正常数据均值 = {normal_mean:.2f}')
    
    # 标记所有数据的均值
    all_mean = np.mean(y_true_with_outlier)
    ax.axvline(x=all_mean, color='r', linestyle='--', alpha=0.5, 
               label=f'含异常值的均值 = {all_mean:.2f}')
    
    # 标记最小损失点
    mse_min_idx = np.argmin(mse_losses)
    huber_min_idx = np.argmin(huber_losses)
    
    ax.plot(y_preds[mse_min_idx], mse_losses[mse_min_idx], 'ro', 
            label=f'MSE最小点: {y_preds[mse_min_idx]:.2f}')
    ax.plot(y_preds[huber_min_idx], huber_losses[huber_min_idx], 'go', 
            label=f'Huber最小点: {y_preds[huber_min_idx]:.2f}')
    
    ax.set_title("MSE vs Huber损失 (带异常值数据)")
    ax.set_xlabel("预测值")
    ax.set_ylabel("损失")
    ax.legend()
    ax.grid(alpha=0.3)
    
    st.pyplot(fig)
    
    # 分析比较结果
    st.markdown(f"""
    **观察结果**：
    
    1. **MSE损失**最小点在 {y_preds[mse_min_idx]:.2f}，这接近于所有数据的均值 {all_mean:.2f}
    2. **Huber损失**最小点在 {y_preds[huber_min_idx]:.2f}，这更接近于正常数据的均值 {normal_mean:.2f}
    
    这说明:
    - MSE对异常值更敏感，因为它平方化了误差
    - Huber损失通过线性处理大误差，降低了异常值的影响
    - 当数据中存在异常值时，Huber损失通常能提供更稳健的估计
    """)
    
    # 练习4：损失函数的梯度
    st.markdown("""
    ## 练习4：损失函数的梯度
    
    在神经网络训练中，我们需要计算损失函数相对于模型参数的梯度。
    以下是不同损失函数的梯度：
    """)
    
    st.code("""
def mse_gradient(y_true, y_pred):
    \"\"\"
    均方误差损失函数的梯度
    
    参数:
        y_true: 真实标签
        y_pred: 预测值
        
    返回:
        gradient: 相对于y_pred的梯度
    \"\"\"
    return 2 * (y_pred - y_true) / len(y_true)

def binary_crossentropy_gradient(y_true, y_pred):
    \"\"\"
    二元交叉熵损失函数的梯度
    
    参数:
        y_true: 真实标签 (0 或 1)
        y_pred: 预测概率 (0到1之间)
        
    返回:
        gradient: 相对于y_pred的梯度
    \"\"\"
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / len(y_true)

def huber_loss_gradient(y_true, y_pred, delta=1.0):
    \"\"\"
    Huber损失函数的梯度
    
    参数:
        y_true: 真实标签
        y_pred: 预测值
        delta: 阈值参数
        
    返回:
        gradient: 相对于y_pred的梯度
    \"\"\"
    error = y_pred - y_true
    abs_error = np.abs(error)
    
    # 对于|error| <= delta的情况，梯度是error
    # 对于|error| > delta的情况，梯度是delta * sign(error)
    gradient = np.where(abs_error <= delta, 
                      error, 
                      delta * np.sign(error))
    
    return gradient / len(y_true)
""")
    
    # 可视化损失函数的梯度
    st.markdown("### 梯度可视化")
    
    gradient_loss_function = st.selectbox(
        "选择要可视化梯度的损失函数",
        ["均方误差 (MSE)", "二元交叉熵 (BCE)", "Huber损失"],
        key="gradient_viz"
    )
    
    # 生成数据点
    y_true_for_grad = 0.5
    y_preds_for_grad = np.linspace(0.01, 0.99, 1000)
    
    # 计算损失和梯度
    if gradient_loss_function == "均方误差 (MSE)":
        losses = (y_true_for_grad - y_preds_for_grad) ** 2
        gradients = 2 * (y_preds_for_grad - y_true_for_grad)
        title = "均方误差 (MSE) 及其梯度"
    elif gradient_loss_function == "二元交叉熵 (BCE)":
        epsilon = 1e-15
        y_preds_clipped = np.clip(y_preds_for_grad, epsilon, 1 - epsilon)
        losses = -(y_true_for_grad * np.log(y_preds_clipped) + 
                  (1 - y_true_for_grad) * np.log(1 - y_preds_clipped))
        gradients = ((1 - y_true_for_grad) / (1 - y_preds_clipped) - 
                    y_true_for_grad / y_preds_clipped)
        title = "二元交叉熵 (BCE) 及其梯度"
    else:  # Huber损失
        delta = st.slider("Huber损失的delta参数", 0.1, 2.0, 1.0, 0.1, key="huber_grad_delta")
        error = y_true_for_grad - y_preds_for_grad
        abs_error = np.abs(error)
        quadratic = np.minimum(abs_error, delta)
        linear = abs_error - quadratic
        losses = 0.5 * quadratic ** 2 + delta * linear
        
        error_for_grad = y_preds_for_grad - y_true_for_grad
        abs_error_for_grad = np.abs(error_for_grad)
        gradients = np.where(abs_error_for_grad <= delta, 
                          error_for_grad, 
                          delta * np.sign(error_for_grad))
        
        title = f"Huber损失 (delta={delta}) 及其梯度"
    
    # 绘制损失函数和梯度
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # 绘制损失曲线
    ax1.plot(y_preds_for_grad, losses, 'b-')
    ax1.set_title(f"{gradient_loss_function} 损失曲线")
    ax1.set_xlabel("预测值")
    ax1.set_ylabel("损失值")
    ax1.axvline(x=y_true_for_grad, color='r', linestyle='--', alpha=0.5, 
               label=f'真实值 = {y_true_for_grad}')
    ax1.grid(alpha=0.3)
    ax1.legend()
    
    # 绘制梯度曲线
    ax2.plot(y_preds_for_grad, gradients, 'g-')
    ax2.set_title(f"{gradient_loss_function} 梯度曲线")
    ax2.set_xlabel("预测值")
    ax2.set_ylabel("梯度值")
    ax2.axvline(x=y_true_for_grad, color='r', linestyle='--', alpha=0.5, 
               label=f'真实值 = {y_true_for_grad}')
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.grid(alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # 总结不同损失函数的特点和使用场景
    st.markdown("""
    ## 不同损失函数的特点和使用场景
    
    | 损失函数 | 特点 | 适用场景 | 优点 | 缺点 |
    |---------|------|----------|------|------|
    | 均方误差 (MSE) | 计算简单，梯度平滑 | 回归问题 | 易于理解和实现 | 对异常值敏感 |
    | 二元交叉熵 (BCE) | 基于概率分布的差异 | 二分类问题 | 对于概率预测非常有效 | 需要预测值在[0,1]区间内 |
    | 分类交叉熵 (CCE) | 衡量两个概率分布的差异 | 多分类问题 | 在分类任务中表现优异 | 需要稳定的数值计算 |
    | Huber损失 | 结合MSE和MAE的优点 | 有异常值的回归问题 | 对异常值更稳健 | 有额外的超参数delta |
    | L1正则化 | 产生稀疏解 | 特征选择和稀疏模型 | 促进模型稀疏性 | 在零点梯度不连续 |
    | L2正则化 | 使权重均匀变小 | 防止过拟合 | 有解析解，计算高效 | 不产生稀疏解 |
    """)
    
    # 思考题
    st.markdown("""
    ## 思考题
    
    1. 为什么在二分类问题中使用二元交叉熵而不是MSE作为损失函数?
    
    2. 在什么情况下应当考虑使用Huber损失而不是MSE?
    
    3. 损失函数的梯度如何影响模型的训练过程?
    
    4. 为什么在深度学习中经常需要对损失函数进行数值稳定性处理?
    """)
    
    with st.expander("查看思考题提示"):
        st.markdown("""
        **思考题提示**：
        
        1. **二元交叉熵 vs MSE**:
           - 二元交叉熵在概率接近0或1时梯度更大，促进更快学习
           - MSE的梯度在极端情况下变小，可能导致训练缓慢
           - 二元交叉熵是从最大似然估计导出的，与分类问题的统计基础更相符
        
        2. **Huber损失的使用场景**:
           - 当数据中包含异常值/离群点时
           - 当你希望减少异常值对模型的影响时
           - 当你需要比MSE更稳健但又比MAE更平滑的损失函数时
        
        3. **梯度对训练的影响**:
           - 梯度大小决定参数更新的步长
           - 梯度方向指导优化的方向
           - 梯度消失和爆炸会导致训练问题
           - 不同损失函数产生不同的梯度景观，影响优化性能
        
        4. **数值稳定性处理**:
           - 防止计算极限情况如log(0)导致的数值错误
           - 避免过大的数值导致的浮点溢出
           - 减少数值计算中的舍入误差
           - 确保梯度计算精确，避免梯度消失或爆炸
        """)
    
    # 小结
    st.markdown("""
    ## 小结
    
    在本练习中，你：
    1. 了解了常见损失函数的特点和实现
    2. 可视化了不同损失函数的行为
    3. 比较了不同损失函数对异常值的处理
    4. 学习了如何计算和可视化损失函数的梯度
    
    选择适当的损失函数对于神经网络的训练至关重要，它应当与你的任务类型和数据特点相匹配。
    """)

if __name__ == "__main__":
    loss_function_exercise() 