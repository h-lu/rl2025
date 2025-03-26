import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import networkx as nx

def nn_structure_exercise():
    """神经网络结构练习页面"""
    st.title("练习：神经网络结构")
    
    st.markdown("""
    ## 练习目标
    
    通过本练习，你将：
    1. 理解神经网络的结构组成
    2. 学习设计不同结构的神经网络
    3. 理解神经网络的层级连接方式
    
    完成这些练习将帮助你构建神经网络架构的直观理解。
    """)
    
    # 练习1：神经网络结构可视化
    st.markdown("""
    ## 练习1：神经网络结构可视化
    
    设计一个简单的神经网络可视化工具，探索不同网络结构。
    """)
    
    # 网络结构参数
    col1, col2 = st.columns(2)
    
    with col1:
        n_input = st.slider("输入层神经元数量", 1, 10, 3)
        n_hidden_layers = st.slider("隐藏层数量", 1, 5, 2)
    
    with col2:
        hidden_neurons = []
        for i in range(n_hidden_layers):
            hidden_neurons.append(st.slider(f"隐藏层 {i+1} 神经元数量", 1, 10, 4))
        n_output = st.slider("输出层神经元数量", 1, 10, 2)
    
    # 可视化神经网络结构
    st.markdown("### 网络结构可视化")
    
    # 创建层列表
    layer_sizes = [n_input] + hidden_neurons + [n_output]
    
    # 计算神经元总数
    total_neurons = sum(layer_sizes)
    
    # 创建网络图
    G = nx.DiGraph()
    
    # 添加节点
    node_positions = {}
    neuron_count = 0
    for layer_idx, layer_size in enumerate(layer_sizes):
        for neuron_idx in range(layer_size):
            node_id = f"L{layer_idx}_N{neuron_idx}"
            # 设置节点位置：x坐标基于层索引，y坐标基于在层中的位置
            x_pos = layer_idx
            y_pos = neuron_idx - (layer_size - 1) / 2  # 使每层居中
            node_positions[node_id] = (x_pos, y_pos)
            
            # 设置节点特性
            if layer_idx == 0:
                node_color = 'skyblue'
                node_label = f"x{neuron_idx+1}"
            elif layer_idx == len(layer_sizes) - 1:
                node_color = 'salmon'
                node_label = f"y{neuron_idx+1}"
            else:
                node_color = 'lightgreen'
                node_label = f"h{layer_idx}_{neuron_idx+1}"
            
            G.add_node(node_id, color=node_color, label=node_label)
            neuron_count += 1
    
    # 添加边 (全连接层之间的连接)
    for layer_idx in range(len(layer_sizes) - 1):
        for source_idx in range(layer_sizes[layer_idx]):
            for target_idx in range(layer_sizes[layer_idx + 1]):
                source_id = f"L{layer_idx}_N{source_idx}"
                target_id = f"L{layer_idx+1}_N{target_idx}"
                G.add_edge(source_id, target_id)
    
    # 绘制神经网络
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制神经元
    for node_id, (x, y) in node_positions.items():
        color = G.nodes[node_id]['color']
        label = G.nodes[node_id]['label']
        
        circle = plt.Circle((x, y), 0.2, color=color, alpha=0.8)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=9)
    
    # 绘制连接
    drawn_edges = set()
    for edge in G.edges():
        source_id, target_id = edge
        source_pos = node_positions[source_id]
        target_pos = node_positions[target_id]
        
        # 使用简单的线段连接
        ax.plot([source_pos[0], target_pos[0]], 
                [source_pos[1], target_pos[1]], 
                'gray', alpha=0.5, linewidth=0.5)
    
    # 设置图表
    ax.set_xlim(-0.5, len(layer_sizes) - 0.5)
    min_y = min([y for _, y in node_positions.values()]) - 0.5
    max_y = max([y for _, y in node_positions.values()]) + 0.5
    ax.set_ylim(min_y, max_y)
    
    # 添加层标签
    layer_names = ["输入层"] + [f"隐藏层 {i+1}" for i in range(n_hidden_layers)] + ["输出层"]
    for i, name in enumerate(layer_names):
        ax.text(i, max_y + 0.3, name, ha='center', va='center', fontsize=11)
    
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("神经网络结构")
    
    st.pyplot(fig)
    
    # 显示网络参数
    st.markdown(f"""
    ### 网络参数
    
    1. **总层数**: {len(layer_sizes)}
    2. **总神经元数**: {total_neurons}
    3. **总连接数**: {len(G.edges())}
    4. **各层神经元数**: {layer_sizes}
    """)
    
    # 练习2：网络连接和权重
    st.markdown("""
    ## 练习2：网络连接和权重
    
    理解神经网络中的连接和权重，以及如何使用矩阵表示。
    """)
    
    # 简单网络参数
    st.markdown("### 简单的全连接网络示例")
    
    col1, col2 = st.columns(2)
    
    with col1:
        simple_input = st.slider("简单网络输入层大小", 1, 3, 2)
        simple_hidden = st.slider("简单网络隐藏层大小", 1, 4, 3)
    
    with col2:
        simple_output = st.slider("简单网络输出层大小", 1, 3, 2)
        weights_range = st.slider("权重生成范围", 0.1, 2.0, 1.0)
    
    # 生成随机权重矩阵
    np.random.seed(42)
    W1 = np.random.uniform(-weights_range, weights_range, (simple_hidden, simple_input))
    b1 = np.random.uniform(-weights_range, weights_range, (simple_hidden, 1))
    W2 = np.random.uniform(-weights_range, weights_range, (simple_output, simple_hidden))
    b2 = np.random.uniform(-weights_range, weights_range, (simple_output, 1))
    
    # 显示权重矩阵
    st.markdown("### 权重矩阵")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 隐藏层权重 (W1)")
        st.write(W1)
        st.markdown("#### 隐藏层偏置 (b1)")
        st.write(b1)
    
    with col2:
        st.markdown("#### 输出层权重 (W2)")
        st.write(W2)
        st.markdown("#### 输出层偏置 (b2)")
        st.write(b2)
    
    # 练习3：参数计算
    st.markdown("""
    ## 练习3：参数计算
    
    计算神经网络中的参数数量，理解网络复杂度。
    """)
    
    # 可以根据上面定义的网络参数计算
    params_W1 = simple_input * simple_hidden
    params_b1 = simple_hidden
    params_W2 = simple_hidden * simple_output
    params_b2 = simple_output
    total_params = params_W1 + params_b1 + params_W2 + params_b2
    
    st.markdown(f"""
    ### 参数统计
    
    计算结果：
    
    1. **隐藏层权重参数数量**: {params_W1} (输入维度 * 隐藏层神经元数)
    2. **隐藏层偏置参数数量**: {params_b1} (隐藏层神经元数)
    3. **输出层权重参数数量**: {params_W2} (隐藏层神经元数 * 输出维度)
    4. **输出层偏置参数数量**: {params_b2} (输出维度)
    5. **总参数数量**: {total_params}
    """)
    
    # 参数计算公式
    st.markdown("""
    ### 参数计算通用公式
    
    对于一个具有 L 层的全连接神经网络：
    
    - 参数数量 = Σ[(前一层神经元数 + 1) * 当前层神经元数]
    
    其中，+1 来自于偏置项。
    """)
    
    # 练习4：自定义神经网络结构
    st.markdown("""
    ## 练习4：不同网络结构的比较
    
    比较不同网络结构的参数数量和表达能力。
    """)
    
    # 几种常见网络结构的对比
    network_structures = {
        "扁平网络": [30, 10],
        "深窄网络": [10, 10, 10, 10],
        "宽浅网络": [50, 50],
        "漏斗网络": [50, 30, 10]
    }
    
    # 计算每种结构的参数数量
    input_size = 100  # 假设输入大小为100
    output_size = 5   # 假设输出大小为5
    
    structure_params = {}
    
    for name, hidden_layers in network_structures.items():
        layers = [input_size] + hidden_layers + [output_size]
        params = 0
        
        for i in range(1, len(layers)):
            params += layers[i-1] * layers[i] + layers[i]  # 权重和偏置
        
        structure_params[name] = params
    
    # 可视化不同结构的参数数量
    fig, ax = plt.subplots(figsize=(10, 6))
    
    structures = list(structure_params.keys())
    param_counts = list(structure_params.values())
    
    bars = ax.bar(structures, param_counts, color=['skyblue', 'lightgreen', 'salmon', 'lightgray'])
    
    # 添加数字标签
    for bar, count in zip(bars, param_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 5,
                f'{count:,}', ha='center', va='bottom')
    
    ax.set_title('不同网络结构的参数数量比较')
    ax.set_ylabel('参数数量')
    ax.set_xlabel('网络结构')
    ax.grid(axis='y', alpha=0.3)
    
    st.pyplot(fig)
    
    # 结构特点分析
    st.markdown("""
    ### 结构特点分析
    
    1. **扁平网络**:
       - 简单、参数较少
       - 表达能力有限
       - 训练速度快
    
    2. **深窄网络**:
       - 层次多，但每层神经元少
       - 可以学习更复杂的特征层次
       - 容易出现梯度问题
    
    3. **宽浅网络**:
       - 层数少但每层神经元多
       - 参数数量大
       - 适合学习复杂的单层特征
    
    4. **漏斗网络**:
       - 逐层减少神经元数量
       - 类似于特征提取和降维
       - 常用于自编码器等结构
    """)
    
    # 思考练习
    st.markdown("""
    ## 思考练习
    
    1. 在设计神经网络时，我们应该优先考虑增加网络的宽度还是深度？为什么？
    
    2. 如何权衡神经网络的表达能力和计算复杂度？
    
    3. 不同的网络结构适合解决什么类型的问题？
    """)
    
    # 提示
    with st.expander("查看思考题提示"):
        st.markdown("""
        **思考题提示**：
        
        1. **宽度与深度的选择**:
           - 深度增加可以学习更抽象的特征层次和复杂函数
           - 宽度增加可以增强单层的表达能力
           - 深度学习的成功表明，在许多情况下，深度比宽度更重要
        
        2. **表达能力与复杂度权衡**:
           - 更大的网络通常有更强的表达能力，但训练需要更多资源
           - 考虑使用技术如剪枝、量化、知识蒸馏等减少复杂度
           - 从小网络开始，逐步增加复杂度直到性能不再显著提升
        
        3. **不同结构的适用问题**:
           - CNN适合图像和空间数据
           - RNN/LSTM适合序列和时间序列数据
           - 宽浅网络可能适合特征已经高度结构化的数据
           - 深度网络适合需要多层抽象的复杂问题
        """)
    
    # 小结
    st.markdown("""
    ## 小结
    
    在本练习中，你：
    1. 探索了神经网络的结构设计
    2. 理解了网络连接、权重和偏置
    3. 学习了如何计算网络参数
    4. 比较了不同网络结构的特点
    
    这些知识将帮助你为不同的问题设计适当的神经网络架构。
    """)

if __name__ == "__main__":
    nn_structure_exercise() 