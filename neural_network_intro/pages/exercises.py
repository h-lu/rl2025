import streamlit as st
from exercises.basic_neuron_exercise import basic_neuron_exercise
from exercises.nn_structure_exercise import nn_structure_exercise
from exercises.forward_propagation_exercise import forward_propagation_exercise
from exercises.loss_function_exercise import loss_function_exercise
from exercises.nn_dqn_exercise import nn_dqn_exercise
from exercises.activation_functions_exercise import activation_functions_exercise
from exercises.optimizer_backprop_exercise import optimizer_backprop_exercise

def show_exercises():
    """显示练习页面"""
    st.title("神经网络与深度学习练习")
    
    st.markdown("""
    ## 欢迎来到实践练习环节！
    
    在本部分，你将通过一系列交互式练习巩固对神经网络和深度Q网络(DQN)的理解。
    
    请从左侧选择一个练习模块开始。
    """)
    
    # 创建侧边栏选择器
    exercise_option = st.sidebar.selectbox(
        "选择练习模块",
        ["概述", "基础神经元练习", "激活函数详解", "神经网络结构练习", "前向传播练习", 
         "损失函数练习", "优化器与反向传播练习", "神经网络与DQN练习"]
    )
    
    # 根据选择显示不同的练习
    if exercise_option == "概述":
        show_overview()
    elif exercise_option == "基础神经元练习":
        basic_neuron_exercise()
    elif exercise_option == "激活函数详解":
        activation_functions_exercise()
    elif exercise_option == "神经网络结构练习":
        nn_structure_exercise()
    elif exercise_option == "前向传播练习":
        forward_propagation_exercise()
    elif exercise_option == "损失函数练习":
        loss_function_exercise()
    elif exercise_option == "优化器与反向传播练习":
        optimizer_backprop_exercise()
    elif exercise_option == "神经网络与DQN练习":
        nn_dqn_exercise()

def show_overview():
    """显示练习概述"""
    st.markdown("""
    ## 练习模块概述
    
    本课程提供以下练习模块：
    
    ### 1. 基础神经元练习
    - 理解单个神经元的工作原理
    - 实现不同的激活函数
    - 观察单个神经元的决策边界
    
    ### 2. 激活函数详解
    - 深入理解各种激活函数的特性和行为
    - 实现并可视化常见激活函数及其梯度
    - 分析不同激活函数对神经网络训练的影响
    
    ### 3. 神经网络结构练习
    - 设计不同结构的神经网络
    - 理解网络连接和参数
    - 比较不同网络架构的特点
    
    ### 4. 前向传播练习
    - 实现神经网络的前向传播
    - 理解矩阵计算在神经网络中的应用
    - 观察不同层和激活函数的影响
    
    ### 5. 损失函数练习
    - 理解各种损失函数的特点
    - 观察损失函数的梯度
    - 学习如何选择合适的损失函数
    
    ### 6. 优化器与反向传播练习
    - 理解梯度下降和反向传播算法
    - 比较不同优化器的性能
    - 观察优化过程的收敛行为
    
    ### 7. 神经网络与DQN练习
    - 理解DQN中的神经网络结构
    - 学习Q值与动作选择的关系
    - 掌握DQN的训练稳定性技术
    
    选择左侧的一个模块开始你的练习吧！
    """)
    
    # 显示模块间的关系图
    st.markdown("### 模块之间的关系")
    
    # 使用Graphviz绘制关系图
    graph_code = """
    digraph {
        rankdir=LR;
        node [shape=box, style=filled, color=lightblue];
        
        neuron [label="基础神经元"];
        activation [label="激活函数详解"];
        structure [label="神经网络结构"];
        forward [label="前向传播"];
        loss [label="损失函数"];
        optimizer [label="优化器与反向传播"];
        dqn [label="神经网络与DQN"];
        
        neuron -> activation;
        neuron -> structure;
        activation -> structure;
        structure -> forward;
        forward -> loss;
        loss -> optimizer;
        {neuron, activation, structure, forward, loss, optimizer} -> dqn;
    }
    """
    
    try:
        # 尝试使用graphviz包
        from graphviz import Source
        st.graphviz_chart(graph_code)
    except:
        # 如果不可用，显示简单的图片或文本描述
        st.markdown("""
        **模块依赖关系**:
        
        - 基础神经元 → 激活函数详解 → 神经网络结构 → 前向传播 → 损失函数 → 优化器与反向传播
        - 所有模块 → 神经网络与DQN
        
        建议按照上述顺序完成练习，以便更好地理解概念。
        """)
    
    # 练习建议
    st.markdown("""
    ### 练习建议
    
    1. **循序渐进**: 建议按照模块顺序进行练习
    2. **动手实践**: 尝试修改代码和参数，观察结果变化
    3. **思考问题**: 每个练习都有思考题，试着深入思考这些问题
    4. **构建联系**: 思考各模块之间的关联，形成系统性知识
    
    祝你学习愉快！
    """)

if __name__ == "__main__":
    show_exercises() 