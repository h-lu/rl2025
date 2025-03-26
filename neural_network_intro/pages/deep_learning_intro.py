import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

def show_deep_learning_intro():
    """显示深度学习入门页面"""
    st.title("深度学习要点")
    
    st.markdown("""
    ## 什么是深度学习？
    
    深度学习是机器学习的一个子领域，专注于使用具有多个隐藏层的神经网络（深度神经网络）解决复杂问题。
    
    关键特点：
    - 多层次的特征学习
    - 端到端的学习过程
    - 大规模数据和计算的应用
    """)
    
    # 深度 vs 浅层网络
    st.markdown("""
    ## 深度网络 vs 浅层网络
    
    ### 为什么需要深度网络？
    
    深度网络相比浅层网络具有以下优势：
    
    1. **表示能力更强**：可以学习更复杂的特征和模式
    2. **层次化特征提取**：低层学习简单特征，高层组合形成复杂特征
    3. **参数效率**：使用较少的参数表达复杂函数
    """)
    
    # 可视化层次特征学习
    st.markdown("### 层次化特征学习")
    
    # 创建简单的图示
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    # 第一层 - 简单特征
    ax[0].set_title('第一层：简单特征')
    ax[0].imshow(np.random.rand(5, 5) > 0.7, cmap='gray')
    ax[0].set_xlabel('例如：边缘、角点')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    
    # 第二层 - 中级特征
    ax[1].set_title('中间层：组合特征')
    pattern = np.zeros((5, 5))
    pattern[1:4, 1:4] = 1
    ax[1].imshow(pattern, cmap='gray')
    ax[1].set_xlabel('例如：形状、纹理')
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    
    # 第三层 - 高级特征
    ax[2].set_title('高层：语义特征')
    complex_pattern = np.zeros((5, 5))
    complex_pattern[0:3, 0:3] = 1
    complex_pattern[2:5, 2:5] = 0.5
    ax[2].imshow(complex_pattern, cmap='viridis')
    ax[2].set_xlabel('例如：物体部分、概念')
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    
    st.pyplot(fig)
    
    st.markdown("""
    随着网络深度增加，神经网络能够学习到越来越抽象的表示。例如，在图像识别中：
    - 第一层可能学习边缘和角点
    - 中间层组合成简单形状和纹理
    - 深层学习复杂的物体部分和概念
    
    这种层次化特征学习使深度网络在处理复杂输入（如图像、音频和自然语言）时特别有效。
    """)
    
    # 深度学习的重要组件
    st.markdown("""
    ## 深度学习的重要组件
    
    现代深度学习系统通常包含以下关键组件：
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 架构组件
        
        - **卷积层**：捕获局部空间模式
        - **池化层**：降维和位置不变性
        - **全连接层**：组合特征进行最终预测
        - **归一化层**：稳定训练过程
        - **跳跃连接**：缓解梯度问题
        """)
    
    with col2:
        st.markdown("""
        ### 训练技术
        
        - **批归一化**：加速训练，提高稳定性
        - **残差连接**：训练非常深的网络
        - **注意力机制**：对重要特征赋予更高权重
        - **迁移学习**：重用预训练模型
        - **正则化方法**：防止过拟合
        """)
    
    # 常见网络架构示例
    st.markdown("""
    ## 常见深度学习架构
    
    现代深度学习有多种成功的网络架构，每种适合不同类型的问题：
    """)
    
    # 使用表格展示不同架构
    architectures = {
        "前馈神经网络 (FNN)": ["一般分类和回归", "简单，适用于表格数据", "DQN的基础架构"],
        "卷积神经网络 (CNN)": ["图像处理，计算机视觉", "利用空间局部性", "可用于处理图像状态的DQN"],
        "循环神经网络 (RNN)": ["序列数据，时间序列", "捕获时间依赖关系", "可用于DRQN处理部分可观测状态"],
        "长短期记忆网络 (LSTM)": ["长序列，语言建模", "解决长期依赖问题", "增强RNN的记忆能力"],
        "变换器 (Transformer)": ["自然语言处理，序列建模", "并行处理，注意力机制", "最新的序列建模技术"]
    }
    
    # 创建一个漂亮的表格
    col1, col2, col3 = st.columns(3)
    col1.markdown("**架构**")
    col2.markdown("**主要应用**")
    col3.markdown("**特点**")
    
    for arch, (apps, features, relevance) in architectures.items():
        col1.markdown(f"**{arch}**")
        col2.markdown(apps)
        col3.markdown(features)
        
        # 为每个架构添加一个分隔线
        col1.markdown("---")
        col2.markdown("---")
        col3.markdown("---")
    
    # DQN中的深度学习应用
    st.markdown("""
    ## 深度学习在DQN中的应用
    
    深度Q网络 (DQN) 是将深度学习应用于强化学习的典型例子：
    
    ### DQN的核心思想
    
    DQN 使用深度神经网络来近似Q函数：$Q(s, a; θ)$，其中 $θ$ 是网络参数。
    
    ### 深度学习在DQN中的作用
    
    1. **函数近似**：神经网络作为Q值的函数近似器
    2. **特征提取**：从原始状态中自动提取有用特征
    3. **泛化能力**：在未见过的状态上做出合理预测
    4. **处理高维输入**：能够处理像素级的游戏画面等高维输入
    """)
    
    # 示例 DQN 架构
    st.markdown("### 典型的DQN神经网络架构")
    
    # 创建一个DQN网络架构图
    def create_dqn_architecture_fig():
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 位置设置
        input_pos = 1
        conv_pos = [2.5, 4, 5.5]
        fc_pos = [7, 8.5, 10]
        output_pos = 11.5
        
        # 高度设置
        layer_heights = {
            'input': 2.0,
            'conv1': 1.6,
            'conv2': 1.2,
            'conv3': 0.8,
            'fc1': 0.6,
            'fc2': 0.4,
            'fc3': 0.3,  # 添加fc3以解决KeyError
            'output': 0.2
        }
        
        # 绘制输入层
        rect_input = plt.Rectangle((input_pos, 0), 1, layer_heights['input'], 
                                 color='skyblue', alpha=0.8)
        ax.add_patch(rect_input)
        ax.text(input_pos + 0.5, layer_heights['input'] + 0.2, "输入层\n(状态)", 
               ha='center', va='center')
        
        # 绘制卷积层
        for i, pos in enumerate(conv_pos):
            height = layer_heights[f'conv{i+1}']
            rect_conv = plt.Rectangle((pos, 0), 1, height, color='salmon', alpha=0.8)
            ax.add_patch(rect_conv)
            ax.text(pos + 0.5, height + 0.2, f"卷积层 {i+1}", ha='center', va='center')
        
        # 绘制全连接层
        for i, pos in enumerate(fc_pos[:2]):
            height = layer_heights[f'fc{i+1}']
            rect_fc = plt.Rectangle((pos, 0), 1, height, color='lightgreen', alpha=0.8)
            ax.add_patch(rect_fc)
            ax.text(pos + 0.5, height + 0.2, f"全连接层 {i+1}", ha='center', va='center')
        
        # 绘制输出层
        rect_output = plt.Rectangle((output_pos, 0), 1, layer_heights['output'], 
                                  color='gold', alpha=0.8)
        ax.add_patch(rect_output)
        ax.text(output_pos + 0.5, layer_heights['output'] + 0.2, "输出层\n(Q值)", 
               ha='center', va='center')
        
        # 添加连接箭头
        ax.arrow(input_pos + 1, layer_heights['input'] / 2, conv_pos[0] - input_pos - 1.1, 0, 
                head_width=0.1, head_length=0.1, fc='gray', ec='gray')
        
        for i in range(len(conv_pos) - 1):
            ax.arrow(conv_pos[i] + 1, layer_heights[f'conv{i+1}'] / 2, 
                    conv_pos[i+1] - conv_pos[i] - 1.1, 0, 
                    head_width=0.1, head_length=0.1, fc='gray', ec='gray')
        
        ax.arrow(conv_pos[-1] + 1, layer_heights[f'conv{len(conv_pos)}'] / 2, 
                fc_pos[0] - conv_pos[-1] - 1.1, 0, 
                head_width=0.1, head_length=0.1, fc='gray', ec='gray')
        
        for i in range(len(fc_pos) - 1):
            ax.arrow(fc_pos[i] + 1, layer_heights[f'fc{i+1}'] / 2, 
                    fc_pos[i+1] - fc_pos[i] - 1.1, 0, 
                    head_width=0.1, head_length=0.1, fc='gray', ec='gray')
        
        # 这里是问题所在, 我们需要正确处理fc_pos的索引
        last_fc_index = min(len(fc_pos), 3)  # 确保不会超出fc_pos的范围
        last_fc_layer = f'fc{last_fc_index}'
        if last_fc_layer not in layer_heights:
            last_fc_layer = f'fc{len(fc_pos)}'  # 回退到可用的最后一个fc层
            
        ax.arrow(fc_pos[-1] + 1, layer_heights[last_fc_layer] / 2, 
                output_pos - fc_pos[-1] - 1.1, 0, 
                head_width=0.1, head_length=0.1, fc='gray', ec='gray')
        
        # 设置图表
        ax.set_xlim(0, 13)
        ax.set_ylim(-0.5, 3)
        ax.set_title("典型的DQN网络架构", fontsize=14)
        ax.axis('off')
        
        return fig
    
    # 显示DQN架构图
    dqn_arch_fig = create_dqn_architecture_fig()
    st.pyplot(dqn_arch_fig)
    
    st.markdown("""
    上图展示了一个处理图像输入的典型DQN架构：
    
    1. **输入层**：接收游戏状态（如屏幕像素）
    2. **卷积层**：提取空间特征（如物体、边缘）
    3. **全连接层**：整合特征并推理
    4. **输出层**：每个可能动作的Q值
    
    注意：对于非图像输入的DQN（如处理传感器数据或低维状态），可能只使用全连接层而不需要卷积层。
    """)
    
    # 深度学习的挑战与最佳实践
    st.markdown("""
    ## 深度学习的挑战与最佳实践
    
    尽管深度学习非常强大，但它也面临一些重要挑战：
    """)
    
    challenges = {
        "需要大量数据": ["可能需要大量标记数据", "数据增强、迁移学习、合成数据生成"],
        "计算密集型": ["训练深层网络需要高计算资源", "模型压缩、知识蒸馏、高效架构"],
        "黑盒特性": ["难以解释决策过程", "可解释AI技术、注意力可视化"],
        "超参数调优": ["对超参数敏感", "自动超参数优化、交叉验证"],
        "容易过拟合": ["大模型容易记住而不是泛化", "正则化、数据扩增、提前停止"]
    }
    
    # 使用可折叠部分显示挑战和解决方案
    for challenge, (desc, solution) in challenges.items():
        with st.expander(f"**{challenge}**"):
            st.markdown(f"""
            **问题**：{desc}
            
            **解决方案**：{solution}
            """)
    
    # 深度学习最佳实践
    st.markdown("""
    ### 深度学习最佳实践
    
    在实际应用深度学习时，以下最佳实践可以帮助提高性能：
    
    1. **从简单开始**：先构建简单模型作为基准，再逐步增加复杂度
    2. **仔细预处理数据**：标准化、处理缺失值、处理类别不平衡
    3. **使用迁移学习**：在相关任务上预训练的模型通常表现更好
    4. **系统地调整超参数**：使用网格搜索或贝叶斯优化
    5. **监控验证性能**：防止过拟合，适时停止训练
    """)
    
    # 在实现DQN时的考虑
    st.markdown("""
    ## 实现DQN时的深度学习考虑
    
    当将深度学习应用于DQN时，需要特别注意以下方面：
    
    ### 网络架构设计
    
    - **输入处理**：根据状态表示选择适当的网络结构（CNN用于图像，MLP用于向量）
    - **输出层设计**：输出层神经元数量应等于可能的动作数量
    - **网络规模**：不要过于复杂，通常2-3个隐藏层就足够
    
    ### 稳定训练
    
    - **经验回放**：打破数据相关性，提高样本利用率
    - **目标网络**：使用单独的网络生成训练目标，减少不稳定性
    - **批归一化**：帮助加速训练并提高稳定性
    - **梯度裁剪**：防止梯度爆炸影响训练
    
    ### 泛化性能
    
    - **Dropout**：用于改善泛化能力，特别是在小样本情况下
    - **ε-贪婪策略**：平衡探索和利用，避免过早收敛到次优策略
    - **奖励缩放**：将奖励限制在合理范围内，使训练更稳定
    """)
    
    # PyTorch DQN示例代码
    st.markdown("### PyTorch中的DQN网络实现示例")
    
    with st.expander("查看PyTorch DQN实现"):
        st.code("""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        
        # 对于图像输入，使用卷积层
        if len(input_shape) == 3:  # [C, H, W] 格式
            self.features = nn.Sequential(
                nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU()
            )
            
            # 计算卷积层后的特征图大小
            self.feature_size = self._get_conv_output(input_shape)
            
            # 全连接层
            self.fc = nn.Sequential(
                nn.Linear(self.feature_size, 512),
                nn.ReLU(),
                nn.Linear(512, num_actions)
            )
        else:  # 对于向量输入，只使用全连接层
            self.features = None
            self.fc = nn.Sequential(
                nn.Linear(input_shape[0], 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, num_actions)
            )
    
    def _get_conv_output(self, shape):
        # 计算卷积层输出大小的辅助函数
        o = self.features(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        if self.features:
            x = self.features(x)
            x = x.view(x.size(0), -1)
        return self.fc(x)

# 创建DQN实例
# 例如，对于Atari游戏，输入是[4, 84, 84]的图像（4帧堆叠）
# 输出是可能的动作数量
input_shape = [4, 84, 84]  # 图像输入
# input_shape = [8]  # 或者向量输入
num_actions = 6  # 假设有6个可能的动作

dqn = DQN(input_shape, num_actions)
print(dqn)
        """, language="python")
    
    # 扩展学习资源
    st.sidebar.markdown("""
    ### 深度学习扩展资源
    
    - [Deep Learning Book](https://www.deeplearningbook.org/) - Goodfellow, Bengio, Courville
    - [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) - Michael Nielsen
    - [CS231n: Convolutional Neural Networks](http://cs231n.stanford.edu/) - Stanford
    - [fast.ai](https://www.fast.ai/) - 实用深度学习课程
    """)
    
    # 基础练习
    st.markdown("## 基础练习")
    
    st.markdown("""
    1. 分析为什么深度网络比浅层网络更有效？考虑参数数量、表达能力和计算需求。
    2. 思考DQN中使用的神经网络架构为什么适合强化学习？
    3. 尝试设计一个简单的DQN架构，处理2D游戏状态（如格子世界）和离散动作空间。
    """)
    
    # 小测验
    with st.expander("小测验：检验您的理解"):
        st.markdown("""
        1. 深度学习相比传统机器学习的主要优势是什么？
           - A. 训练速度更快
           - B. 需要更少的训练数据
           - C. 自动学习层次化特征表示
           - D. 总是避免过拟合
        
        2. 在DQN中，神经网络的输出层通常表示什么？
           - A. 状态的概率分布
           - B. 每个可能动作的Q值
           - C. 状态转移函数
           - D. 奖励函数
        
        3. 以下哪种技术是为了解决深度神经网络训练中的梯度问题？
           - A. 数据增强
           - B. 批归一化
           - C. 交叉验证
           - D. 集成学习
           
        4. 为什么DQN使用两个神经网络（策略网络和目标网络）？
           - A. 提高计算效率
           - B. 减少内存使用
           - C. 处理多个任务
           - D. 稳定训练过程
        
        **答案**: 1-C, 2-B, 3-B, 4-D
        """)
    
    # 下一章预告
    st.markdown("""
    ## 下一章预告
    
    在下一章，我们将通过一个实际的**分类案例**演示神经网络的训练和应用，帮助您巩固所学知识。
    """) 