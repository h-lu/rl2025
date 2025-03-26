import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from matplotlib.animation import FuncAnimation

def nn_dqn_exercise():
    """神经网络与DQN练习页面"""
    st.title("练习：神经网络与DQN")
    
    st.markdown("""
    ## 练习目标
    
    通过本练习，你将：
    1. 理解神经网络如何应用于DQN算法
    2. 实现简单的DQN神经网络结构
    3. 了解DQN训练中的关键技术
    4. 模拟DQN的更新过程
    
    完成这些练习将帮助你将神经网络知识应用到强化学习领域。
    """)
    
    # 练习1：DQN中的神经网络结构
    st.markdown("""
    ## 练习1：DQN中的神经网络结构
    
    深度Q网络(DQN)的核心是使用神经网络来近似Q函数。让我们设计一个简单的DQN神经网络结构。
    """)
    
    st.markdown("""
    ### DQN的神经网络架构
    
    DQN神经网络的输入是状态(state)，输出是每个可能动作的Q值。
    
    对于不同类型的输入，网络结构也会有所不同：
    """)
    
    # 选择输入类型
    input_type = st.selectbox(
        "选择输入类型",
        ["图像输入", "向量输入"]
    )
    
    if input_type == "图像输入":
        st.markdown("""
        #### 图像输入的DQN架构
        
        通常使用卷积神经网络(CNN)处理图像输入：
        
        ```python
        class DQN(nn.Module):
            def __init__(self, h, w, outputs):
                super(DQN, self).__init__()
                self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
                self.bn1 = nn.BatchNorm2d(16)
                self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
                self.bn2 = nn.BatchNorm2d(32)
                self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
                self.bn3 = nn.BatchNorm2d(32)
                
                # 计算卷积层输出尺寸
                def conv2d_size_out(size, kernel_size=5, stride=2):
                    return (size - (kernel_size - 1) - 1) // stride + 1
                
                convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
                convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
                linear_input_size = convw * convh * 32
                
                self.head = nn.Linear(linear_input_size, outputs)
                
            def forward(self, x):
                x = F.relu(self.bn1(self.conv1(x)))
                x = F.relu(self.bn2(self.conv2(x)))
                x = F.relu(self.bn3(self.conv3(x)))
                return self.head(x.view(x.size(0), -1))
        ```
        """)
        
        # 可视化CNN网络结构
        st.markdown("#### CNN架构可视化")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 定义层位置
        layers = [
            {"name": "输入图像", "size": (84, 84, 3), "pos": 0},
            {"name": "Conv1", "size": (40, 40, 16), "pos": 1},
            {"name": "Conv2", "size": (18, 18, 32), "pos": 2},
            {"name": "Conv3", "size": (7, 7, 32), "pos": 3},
            {"name": "全连接", "size": (512, 1), "pos": 4},
            {"name": "输出Q值", "size": (6, 1), "pos": 5}  # 假设6个动作
        ]
        
        # 绘制层
        for i, layer in enumerate(layers):
            if i == 0:  # 输入层
                rect = plt.Rectangle((layer["pos"] - 0.4, -0.4), 0.8, 0.8, color='skyblue', alpha=0.8)
                ax.add_patch(rect)
                ax.text(layer["pos"], -0.6, layer["name"], ha='center', va='center', fontsize=10)
                ax.text(layer["pos"], 0, f"{layer['size'][0]}x{layer['size'][1]}x{layer['size'][2]}", 
                       ha='center', va='center', fontsize=8)
            elif i < 4:  # 卷积层
                w_scale = min(0.8, layer["size"][0] / layers[0]["size"][0] * 0.8)
                h_scale = min(0.8, layer["size"][1] / layers[0]["size"][1] * 0.8)
                rect = plt.Rectangle((layer["pos"] - w_scale/2, -h_scale/2), w_scale, h_scale, color='lightgreen', alpha=0.8)
                ax.add_patch(rect)
                ax.text(layer["pos"], -0.6, layer["name"], ha='center', va='center', fontsize=10)
                ax.text(layer["pos"], 0, f"{layer['size'][0]}x{layer['size'][1]}x{layer['size'][2]}", 
                       ha='center', va='center', fontsize=8)
            elif i == 4:  # 全连接层
                rect = plt.Rectangle((layer["pos"] - 0.3, -0.3), 0.6, 0.6, color='lightyellow', alpha=0.8)
                ax.add_patch(rect)
                ax.text(layer["pos"], -0.6, layer["name"], ha='center', va='center', fontsize=10)
                ax.text(layer["pos"], 0, f"{layer['size'][0]}", ha='center', va='center', fontsize=8)
            else:  # 输出层
                rect = plt.Rectangle((layer["pos"] - 0.2, -0.4), 0.4, 0.8, color='salmon', alpha=0.8)
                ax.add_patch(rect)
                ax.text(layer["pos"], -0.6, layer["name"], ha='center', va='center', fontsize=10)
                ax.text(layer["pos"], 0, f"{layer['size'][0]}", ha='center', va='center', fontsize=8)
            
            # 连接线
            if i > 0:
                ax.plot([layers[i-1]["pos"] + 0.4, layer["pos"] - 0.4], [0, 0], 'k-', alpha=0.5)
        
        # 设置图表
        ax.set_xlim(-0.5, 5.5)
        ax.set_ylim(-1, 1)
        ax.set_title("图像输入的DQN网络架构")
        ax.axis('off')
        
        st.pyplot(fig)
        
    else:  # 向量输入
        st.markdown("""
        #### 向量输入的DQN架构
        
        对于向量输入，通常使用多层感知器(MLP)：
        
        ```python
        class DQN(nn.Module):
            def __init__(self, input_dim, output_dim):
                super(DQN, self).__init__()
                self.fc1 = nn.Linear(input_dim, 128)
                self.fc2 = nn.Linear(128, 128)
                self.fc3 = nn.Linear(128, output_dim)
                
            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                return self.fc3(x)
        ```
        """)
        
        # 可视化MLP网络结构
        st.markdown("#### MLP架构可视化")
        
        # 设置网络参数
        input_dim = st.slider("输入维度", 2, 8, 4)
        hidden_neurons = st.slider("隐藏层神经元数量", 8, 128, 64)
        output_dim = st.slider("输出维度 (动作数量)", 2, 10, 4)
        
        # 绘制网络结构
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 定义每层的神经元位置
        layer_sizes = [input_dim, hidden_neurons, hidden_neurons, output_dim]
        layer_positions = [1, 3, 5, 7]
        layer_names = ["输入层", "隐藏层1", "隐藏层2", "输出层"]
        layer_colors = ['skyblue', 'lightgreen', 'lightgreen', 'salmon']
        
        # 存储神经元位置
        neuron_positions = {}
        
        # 绘制每一层的神经元
        for l, (size, pos, name, color) in enumerate(zip(layer_sizes, layer_positions, layer_names, layer_colors)):
            # 只绘制有限数量的神经元
            visible_neurons = min(size, 10)
            
            # 如果神经元太多，绘制一个断点表示省略
            if size > visible_neurons:
                draw_sizes = [visible_neurons//2, 3, visible_neurons - visible_neurons//2 - 3]
                spacing = [-0.5, 0, 0.5]
            else:
                draw_sizes = [visible_neurons]
                spacing = [0]
            
            neuron_count = 0
            for group, group_size in enumerate(draw_sizes):
                for i in range(group_size):
                    if group == 1 and i == 1:  # 中间点绘制省略号
                        ax.text(pos, 0, "...", ha='center', va='center', fontsize=14)
                        continue
                        
                    # 计算垂直位置，使神经元分布均匀
                    if group_size == 1:
                        y = 0
                    else:
                        y = (i - (group_size - 1) / 2) * 0.3 + spacing[group]
                    
                    # 存储位置
                    neuron_positions[(l, neuron_count)] = (pos, y)
                    neuron_count += 1
                    
                    # 绘制神经元
                    circle = plt.Circle((pos, y), 0.2, color=color, alpha=0.8)
                    ax.add_patch(circle)
                    
                    # 省略神经元标签，避免过于拥挤
            
            # 添加层标签
            ax.text(pos, 2.5, name, ha='center', va='center', fontsize=11)
            ax.text(pos, -2.5, f"大小: {size}", ha='center', va='center', fontsize=9)
        
        # 绘制层间连接
        for l in range(len(layer_sizes) - 1):
            for i in range(min(layer_sizes[l], 10)):
                for j in range(min(layer_sizes[l + 1], 10)):
                    # 如果是省略的神经元，跳过
                    if (l, i) not in neuron_positions or (l+1, j) not in neuron_positions:
                        continue
                        
                    # 获取位置
                    pos1 = neuron_positions[(l, i)]
                    pos2 = neuron_positions[(l+1, j)]
                    
                    # 绘制连接线
                    ax.plot([pos1[0] + 0.2, pos2[0] - 0.2], 
                            [pos1[1], pos2[1]], 
                            'gray', alpha=0.2, linewidth=0.5)
        
        # 设置图表
        ax.set_xlim(0, 8)
        ax.set_ylim(-3, 3)
        ax.set_title("向量输入的DQN网络架构")
        ax.axis('off')
        
        st.pyplot(fig)
    
    # 练习2：神经网络输出与Q值
    st.markdown("""
    ## 练习2：神经网络输出与Q值
    
    在DQN中，神经网络的输出直接对应每个动作的Q值。
    """)
    
    st.markdown("""
    ### Q值与动作选择
    
    神经网络输出的Q值用于选择动作，方法是选择Q值最大的动作（贪婪策略）或根据Q值进行概率选择（ε-贪婪策略）。
    """)
    
    # 设置参数
    n_actions = st.slider("动作数量", 2, 6, 4, key="q_actions")
    epsilon = st.slider("探索率 ε", 0.0, 1.0, 0.1, 0.01)
    
    # 生成随机Q值
    np.random.seed(42)
    q_values = np.random.normal(0, 1, n_actions)
    
    # 创建数据框显示Q值
    q_df = pd.DataFrame({
        "动作": [f"动作 {i+1}" for i in range(n_actions)],
        "Q值": q_values
    })
    
    st.markdown("#### 神经网络输出的Q值")
    st.write(q_df)
    
    # 确定贪婪动作和ε-贪婪动作
    greedy_action = np.argmax(q_values)
    
    # 模拟ε-贪婪选择
    if random.random() < epsilon:
        random_action = random.randint(0, n_actions - 1)
        while random_action == greedy_action:
            random_action = random.randint(0, n_actions - 1)
        epsilon_greedy_action = random_action
        selection_type = "随机探索"
    else:
        epsilon_greedy_action = greedy_action
        selection_type = "贪婪选择"
    
    # 显示选择结果
    st.markdown(f"""
    #### 动作选择结果
    
    - **贪婪策略选择**: 动作 {greedy_action + 1} (Q值 = {q_values[greedy_action]:.4f})
    - **ε-贪婪策略选择**: 动作 {epsilon_greedy_action + 1} ({selection_type}, Q值 = {q_values[epsilon_greedy_action]:.4f})
    
    ε-贪婪策略有{epsilon*100:.1f}%的概率随机选择动作，{100-epsilon*100:.1f}%的概率选择Q值最大的动作。
    """)
    
    # 可视化Q值和选择
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(range(n_actions), q_values, color='lightblue')
    
    # 标记贪婪动作
    bars[greedy_action].set_color('green')
    ax.text(greedy_action, q_values[greedy_action] + 0.1, "贪婪选择", 
           ha='center', va='bottom', fontweight='bold')
    
    # 如果ε-贪婪选择了不同的动作，标记它
    if epsilon_greedy_action != greedy_action:
        bars[epsilon_greedy_action].set_color('orange')
        ax.text(epsilon_greedy_action, q_values[epsilon_greedy_action] + 0.1, "ε-贪婪选择", 
               ha='center', va='bottom', fontweight='bold')
    
    ax.set_xticks(range(n_actions))
    ax.set_xticklabels([f"动作 {i+1}" for i in range(n_actions)])
    ax.set_ylabel('Q值')
    ax.set_title('基于Q值的动作选择')
    ax.grid(alpha=0.3)
    
    st.pyplot(fig)
    
    # 练习3：DQN更新过程
    st.markdown("""
    ## 练习3：DQN更新过程
    
    DQN的训练涉及使用时序差分(TD)误差来更新网络参数。
    """)
    
    st.markdown("""
    ### DQN的更新公式
    
    DQN使用以下公式计算目标Q值和损失：
    
    $$Q_{target}(s, a) = r + \gamma \max_{a'} Q(s', a')$$
    
    $$Loss = \frac{1}{N}\sum_i (Q_{target}(s_i, a_i) - Q(s_i, a_i))^2$$
    
    其中：
    - $r$ 是即时奖励
    - $\gamma$ 是折扣因子
    - $s'$ 是下一状态
    - $a'$ 是下一状态的所有可能动作
    - $N$ 是批量大小
    """)
    
    # 交互式DQN更新演示
    st.markdown("### 交互式DQN更新演示")
    
    # 设置参数
    reward = st.slider("即时奖励 r", -10.0, 10.0, 1.0, 0.1)
    gamma = st.slider("折扣因子 γ", 0.0, 1.0, 0.9, 0.01)
    
    # 为当前状态和下一状态生成Q值
    current_q = np.random.normal(0, 1, n_actions)
    next_q = np.random.normal(0, 1, n_actions)
    
    # 选择一个动作
    action = random.randint(0, n_actions - 1)
    
    # 计算目标Q值
    max_next_q = np.max(next_q)
    target_q = reward + gamma * max_next_q
    
    # 计算TD误差和损失
    td_error = target_q - current_q[action]
    loss = td_error ** 2
    
    # 显示计算过程
    st.markdown("#### 数据")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**当前状态Q值**")
        current_q_df = pd.DataFrame({
            "动作": [f"动作 {i+1}" for i in range(n_actions)],
            "Q值": current_q
        })
        st.write(current_q_df)
        
        st.markdown(f"**选择的动作**: 动作 {action + 1}")
        st.markdown(f"**当前Q值 Q(s,a)**: {current_q[action]:.4f}")
    
    with col2:
        st.markdown("**下一状态Q值**")
        next_q_df = pd.DataFrame({
            "动作": [f"动作 {i+1}" for i in range(n_actions)],
            "Q值": next_q
        })
        st.write(next_q_df)
        
        st.markdown(f"**最大下一Q值 max Q(s',a')**: {max_next_q:.4f}")
    
    # 计算结果
    st.markdown("""
    #### 计算结果
    """)
    
    st.markdown(f"""
    **目标Q值计算**:
    
    $Q_{{target}}(s, a) = r + \gamma \max_{{a'}} Q(s', a')$
    
    $Q_{{target}}(s, a) = {reward} + {gamma} \times {max_next_q:.4f} = {target_q:.4f}$
    
    **TD误差计算**:
    
    $TD_{error} = Q_{{target}}(s, a) - Q(s, a)$
    
    $TD_{error} = {target_q:.4f} - {current_q[action]:.4f} = {td_error:.4f}$
    
    **损失计算**:
    
    $Loss = (TD_{error})^2 = {td_error:.4f}^2 = {loss:.4f}$
    
    这个损失将用于通过反向传播更新DQN的参数。
    """)
    
    # 可视化Q值更新
    st.markdown("### Q值更新可视化")
    
    # 定义学习率
    learning_rate = st.slider("学习率 α", 0.01, 1.0, 0.1, 0.01)
    
    # 计算更新后的Q值
    updated_q = current_q.copy()
    updated_q[action] = current_q[action] + learning_rate * td_error
    
    # 可视化
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(n_actions)
    width = 0.35
    
    ax.bar(x - width/2, current_q, width, label='当前Q值')
    ax.bar(x + width/2, updated_q, width, label='更新后Q值')
    
    # 标记选择的动作
    ax.plot([action - width/2, action - width/2], [0, current_q[action]], 'r--', linewidth=2)
    ax.plot([action + width/2, action + width/2], [0, updated_q[action]], 'r--', linewidth=2)
    
    ax.text(action, max(current_q[action], updated_q[action]) + 0.2, 
           f"Δ = {updated_q[action] - current_q[action]:.4f}", 
           ha='center', va='bottom')
    
    ax.set_xticks(x)
    ax.set_xticklabels([f"动作 {i+1}" for i in range(n_actions)])
    ax.set_ylabel('Q值')
    ax.set_title('DQN更新效果')
    ax.legend()
    ax.grid(alpha=0.3)
    
    st.pyplot(fig)
    
    # 练习4：DQN的稳定性技巧
    st.markdown("""
    ## 练习4：DQN的稳定性技巧
    
    DQN引入了两项关键技术来解决训练不稳定性：经验回放和目标网络。
    """)
    
    # 经验回放
    st.markdown("""
    ### 经验回放 (Experience Replay)
    
    经验回放通过存储和重用过去的经验打破样本之间的相关性，提高数据利用效率。
    """)
    
    # 可视化经验回放过程
    st.markdown("#### 经验回放可视化")
    
    buffer_size = st.slider("经验回放缓冲区大小", 10, 100, 20)
    batch_size = st.slider("训练批量大小", 2, 20, 4)
    
    # 创建模拟经验数据
    experiences = []
    for i in range(buffer_size):
        experiences.append({
            "id": i,
            "state": f"状态{i}",
            "action": f"动作{random.randint(1, 4)}",
            "reward": round(random.uniform(-5, 5), 1),
            "next_state": f"状态{i+1}"
        })
    
    # 可视化缓冲区和采样
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 绘制经验缓冲区
    for i, exp in enumerate(experiences):
        rect = plt.Rectangle((i - buffer_size/2, 0), 0.8, 1, color='lightblue', alpha=0.8)
        ax.add_patch(rect)
        ax.text(i - buffer_size/2 + 0.4, 0.5, f"{exp['id']}", ha='center', va='center', fontsize=8)
    
    # 随机选择一些样本
    sampled_indices = random.sample(range(buffer_size), batch_size)
    
    # 标记采样的经验
    for idx in sampled_indices:
        rect = plt.Rectangle((idx - buffer_size/2, 0), 0.8, 1, color='orange', alpha=0.8)
        ax.add_patch(rect)
        ax.text(idx - buffer_size/2 + 0.4, 0.5, f"{experiences[idx]['id']}", ha='center', va='center', fontsize=8)
    
    # 绘制训练批次
    for i, idx in enumerate(sampled_indices):
        rect = plt.Rectangle((i - batch_size/2, -2), 0.8, 1, color='orange', alpha=0.8)
        ax.add_patch(rect)
        ax.text(i - batch_size/2 + 0.4, -1.5, f"{experiences[idx]['id']}", ha='center', va='center', fontsize=8)
        
        # 连接线
        ax.plot([idx - buffer_size/2 + 0.4, i - batch_size/2 + 0.4], [0, -1], 'k--', alpha=0.5)
    
    # 标签
    ax.text(-buffer_size/2 - 1, 0.5, "经验缓冲区", ha='right', va='center', fontsize=12)
    ax.text(-batch_size/2 - 1, -1.5, "训练批次", ha='right', va='center', fontsize=12)
    
    # 设置图表
    ax.set_xlim(-buffer_size/2 - 2, buffer_size/2 + 1)
    ax.set_ylim(-3, 2)
    ax.set_title("经验回放采样过程")
    ax.axis('off')
    
    st.pyplot(fig)
    
    # 显示采样的经验详情
    st.markdown("#### 采样的经验详情")
    
    sampled_exp = [experiences[idx] for idx in sampled_indices]
    sampled_df = pd.DataFrame(sampled_exp)
    st.write(sampled_df)
    
    # 目标网络
    st.markdown("""
    ### 目标网络 (Target Network)
    
    目标网络是DQN的另一个关键稳定性技术，它使用单独的网络生成训练目标，减少参数更新的振荡。
    """)
    
    # 目标网络更新演示
    st.markdown("#### 目标网络更新演示")
    
    # 设置参数
    update_frequency = st.slider("目标网络更新频率 (步数)", 10, 1000, 100)
    current_step = st.slider("当前步数", 0, 1000, 150)
    
    # 计算上次更新和下次更新的步数
    last_update = (current_step // update_frequency) * update_frequency
    next_update = last_update + update_frequency
    
    # 生成主网络和目标网络的参数
    np.random.seed(42)
    main_params = np.random.normal(0, 1, 5)
    
    # 目标网络参数在更新点会复制主网络参数
    if last_update == current_step:
        target_params = main_params.copy()
    else:
        # 使用早期的主网络参数作为目标网络参数
        target_params = main_params * 0.7 + np.random.normal(0, 0.1, 5)
    
    # 可视化网络参数
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(main_params))
    width = 0.35
    
    ax.bar(x - width/2, main_params, width, label='主网络参数')
    ax.bar(x + width/2, target_params, width, label='目标网络参数')
    
    ax.set_xticks(x)
    ax.set_xticklabels([f"参数 {i+1}" for i in range(len(main_params))])
    ax.set_ylabel('参数值')
    ax.set_title(f'主网络和目标网络参数 (步数: {current_step})')
    ax.legend()
    ax.grid(alpha=0.3)
    
    st.pyplot(fig)
    
    # 目标网络时间线
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # 绘制时间线
    ax.plot([0, 1000], [0, 0], 'k-', alpha=0.3)
    
    # 标记更新点
    for step in range(0, 1001, update_frequency):
        ax.plot([step, step], [-0.2, 0.2], 'g-', alpha=0.7 if step <= current_step else 0.3)
        ax.text(step, 0.3, str(step), ha='center', va='bottom', 
               fontweight='bold' if step == last_update else 'normal',
               alpha=0.7 if step <= current_step else 0.3)
    
    # 标记当前步骤
    ax.plot([current_step, current_step], [-0.4, 0.4], 'r-')
    ax.text(current_step, 0.5, f"当前步数: {current_step}", ha='center', va='bottom', color='red')
    
    # 标记上次更新和下次更新
    ax.text(last_update, -0.5, f"上次更新: {last_update}", ha='center', va='top', color='blue')
    ax.text(next_update, -0.5, f"下次更新: {next_update}", ha='center', va='top', color='purple')
    
    # 设置图表
    ax.set_xlim(0, 1000)
    ax.set_ylim(-1, 1)
    ax.set_title("目标网络更新时间线")
    ax.set_xlabel("训练步数")
    ax.set_yticks([])
    
    st.pyplot(fig)
    
    st.markdown(f"""
    **目标网络更新说明**:
    
    1. 目标网络每{update_frequency}步更新一次
    2. 当前步数: {current_step}
    3. 上次更新步数: {last_update}
    4. 距离下次更新还有 {next_update - current_step} 步
    
    在更新点，目标网络的参数会直接复制主网络的参数。这种延迟更新机制提供了更稳定的学习目标。
    """)
    
    # 思考题
    st.markdown("""
    ## 思考题
    
    1. 为什么DQN需要使用经验回放和目标网络来稳定训练？
    
    2. 如何选择合适的神经网络结构用于DQN？不同的任务可能需要什么样的网络设计？
    
    3. DQN的哪些方面使它能够处理高维状态空间，相比传统Q学习有何优势？
    """)
    
    # 提示
    with st.expander("查看思考题提示"):
        st.markdown("""
        **思考题提示**：
        
        1. **DQN的稳定性问题**:
           - 神经网络训练需要iid数据假设，但RL中连续收集的样本高度相关
           - Q-learning是自举算法，目标值依赖于当前估计，容易导致振荡
           - 非线性函数近似(神经网络)可能导致不收敛
           - 经验回放打破样本相关性，目标网络提供稳定的学习目标
        
        2. **神经网络结构选择**:
           - 图像输入: 使用CNN捕获空间特征
           - 向量状态: 使用MLP更高效
           - 序列决策问题: 可以考虑RNN/LSTM
           - 复杂环境: 网络深度和宽度需要增加
           - 简单环境: 较小的网络可以更快收敛
        
        3. **DQN处理高维状态空间的能力**:
           - 神经网络可以自动学习特征表示，不需要手动特征工程
           - 端到端学习直接从原始感知数据到动作
           - 具有泛化能力，可应用于未见过的状态
           - 与传统Q学习相比，不受维度灾难影响
        """)
    
    # 小结
    st.markdown("""
    ## 小结
    
    在本练习中，你：
    1. 了解了DQN中神经网络的结构设计
    2. 学习了Q值与动作选择的关系
    3. 掌握了DQN的更新过程和公式
    4. 理解了经验回放和目标网络的作用
    
    这些知识将帮助你理解如何将神经网络技术应用于强化学习，构建能够处理复杂任务的智能体。
    """)

if __name__ == "__main__":
    nn_dqn_exercise() 