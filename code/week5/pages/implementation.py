"""
DQN代码实现页面

展示和解释DQN算法的关键代码实现
"""

import streamlit as st
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from utils.visualization import display_dqn_code_segments, configure_matplotlib_fonts

# 确保matplotlib可以显示中文字体
configure_matplotlib_fonts()

def render_implementation_page():
    """渲染DQN代码实现页面"""
    st.title(f"{config.PAGE_ICONS['implementation']} {config.PAGE_TITLES['implementation']}")
    
    st.markdown("""
    ## DQN算法的代码实现
    
    本页面详细介绍了DQN算法的核心代码实现。我们将从以下几个关键组件展开：
    1. 网络架构
    2. 经验回放缓冲区
    3. DQN智能体
    4. 训练流程
    5. 评估和可视化
    """)
    
    # 多选项卡页面
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "网络架构", 
        "经验回放缓冲区", 
        "DQN智能体", 
        "训练流程", 
        "评估和可视化"
    ])
    
    # 网络架构选项卡
    with tab1:
        st.markdown("""
        ### DQN网络架构
        
        DQN使用神经网络来近似Q函数。对于CartPole环境，我们使用简单的全连接网络，
        但对于更复杂的任务（如Atari游戏），通常会使用卷积神经网络处理图像输入。
        """)
        
        st.code("""
def create_dqn_model(state_size, action_size, hidden_size=64):
    '创建DQN神经网络模型'
    model = keras.Sequential([
        # 输入层与第一个隐藏层
        layers.Dense(hidden_size, activation='relu', input_shape=(state_size,)),
        # 第二个隐藏层
        layers.Dense(hidden_size, activation='relu'),
        # 输出层 - 没有激活函数，因为我们需要预测Q值（可以是任意值）
        layers.Dense(action_size, activation='linear')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse'  # 均方误差损失函数
    )
    
    return model
        """, language="python")
        
        st.markdown("""
        #### 网络架构解析
        
        上述代码创建了一个简单的前馈神经网络，包含：
        
        1. **输入层**：接收环境状态向量，维度为`state_size`（CartPole为4）
        2. **隐藏层**：两个全连接层，每层包含`hidden_size`个神经元（默认64），使用ReLU激活函数
        3. **输出层**：神经元数量等于动作空间大小`action_size`（CartPole为2），使用线性激活函数
        
        #### 注意点
        
        - 输出层使用**线性激活函数**而不是sigmoid或softmax，因为Q值可以是任意实数
        - 我们使用**均方误差(MSE)损失函数**来最小化预测Q值和目标Q值之间的差距
        - 使用**Adam优化器**进行参数更新，这是一种自适应学习率的优化算法
        """)
        
        # 对比不同环境的网络架构
        st.subheader("不同环境的网络架构")
        
        arch_df = pd.DataFrame([
            {
                "环境": "CartPole（低维状态）", 
                "输入": "4维状态向量",
                "网络结构": "全连接网络 (64-64-2)",
                "特点": "简单，训练快速",
            },
            {
                "环境": "Atari游戏（图像输入）", 
                "输入": "84x84像素图像",
                "网络结构": "卷积网络 + 全连接层",
                "特点": "处理图像特征，参数更多",
            },
            {
                "环境": "连续动作空间", 
                "输入": "状态向量",
                "网络结构": "双网络（策略网络+Q网络）",
                "特点": "需要使用不同的算法（如DDPG）",
            }
        ])
        
        st.table(arch_df)
    
    # 经验回放缓冲区选项卡
    with tab2:
        st.markdown("""
        ### 经验回放缓冲区实现
        
        经验回放是DQN的核心创新之一，它存储智能体的交互经验，并随机采样来训练网络，从而打破样本之间的时序相关性。
        """)
        
        st.code("""
class ReplayBuffer:
    def __init__(self, capacity):
        '''初始化经验回放缓冲区'''
        self.buffer = deque(maxlen=capacity)  # 使用双端队列存储经验，自动管理容量
    
    def add(self, state, action, reward, next_state, done):
        '''添加经验到缓冲区'''
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        '''随机采样一批经验'''
        # 确保batch_size不超过缓冲区大小
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
            
        # 随机采样
        experiences = random.sample(self.buffer, batch_size)
        
        # 解包经验数据
        states, actions, rewards, next_states, dones = zip(*experiences)
        
        # 转换为NumPy数组，便于批量处理
        return (
            np.array(states, dtype=np.float32), 
            np.array(actions, dtype=np.int32), 
            np.array(rewards, dtype=np.float32), 
            np.array(next_states, dtype=np.float32), 
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        '''获取缓冲区中的经验数量'''
        return len(self.buffer)
    
    def is_ready(self, batch_size):
        '''检查是否有足够的经验可供采样'''
        return len(self.buffer) >= batch_size
        """, language="python")
        
        st.markdown("""
        #### 经验回放缓冲区解析
        
        1. **初始化**：创建一个固定容量的双端队列(`deque`)存储经验
        2. **添加经验**：将经验元组`(state, action, reward, next_state, done)`添加到缓冲区
        3. **随机采样**：从缓冲区随机选择一批经验用于训练
        4. **数据处理**：将采样的经验转换为NumPy数组，便于批量处理
        
        #### 优化方向
        
        基本的经验回放可以进一步优化，例如：
        
        - **优先经验回放(Prioritized Experience Replay)**：
          根据TD误差为经验分配优先级，更频繁地采样有更高学习价值的经验
          
        - **分层经验回放(Hindsight Experience Replay)**：
          通过重新标记奖励，从失败的经验中学习
          
        - **压缩存储**：对于大型问题，可以压缩存储经验以节省内存
        """)
        
        # 缓冲区容量对比
        st.subheader("缓冲区容量的影响")
        
        buffer_df = pd.DataFrame([
            {
                "缓冲区大小": "小 (1,000)", 
                "优点": "内存占用小，更新频繁",
                "缺点": "容易忘记旧经验，数据利用率低",
                "适用场景": "简单环境，快速原型设计",
            },
            {
                "缓冲区大小": "中 (10,000-100,000)", 
                "优点": "平衡内存占用和样本多样性",
                "缺点": "可能需要更多训练才能充满缓冲区",
                "适用场景": "CartPole等中等复杂度环境",
            },
            {
                "缓冲区大小": "大 (1,000,000+)", 
                "优点": "保留更多多样化经验，更稳定的训练",
                "缺点": "内存占用大，初始采样质量低",
                "适用场景": "Atari游戏等复杂环境",
            }
        ])
        
        st.table(buffer_df)
    
    # DQN智能体选项卡
    with tab3:
        st.markdown("""
        ### DQN智能体实现
        
        DQN智能体整合了所有组件，管理与环境的交互、经验存储、网络训练等核心功能。下面是智能体的关键部分实现：
        """)
        
        # 初始化方法
        st.subheader("智能体初始化")
        st.code("""
def __init__(self, state_size, action_size, gamma=0.99, epsilon_start=1.0, 
             epsilon_end=0.01, epsilon_decay=0.995, buffer_size=10000, 
             batch_size=64, update_target_every=10):
    # 状态和动作空间
    self.state_size = state_size
    self.action_size = action_size
    
    # 超参数
    self.gamma = gamma  # 折扣因子
    self.epsilon = epsilon_start  # 当前探索率
    self.epsilon_end = epsilon_end  # 最小探索率
    self.epsilon_decay = epsilon_decay  # 探索率衰减系数
    self.batch_size = batch_size  # 训练批次大小
    self.update_target_every = update_target_every  # 目标网络更新频率
    
    # 创建Q网络和目标网络
    self.q_network = create_dqn_model(state_size, action_size)
    self.target_network = create_dqn_model(state_size, action_size)
    self.target_network.set_weights(self.q_network.get_weights())  # 初始权重相同
    
    # 创建经验回放缓冲区
    self.memory = ReplayBuffer(buffer_size)
    
    # 记录学习步数和训练信息
    self.t_step = 0
    self.loss_history = []
        """, language="python")
        
        # 动作选择方法
        st.subheader("动作选择(ε-贪婪策略)")
        st.code("""
def act(self, state, eval_mode=False):
    '''根据当前状态选择动作'''
    # 将状态转换为批次格式
    state = np.reshape(state, [1, self.state_size])
    
    # 评估模式或使用ε-贪婪策略选择动作
    if eval_mode or random.random() > self.epsilon:
        # 开发：选择Q值最大的动作
        action_values = self.q_network.predict(state, verbose=0)
        return np.argmax(action_values[0])
    else:
        # 探索：随机选择动作
        return random.choice(np.arange(self.action_size))
        """, language="python")
        
        # 学习方法
        st.subheader("学习方法(核心训练逻辑)")
        st.code("""
def learn(self, experiences):
    '''从经验批次中学习'''
    states, actions, rewards, next_states, dones = experiences
    
    # 从目标网络中获取下一个状态的最大Q值
    target_q_values = self.target_network.predict(next_states, verbose=0)
    max_target_q = np.max(target_q_values, axis=1)
    
    # 计算目标Q值
    targets = rewards + (self.gamma * max_target_q * (1 - dones))
    
    # 获取当前预测的Q值并更新目标
    target_f = self.q_network.predict(states, verbose=0)
    
    # 只更新选择的动作对应的Q值
    for i, action in enumerate(actions):
        target_f[i][action] = targets[i]
    
    # 训练Q网络
    history = self.q_network.fit(states, target_f, epochs=1, verbose=0)
    loss = history.history['loss'][0]
    self.loss_history.append(loss)
    
    # 更新ε值，逐渐减少探索
    self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    return loss
        """, language="python")
        
        # 完整训练步骤
        st.subheader("完整训练步骤")
        st.code("""
def step(self, state, action, reward, next_state, done):
    '''执行学习步骤'''
    # 将经验添加到缓冲区
    self.memory.add(state, action, reward, next_state, done)
    
    # 更新时间步
    self.t_step += 1
    
    # 如果缓冲区中有足够的样本，执行学习
    loss = None
    if self.memory.is_ready(self.batch_size) and self.t_step % 4 == 0:
        experiences = self.memory.sample(self.batch_size)
        loss = self.learn(experiences)
        
        # 定期更新目标网络
        if self.t_step % self.update_target_every == 0:
            self.target_network.set_weights(self.q_network.get_weights())
    
    return loss
        """, language="python")
        
        st.markdown("""
        #### DQN智能体的关键要点
        
        1. **初始化**：创建两个网络（Q网络和目标网络）和经验回放缓冲区
        
        2. **动作选择**：使用ε-贪婪策略平衡探索和利用
           - 以概率ε随机选择动作
           - 以概率1-ε选择Q值最大的动作
           - ε随时间逐渐衰减，智能体逐渐从探索转向利用
        
        3. **学习过程**：
           - 从目标网络获取目标Q值
           - 计算TD目标：$y = r + \gamma \max_{a'} Q(s', a'; \theta^-)$ (非终止状态)
           - 仅更新所选动作的Q值，其他动作的Q值保持不变
           - 使用随机梯度下降更新网络参数
        
        4. **训练步骤**：
           - 存储经验到缓冲区
           - 每隔几步从缓冲区采样并学习
           - 定期更新目标网络
        """)
    
    # 训练流程选项卡
    with tab4:
        st.markdown("""
        ### DQN训练流程
        
        训练流程协调智能体与环境的交互，记录训练进度，并决定何时停止训练。
        """)
        
        st.code("""
def train_dqn(env_name='CartPole-v1', n_episodes=1000, max_t=1000, eps_start=1.0, 
              eps_end=0.01, eps_decay=0.995, target_score=195.0):
    # 创建环境
    env = gym.make(env_name)
    print(f"状态空间: {env.observation_space.shape[0]}")
    print(f"动作空间: {env.action_space.n}")
    
    # 创建DQN智能体
    agent = DQNAgent(
        state_size=env.observation_space.shape[0], 
        action_size=env.action_space.n, 
        epsilon_start=eps_start, 
        epsilon_end=eps_end, 
        epsilon_decay=eps_decay
    )
    
    # 记录训练过程
    scores = []
    avg_scores = []
    epsilon_history = []
    
    # 训练循环
    for i_episode in range(1, n_episodes+1):
        # 重置环境
        state, _ = env.reset(seed=42)
        score = 0
        
        # 单个回合循环
        for t in range(max_t):
            # 智能体选择动作
            action = agent.act(state)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 智能体学习
            agent.step(state, action, reward, next_state, done)
            
            # 更新状态和累积奖励
            state = next_state
            score += reward
            
            # 如果回合结束，跳出循环
            if done:
                break
        
        # 记录回合结果
        scores.append(score)
        avg_score = np.mean(scores[-100:])  # 计算最近100回合的平均分数
        avg_scores.append(avg_score)
        epsilon_history.append(agent.epsilon)
        
        # 打印进度
        print(f'回合 {i_episode}/{n_episodes}, 分数: {score:.2f}, 平均分数: {avg_score:.2f}, epsilon: {agent.epsilon:.4f}')
        
        # 如果达到目标分数，提前结束训练
        if avg_score >= target_score and i_episode >= 100:
            print(f'环境在{i_episode}回合后解决!')
            agent.save(f'dqn_checkpoint_{env_name}_{i_episode}.h5')
            break
    
    return agent, scores, avg_scores, epsilon_history
        """, language="python")
        
        st.markdown("""
        #### 训练流程解析
        
        1. **环境初始化**：创建指定的gym环境
        
        2. **智能体初始化**：创建DQN智能体，设置探索参数
        
        3. **训练循环**：包含两层循环
           - 外层循环：遍历每个训练回合
           - 内层循环：在单个回合内，智能体与环境交互直到回合结束
        
        4. **记录与监控**：
           - 记录每个回合的分数
           - 计算移动平均分数（最近100回合）
           - 跟踪ε值的变化
        
        5. **提前停止**：当达到目标分数时保存模型并提前结束训练
        
        #### 训练评估指标
        
        - **单回合分数**：单个回合内获得的总奖励
        - **平均分数**：最近100回合的平均分数，用于评估训练进度
        - **探索率(ε)**：跟踪探索率的变化
        - **训练稳定性**：分数的方差或标准差
        """)
        
        # 训练参数比较
        st.subheader("训练参数的影响")
        
        params_df = pd.DataFrame([
            {
                "参数": "折扣因子 (γ)", 
                "常用范围": "0.9-0.99",
                "影响": "控制未来奖励的重要性，值越大越重视长期奖励",
                "调整建议": "长期任务使用接近1的值；短期任务可以使用较小的值",
            },
            {
                "参数": "初始探索率 (ε_start)", 
                "常用范围": "0.9-1.0",
                "影响": "初始随机探索的程度",
                "调整建议": "通常从1.0开始，确保初期充分探索",
            },
            {
                "参数": "探索率衰减 (ε_decay)", 
                "常用范围": "0.95-0.999",
                "影响": "控制从探索向利用的转换速度",
                "调整建议": "复杂环境使用较慢的衰减；简单环境可以快速衰减",
            },
            {
                "参数": "目标网络更新频率", 
                "常用范围": "10-1000步",
                "影响": "控制目标的稳定性",
                "调整建议": "太频繁会导致不稳定；太慢会延迟学习",
            },
            {
                "参数": "学习率", 
                "常用范围": "0.0001-0.001",
                "影响": "控制网络更新的步长",
                "调整建议": "太大导致不稳定；太小则收敛缓慢",
            }
        ])
        
        st.table(params_df)
    
    # 评估和可视化选项卡
    with tab5:
        st.markdown("""
        ### 评估和可视化
        
        训练完成后，我们需要评估智能体的性能并可视化训练过程，以便理解学习效果和改进算法。
        """)
        
        # 评估代码
        st.subheader("智能体评估")
        st.code("""
def evaluate_agent(agent, env_name='CartPole-v1', n_episodes=10, render=False):
    '''评估训练好的智能体'''
    # 创建环境
    env = gym.make(env_name, render_mode="human" if render else None)
    scores = []
    
    for i_episode in range(1, n_episodes+1):
        state, _ = env.reset(seed=42)
        score = 0
        
        while True:
            # 使用评估模式选择动作（关闭探索）
            action = agent.act(state, eval_mode=True)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 更新状态和累积奖励
            state = next_state
            score += reward
            
            if done:
                break
        
        scores.append(score)
        print(f'测试回合 {i_episode}/{n_episodes}, 分数: {score:.2f}')
    
    print(f'平均测试分数: {np.mean(scores):.2f}')
    return scores
        """, language="python")
        
        # 可视化代码
        st.subheader("训练过程可视化")
        st.code("""
def plot_training_progress(scores, avg_scores, epsilon_history=None):
    '''绘制训练进度图表'''
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # 绘制分数
    color = 'tab:blue'
    ax1.set_xlabel('回合数')
    ax1.set_ylabel('分数', color=color)
    ax1.plot(scores, alpha=0.3, color=color, label='回合分数')
    ax1.plot(avg_scores, color=color, label='平均分数(最近100回合)')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # 如果提供了epsilon历史记录，也绘制它
    if epsilon_history is not None:
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Epsilon', color=color)
        ax2.plot(epsilon_history, color=color, label='Epsilon')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim([0, 1.1])
    
    fig.tight_layout()
    lines1, labels1 = ax1.get_legend_handles_labels()
    
    if epsilon_history is not None:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    else:
        ax1.legend(lines1, labels1, loc='upper left')
    
    plt.title('DQN训练进度')
    
    return fig
        """, language="python")
        
        st.markdown("""
        #### 评估方法解析
        
        1. **智能体评估**：
           - 在评估模式下运行智能体，关闭随机探索
           - 记录多个回合的分数
           - 计算平均性能指标
        
        2. **可视化训练过程**：
           - 绘制回合分数随时间的变化
           - 显示移动平均分数，反映训练稳定性
           - 跟踪ε值的衰减曲线
        
        #### 可视化的重要性
        
        1. **监控收敛性**：观察分数是否稳定增长并最终收敛
        2. **检测过拟合**：如果性能开始下降，可能是过拟合或不稳定性的征兆
        3. **超参数调优**：通过比较不同参数设置下的学习曲线，找到最佳配置
        4. **理解算法行为**：分析探索率与性能之间的关系
        """)
        
        # 添加示例图
        st.subheader("训练曲线示例")
        
        # 模拟数据生成简单示例
        episodes = np.arange(200)
        
        # 生成模拟的训练数据
        np.random.seed(42)
        scores = []
        score = 20
        for i in range(200):
            if i < 50:
                score += np.random.normal(1, 5)
            else:
                score += np.random.normal(2, 3) if score < 200 else np.random.normal(0, 2)
            score = max(0, min(score, 210))  # 限制在0-210范围内
            scores.append(score)
        
        # 计算移动平均
        avg_scores = []
        for i in range(200):
            avg_scores.append(np.mean(scores[max(0, i-99):i+1]))
        
        # 生成epsilon值
        epsilon = 1.0
        epsilon_end = 0.01
        epsilon_decay = 0.98
        epsilon_history = []
        for i in range(200):
            epsilon_history.append(epsilon)
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # 绘制带有双轴的训练曲线
        fig, ax1 = plt.subplots(figsize=(10, 5))
        
        color = 'tab:blue'
        ax1.set_xlabel('回合数')
        ax1.set_ylabel('分数', color=color)
        ax1.plot(episodes, scores, alpha=0.3, color=color, label='回合分数')
        ax1.plot(episodes, avg_scores, color=color, label='平均分数(最近100回合)')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim([0, 220])
        
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Epsilon', color=color)
        ax2.plot(episodes, epsilon_history, color=color, label='Epsilon')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim([0, 1.1])
        
        fig.tight_layout()
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.title('DQN训练进度示例')
        
        st.pyplot(fig)
        st.caption("典型的DQN训练曲线：回合分数、移动平均分数和探索率(ε)随回合数的变化")
    
    # 显示DQN关键代码段
    st.subheader("DQN关键代码段概览")
    display_dqn_code_segments() 