"""
DQN算法和课件的配置文件
"""

# 随机种子
SEED = 42

# 环境参数
ENV_NAME = "CartPole-v1"
RENDER_MODE = None  # 设置为"human"可以在训练过程中渲染环境

# DQN超参数
GAMMA = 0.99  # 折扣因子
EPSILON_START = 1.0  # 初始探索率
EPSILON_END = 0.01  # 最小探索率
EPSILON_DECAY = 0.995  # 探索率衰减
LEARNING_RATE = 0.001  # 学习率
BUFFER_SIZE = 10000  # 经验回放缓冲区大小
BATCH_SIZE = 64  # 批次大小
UPDATE_TARGET_EVERY = 10  # 目标网络更新频率
HIDDEN_SIZE = 64  # 神经网络隐藏层大小

# 训练参数
MAX_EPISODES = 500  # 最大回合数
MAX_STEPS = 1000  # 每个回合最大步数
TARGET_SCORE = 195.0  # 目标分数（解决环境的标准）

# 可视化参数
PLOT_UPDATE_FREQ = 50  # 绘图更新频率

# 页面标题
PAGE_TITLES = {
    "home": "深度Q网络(DQN)交互式课件",
    "theory": "DQN理论基础",
    "implementation": "DQN代码实现",
    "demo": "DQN算法演示",
    "interactive": "交互式实验"
}

# 页面图标
PAGE_ICONS = {
    "home": "🏠",
    "theory": "📚",
    "implementation": "💻",
    "demo": "🎮",
    "interactive": "🔬"
}

# 侧边栏选项
SIDEBAR_OPTIONS = [
    "首页",
    "理论基础",
    "代码实现",
    "算法演示",
    "交互式实验"
]

# 理论页面内容
THEORY_SECTIONS = [
    "Q-learning基础",
    "深度Q网络(DQN)介绍",
    "经验回放(Experience Replay)",
    "目标网络(Target Network)",
    "DQN算法流程"
]

# DQN算法伪代码
DQN_PSEUDOCODE = """
初始化:
    初始化 Q 网络参数 θ
    初始化目标网络参数 θ⁻ = θ
    初始化经验回放缓冲区 D

循环 每个回合:
    重置环境，获取初始状态 s
    循环 每个步骤 t:
        使用 ε-贪婪策略根据 Q(s, a; θ) 选择动作 a
        执行动作 a，观察奖励 r 和下一个状态 s'
        将经验 (s, a, r, s') 存储到回放缓冲区 D 中
        从 D 中随机采样一批经验 (sⱼ, aⱼ, rⱼ, s'ⱼ)
        计算目标值:
            如果 s'ⱼ 是终止状态: yⱼ = rⱼ
            否则: yⱼ = rⱼ + γ * max_a' Q(s'ⱼ, a'; θ⁻)
        执行梯度下降优化 [yⱼ - Q(sⱼ, aⱼ; θ)]² 相对于 θ
        每 C 步更新目标网络参数: θ⁻ = θ
        s = s'
        如果回合结束: 跳出内循环
""" 