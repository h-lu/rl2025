# DQN改进算法微型实现

这个项目包含了几种DQN（Deep Q-Network）改进算法的轻量级实现，专为Blackjack-v1环境优化。这些实现保持了算法的核心思想，同时简化了网络架构，使得训练速度更快，更适合在学生笔记本上运行。

## 项目结构

- `dqn_mini.py`: 基础DQN算法实现
- `double_dqn_mini.py`: Double DQN算法实现
- `dueling_dqn_mini.py`: Dueling DQN算法实现
- `prioritized_replay_mini.py`: 优先经验回放算法实现
- `run_experiments.py`: 运行并比较所有算法的脚本
- `main.py`: 用户友好的主菜单界面
- `results/`: 保存实验结果的目录
- `models/`: 保存训练模型的目录

## 算法说明

### 1. 基础DQN

基础DQN（Deep Q-Network）是将深度学习与Q-learning相结合的强化学习算法。它使用神经网络来近似Q值函数，并利用经验回放和目标网络来稳定训练过程。

关键特性：
- 经验回放：打破样本间的相关性
- 目标网络：减少训练不稳定性
- ϵ-greedy探索策略：平衡探索与利用

### 2. Double DQN

Double DQN改进了基础DQN中由于使用max操作导致的Q值过高估计问题。它通过分离动作选择和评估来减少这种正向偏差。

关键改进：
- 使用在线网络选择动作，目标网络评估动作价值
- 减少Q值过高估计
- 提高算法稳定性

参考文献：[Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)

### 3. Dueling DQN

Dueling DQN将Q值分解为状态价值函数V(s)和优势函数A(s,a)两部分。这种结构使得网络能够更有效地学习哪些状态有价值，而不必关心每个状态下的每个动作的效果。

关键改进：
- 分离状态价值和动作优势
- 更高效地学习状态价值
- 在动作对结果影响不大的状态下表现更好

参考文献：[Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)

### 4. 优先经验回放 (Prioritized Experience Replay)

优先经验回放改进了标准经验回放中均匀采样的方式，根据经验的TD误差给予重要样本更高的采样概率，使得学习更加高效。

关键改进：
- 根据TD误差给予样本优先级
- 使用SumTree数据结构高效实现优先级采样
- 通过重要性采样权重修正引入的偏差

参考文献：[Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)

## 环境说明

本项目使用Blackjack-v1作为强化学习环境，这是一个21点游戏环境。

### Blackjack-v1环境特点
- **状态空间**：(玩家当前点数, 庄家明牌点数, 玩家是否有可用的A)
- **动作空间**：停牌(0)和要牌(1)两种动作
- **奖励**：赢得游戏+1分，平局0分，输掉游戏-1分

这个环境非常适合算法原型开发和教学，因为它的状态空间相对简单，训练速度快，同时能够展示各种算法改进的效果。

## 使用方法

### 环境配置

1. 确保已安装必要的Python库：
```bash
pip install gymnasium numpy tensorflow matplotlib pandas
```

2. 克隆或下载本项目到本地。

### 运行方式

在项目目录下，有两种方式运行实验：

1. **使用主菜单界面（推荐）**:
```bash
python main.py
```
这将显示一个交互式菜单，您可以选择运行单个算法或所有算法的对比实验。

2. **直接运行对比实验**:
```bash
python run_experiments.py
```
这将自动运行所有四种算法并生成比较结果。

3. **单独运行某个算法**:
```bash
python dqn_mini.py  # 运行基础DQN
python double_dqn_mini.py  # 运行Double DQN
python dueling_dqn_mini.py  # 运行Dueling DQN
python prioritized_replay_mini.py  # 运行优先经验回放
```

### 结果查看

实验结果将保存在`results/`目录下，包括：
- 学习曲线图像
- 获胜率曲线图像
- CSV格式的详细数据
- 性能对比表格

训练好的模型将保存在`models/`目录下，可用于后续评估或应用。

## 性能调优

如果训练速度过慢，可以尝试以下调整：

1. 在各算法文件中减少`TRAIN_EPISODES`的值
2. 减小网络规模（如减少神经元数量）
3. 增大学习率
4. 减小回放缓冲区大小

## 拓展与自定义

您可以通过以下方式扩展本项目：

1. 添加新的DQN变体
2. 尝试不同的强化学习环境
3. 实现不同的超参数搜索策略
4. 添加可视化工具来更好地展示算法行为

## 参考资料

1. [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
2. [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
3. [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)
4. [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
5. [gymnasium](https://gymnasium.farama.org/environments/toy_text/blackjack/) 