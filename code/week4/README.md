# Q-Learning优化技巧演示

本目录包含用于演示Q-Learning算法优化技巧的代码，包括探索策略、Q-Table初始化和奖励函数设计等。

## 文件说明

### 基础环境
- `treasure_maze_env.py`: 基于Gymnasium框架的简单迷宫寻宝环境
- `q_learning_demo.py`: Q-Learning算法实现和各种优化技巧
- `demo.py`: 简单的演示脚本，用于展示如何使用环境和算法

### 复杂环境（更好地展示优化技巧差异）
- `complex_maze_env.py`: 更复杂的迷宫环境，包含移动陷阱、时间惩罚和战争迷雾
- `complex_demo.py`: 用于在复杂环境中演示不同优化技巧的效果差异

## 环境说明

### 简单迷宫环境
简单迷宫寻宝环境是一个基础的网格世界，包含以下元素：
- 空地：智能体可以自由移动
- 墙壁：智能体不能穿过
- 陷阱：掉入会得到负奖励，游戏结束
- 宝藏：找到会得到正奖励，游戏结束

### 复杂迷宫环境
复杂迷宫环境增加了以下特性，使不同优化技巧的差异更加明显：
- **更大的迷宫**：10x10而不是5x5，状态空间更大
- **移动陷阱**：陷阱会随机移动，增加环境的不确定性
- **时间惩罚**：随着步数增加，惩罚会越来越大，鼓励智能体快速找到宝藏
- **战争迷雾**：智能体只能看到周围3x3区域，增加部分可观察性挑战

## Q-Learning优化技巧

本代码演示了以下Q-Learning优化技巧：

### 1. 探索策略

- **ε-greedy策略**：以ε概率随机探索，以1-ε概率选择最优动作
- **ε-greedy退火策略**：训练过程中逐渐减小ε值，从而减少探索增加利用
- **Softmax策略**：根据Q值的概率分布选择动作，而不是简单的随机选择

### 2. Q-Table初始化

- **零值初始化**：将Q表所有值初始化为0
- **随机值初始化**：将Q表初始化为随机值，增加初始探索
- **乐观初始化**：将Q表初始化为较大的正值，鼓励探索未知状态-动作对

### 3. 奖励函数设计

- **稀疏奖励**：只在达到目标或失败时给予奖励
- **密集奖励**：在中间过程中也给予奖励，如接近目标时给予小奖励

## 使用方法

### 简单环境演示

1. 确保已安装必要的依赖：
   ```
   pip install numpy matplotlib gymnasium
   ```

2. 运行演示脚本：
   ```
   python demo.py
   ```

3. 根据提示选择要演示的内容：
   - 单个智能体学习过程
   - 比较不同探索策略
   - 比较不同Q表初始化策略
   - 比较不同奖励函数

### 复杂环境演示

1. 运行复杂环境演示脚本：
   ```
   python complex_demo.py
   ```

2. 根据提示选择要演示的内容：
   - 单个智能体在复杂环境中的学习过程
   - 比较不同探索策略在复杂环境中的效果
   - 比较不同Q表初始化策略在复杂环境中的效果
   - 比较不同奖励函数在复杂环境中的效果

## 为什么需要复杂环境？

在简单环境中，不同优化技巧的效果差异可能不够明显，原因包括：

1. **状态空间小**：简单迷宫的状态空间较小，Q表能够完整覆盖所有状态
2. **问题结构简单**：从起点到终点路径较短，决策分支少
3. **奖励区分度低**：即使是稀疏奖励，也容易通过随机探索找到目标

复杂环境通过增加迷宫大小、移动陷阱、时间惩罚和战争迷雾等特性，使得不同优化技巧的效果差异更加明显，更好地展示了Q-Learning算法的优化技巧在实际应用中的重要性。

## 自定义环境参数

### 简单环境参数
```python
env = TreasureMazeEnv(
    render_mode="human",  # 渲染模式：human或rgb_array
    size=7,               # 迷宫大小
    dense_reward=True,    # 是否使用密集奖励
    treasure_reward=10    # 找到宝藏的奖励值
)
```

### 复杂环境参数
```python
env = ComplexMazeEnv(
    render_mode="human",    # 渲染模式：human或rgb_array
    size=10,                # 迷宫大小
    dense_reward=True,      # 是否使用密集奖励
    treasure_reward=10,     # 找到宝藏的奖励值
    moving_traps=True,      # 是否有移动陷阱
    time_penalty=True,      # 是否有时间惩罚
    fog_of_war=True         # 是否有战争迷雾（有限视野）
)
```

## 自定义智能体参数

```python
agent = QLearningAgent(
    env=env,
    learning_rate=0.1,                  # 学习率
    discount_factor=0.99,               # 折扣因子
    exploration_strategy="epsilon_decay", # 探索策略
    epsilon=0.3,                        # 初始探索概率
    epsilon_decay=0.99,                 # epsilon衰减率
    epsilon_min=0.01,                   # epsilon最小值
    q_init_strategy="optimistic",       # Q表初始化策略
    q_init_value=5.0                    # 初始化值
)
``` 