# DQN优化技术实现

本项目包含深度Q网络(DQN)的多种优化技术实现，这些技术共同组成了Rainbow DQN算法。每个实现都专注于一个特定的优化方面，并提供了独立的代码和示例。

## 目录结构

- `rainbow_dqn.py`: 集成了所有优化技术的完整Rainbow DQN实现
- `double_dqn.py`: Double DQN实现，解决Q值过估计问题
- `dueling_dqn.py`: Dueling DQN实现，改进网络架构以分离状态价值和动作优势
- `prioritized_replay.py`: 优先经验回放实现，基于TD误差进行样本优先级排序
- `n_step_learning.py`: 多步学习实现，使用n步回报加速价值传播
- `noisy_nets.py`: 噪声网络实现，使用参数化噪声进行自适应探索

## 优化技术简介

### Double DQN
解决DQN中的Q值过高估计问题，通过分离动作选择和评估来减少估计偏差。具体来说，使用主网络选择动作，但使用目标网络来评估所选动作的价值。

**参考文献**: [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)

### Dueling DQN
改进Q网络的架构，将Q函数分解为状态价值函数V(s)和动作优势函数A(s,a)。这种设计可以更好地学习状态价值，提高学习的效率和稳定性。

**参考文献**: [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)

### 优先经验回放 (Prioritized Experience Replay)
改进经验回放机制，根据TD误差为样本分配优先级，使得具有更大TD误差的样本被更频繁地采样，从而提高学习效率。同时，使用重要性采样权重来修正引入的偏差。

**参考文献**: [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)

### 多步学习 (N-step Learning)
通过计算n步回报来改进值函数估计，而不是仅使用一步回报。通过在计算TD目标时使用多个时间步的累积奖励，可以加速价值传播并减少估计偏差。

**参考文献**: [Rainbow论文中的组件](https://arxiv.org/abs/1602.01783)

### 噪声网络 (Noisy Networks)
通过向网络层的权重和偏置添加参数化噪声来实现探索，替代传统的epsilon-greedy探索策略。这种方法使得探索行为可以根据环境状态自适应地调整。

**参考文献**: [Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295)

### Rainbow DQN
将上述所有优化技术结合在一起，形成一个更强大的DQN变体。Rainbow DQN在多种Atari游戏上显著提高了性能。

**参考文献**: [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298)

## 使用方法

每个Python文件都可以独立运行，包含了完整的训练和评估代码。默认使用CartPole-v1环境进行测试。

```bash
# 运行Rainbow DQN (包含所有优化)
python rainbow_dqn.py

# 运行特定优化技术的实现
python double_dqn.py
python dueling_dqn.py
python prioritized_replay.py
python n_step_learning.py
python noisy_nets.py
```

## 环境要求

- Python 3.7+
- TensorFlow 2.0+
- Gymnasium
- NumPy
- Matplotlib

可以通过以下命令安装所需依赖：

```bash
pip install tensorflow gymnasium numpy matplotlib
```

## 性能比较

每个实现文件都包含了与标准DQN进行比较的代码（被注释掉），可以取消注释运行查看性能差异。此外，`rainbow_dqn.py`文件允许通过设置参数来选择性地启用或禁用各种优化技术。 