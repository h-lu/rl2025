# DQN算法交互式教程

这是一个基于Streamlit的深度Q网络(DQN)算法交互式教程，旨在帮助用户理解DQN算法的原理和实现。通过这个应用程序，您可以学习DQN的理论基础，查看代码实现，运行实时演示，并进行交互式实验。

## 功能特点

- **理论基础**：详细解释DQN算法的核心概念、数学原理和工作流程
- **代码实现**：展示DQN算法的关键实现部分，包括网络架构、经验回放缓冲区、智能体等
- **算法演示**：可视化DQN算法在CartPole环境中的训练和评估过程
- **交互式实验**：允许用户自定义DQN参数，进行不同配置的对比实验

## 目录结构

```
code/week5/
├── app.py                 # 应用程序入口
├── config.py              # 配置参数
├── agents/                # 智能体实现
│   └── dqn_agent.py       # DQN智能体
├── environments/          # 环境封装
│   └── cart_pole.py       # CartPole环境
├── utils/                 # 工具函数
│   ├── replay_buffer.py   # 经验回放缓冲区
│   └── visualization.py   # 可视化工具
├── pages/                 # 页面
│   ├── theory.py          # 理论基础页面
│   ├── implementation.py  # 代码实现页面
│   ├── demo.py            # 算法演示页面
│   └── interactive.py     # 交互式实验页面
├── components/            # 组件
│   └── sidebar.py         # 侧边栏组件
└── assets/                # 静态资源
    └── cartpole.png       # CartPole环境图片
```

## 运行说明

### 环境要求

- Python 3.8+
- TensorFlow 2.x
- Gymnasium (OpenAI Gym的继任者)
- Streamlit
- NumPy, Matplotlib, Pandas

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行应用

```bash
cd code/week5
streamlit run app.py
```

执行上述命令后，应用程序将在本地启动，并自动在浏览器中打开。

## 使用指南

1. **理论基础**：阅读DQN算法的基本原理、背景和核心概念
2. **代码实现**：查看DQN算法的关键代码实现和解释
3. **算法演示**：观察预训练的DQN模型在CartPole环境中的表现，或者实时训练新模型
4. **交互式实验**：设计实验，比较不同参数配置对DQN性能的影响

## 核心DQN知识点

- Q-learning基础和局限性
- 深度神经网络近似Q函数
- 经验回放机制
- 目标网络稳定训练
- ε-greedy探索策略
- DQN训练流程和评估方法

## 参考资料

- Mnih, V. et al. (2013). Playing Atari with Deep Reinforcement Learning. arXiv:1312.5602.
- Mnih, V. et al. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), 529-533.
- OpenAI Gym/Gymnasium 文档
- TensorFlow 文档
- Streamlit 文档 