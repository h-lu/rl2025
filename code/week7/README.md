# DQN及其改进方法交互式课件

本项目是一个基于Streamlit的交互式课件，用于展示DQN（深度Q网络）算法及其改进方法在21点游戏环境中的应用。

## 主要特点

- 学习和比较不同的DQN改进方法（Double DQN, Dueling DQN, 优先经验回放）
- 直观的可视化和交互式演示
- 使用21点（Blackjack）环境作为演示平台
- 模块化设计，易于扩展

## 环境要求

- Python 3.8+
- 相关Python包：见`requirements.txt`文件

## 安装步骤

1. 克隆本项目（或使用已下载的项目文件夹）
2. 安装依赖包：

```bash
cd code/week7
pip install -r requirements.txt
```

## 使用方法

运行Streamlit应用程序：

```bash
cd code/week7
streamlit run app.py
```

浏览器将自动打开应用程序。如果没有，请访问终端中显示的URL（通常是 http://localhost:8501）。

## 项目结构

```
blackjack_dqn_demo/
├── app.py                    # 主Streamlit应用入口
├── pages/                    # 多页面应用的其他页面
│   ├── 1_double_dqn.py       # Double DQN页面
│   ├── 2_dueling_dqn.py      # Dueling DQN页面
│   ├── 3_per_dqn.py          # 优先经验回放页面
│   └── 4_exploration.py      # 探索策略页面
├── models/                   # 模型定义
│   ├── base_dqn.py           # 基础DQN模型
│   ├── double_dqn.py         # Double DQN实现
│   ├── dueling_dqn.py        # Dueling DQN实现
│   └── per_dqn.py            # 优先经验回放DQN
├── utils/                    # 工具函数
│   ├── buffer.py             # 经验回放缓冲区
│   ├── exploration.py        # 探索策略
│   ├── training.py           # 训练函数
│   └── visualization.py      # 可视化函数
└── environment/              # 环境相关代码
    └── blackjack_env.py      # 21点环境包装器
```

## 主要功能

1. **主页**：介绍DQN算法及21点环境，提供基础DQN训练演示
2. **Double DQN页面**：解释Double DQN原理及演示其在解决过高估计问题中的效果
3. **Dueling DQN页面**：展示Dueling架构及其优势
4. **优先经验回放页面**：展示优先经验回放的工作原理和效果
5. **探索策略页面**：比较不同的探索策略（ε-greedy, Boltzmann, 噪声网络）

## 注意事项

- 训练过程可能需要一些时间，特别是在比较不同算法时
- 为了更好的体验，建议使用现代浏览器并确保有足够的系统资源 