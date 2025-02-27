# 第二周：强化学习框架与迷宫环境 - 交互式课件

这是一个使用Streamlit构建的交互式课件，用于展示第二周的强化学习内容，包括马尔可夫决策过程(MDP)、价值函数、探索与利用平衡，以及Grid World环境的实现。

## 功能特点

- **交互式内容展示**：通过可视化图表和交互式控件展示强化学习概念
- **实时演示**：直观演示Grid World环境和智能体行为
- **动态参数调整**：允许调整参数，观察其对学习过程的影响
- **模块化设计**：采用模块化结构，方便扩展和维护
- **练习环节**：包含基础和进阶练习，帮助学生理解和应用所学内容

## 安装要求

本应用需要以下Python包：

```bash
streamlit==1.29.0
numpy==1.24.0
pandas==2.0.0
matplotlib==3.7.0
seaborn==0.12.0
plotly==5.14.0
altair==5.0.0
gymnasium==0.28.0
pygame==2.5.0
```

## 安装方法

1. 确保已安装Python 3.8或更高版本
2. 克隆或下载本仓库
3. 安装依赖包：

```bash
pip install -r requirements.txt
```

## 使用方法

1. 进入code/week2目录：

```bash
cd code/week2
```

2. 运行应用：

```bash
python run.py
```

或者直接使用streamlit：

```bash
streamlit run app.py
```

3. 在浏览器中访问显示的URL（通常是http://localhost:8501）

## 应用结构

```
code/week2/
├── app.py                  # 主应用入口
├── run.py                  # 启动脚本
├── pages/                  # 页面模块
│   ├── introduction.py     # MDP介绍
│   ├── value_functions.py  # 价值函数介绍
│   ├── exploration.py      # 探索与利用的平衡
│   ├── grid_world.py       # Grid World环境介绍
│   ├── basic_exercises.py  # 基础练习
│   └── advanced_exercises.py # 进阶练习
└── utils/                  # 工具函数
    ├── visualizations.py   # 可视化工具
    └── grid_world_env.py   # Grid World环境实现
```

## 页面说明

- **课程介绍**：介绍马尔可夫决策过程(MDP)的基本概念和组成元素
- **价值函数**：讲解状态价值函数和动作价值函数，以及贝尔曼方程
- **探索与利用的平衡**：介绍探索-利用窘境，以及常见策略如$\epsilon$-greedy
- **Grid World环境**：展示Grid World环境的实现和使用方法
- **基础练习**：提供简单的强化学习练习，如实现随机策略和$\epsilon$-greedy策略
- **进阶练习**：提供更深入的练习，如实现Q-learning算法和设计复杂环境

## 适用对象

本交互式课件适用于：
- 强化学习课程的学生
- 对强化学习感兴趣的自学者
- 教师教学演示

## 注意事项

- 运行Q-learning等算法演示时可能需要较长时间，请耐心等待
- 手动控制Grid World环境时，若智能体到达目标或陷阱，需点击"重置"按钮重新开始

## 版权说明

本课件仅用于教学目的，未经许可不得用于商业用途。 