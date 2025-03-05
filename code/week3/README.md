# 第三周：Q-Learning 算法详解与实践

这是一个基于Streamlit的交互式课件，用于讲授Q-Learning算法的基本概念、原理和应用。

## 功能特点

- **模块化设计**：采用模块化结构，便于维护和扩展
- **交互式演示**：通过可视化和交互元素帮助理解Q-Learning算法
- **动态定价案例**：使用真实场景演示Q-Learning算法的应用
- **练习模块**：提供基础和扩展练习，加深对算法的理解

## 安装与运行

### 环境要求

- Python 3.8+
- 所需依赖包列在`requirements.txt`文件中

### 安装步骤

1. 克隆或下载项目代码

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 运行应用
```bash
cd code/week3
streamlit run app.py
```

## 模块说明

- `app.py`：主应用入口，整合所有模块
- `introduction.py`：理论介绍模块
- `q_table_visualization.py`：Q表可视化模块
- `dynamic_pricing.py`：动态定价案例模块
- `exercises.py`：交互式练习模块
- `utils.py`：工具函数模块
- `images/`：图像资源目录

## 使用指南

1. 启动应用后，使用左侧边栏导航不同模块
2. 在理论介绍部分学习Q-Learning的基本概念
3. 使用Q表可视化模块理解Q表的结构和更新过程
4. 在动态定价案例中体验完整的Q-Learning训练过程
5. 完成练习模块中的基础和扩展练习

## 教学建议

- 从基本概念开始，逐步深入算法细节
- 利用可视化工具帮助学生直观理解Q表和更新过程
- 鼓励学生调整参数，观察对算法性能的影响
- 引导学生思考Q-Learning的局限性和可能的改进方向

## 扩展阅读

- Sutton & Barto 的《强化学习导论》第6章
- David Silver 的强化学习课程
- 有关Q-Learning应用的研究论文 