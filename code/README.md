# 俄罗斯方块DQN强化学习项目

## 项目结构

```
code/
├── tetris_env.py    # 俄罗斯方块游戏环境
├── tetris_dqn.py    # DQN算法实现
└── dqn_cartpole.py  # CartPole示例代码(参考)
```

## 环境要求

- Python 3.7+
- PyTorch
- Gymnasium
- Pygame
- NumPy

安装依赖：
```bash
pip install torch gymnasium pygame numpy
```

## 使用方法

### 1. 人机交互模式
```bash
# 在项目根目录下执行(确保PYTHONPATH包含code目录)
python -c "import sys; sys.path.append('./code'); from tetris_env import TetrisEnv; env = TetrisEnv(render_mode='human'); env.play_human()"
```

控制按键：
- ← → : 左右移动
- ↑ : 旋转方块
- ↓ : 加速下落
- 空格: 直接落到底部

### 2. 训练DQN智能体
```bash
python tetris_dqn.py
```

训练参数可在文件中调整：
- EPISODES: 训练回合数
- BATCH_SIZE: 批大小
- GAMMA: 折扣因子
- EPS_START/EPS_END: 探索率范围

### 3. 使用训练好的模型
```python
from tetris_dqn import DQNAgent
agent = DQNAgent()
agent.load_model('tetris_dqn.pth')  # 加载预训练模型
```

## 实现细节

- 使用PyTorch实现深度Q网络
- 包含经验回放(Experience Replay)
- 使用目标网络(Target Network)稳定训练
- 自定义奖励函数考虑:
  - 消除行数
  - 空洞数量
  - 高度差异

## 扩展建议

1. 尝试实现Double DQN、Dueling DQN等改进算法
2. 调整网络结构或超参数优化性能
3. 添加更多游戏状态特征
4. 实现模型保存和加载功能