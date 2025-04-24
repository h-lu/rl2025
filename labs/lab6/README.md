# DQN CartPole 训练和可视化指南

本项目提供了使用深度Q网络（DQN）算法训练CartPole环境的示例代码，并实现了训练结果的可视化与评估。

## 环境要求

在运行代码之前，请确保已安装以下包：
```bash
pip install "stable-baselines3[extra]" gymnasium matplotlib numpy torch
```

## 文件说明

- `cartpole_sb3.py`: 主训练脚本，用于训练DQN模型并进行简单的可视化。
- `dqn_visualization.py`: 专门用于可视化和分析已训练模型的工具。

## 运行方法

### 训练和基本可视化
直接运行`cartpole_sb3.py`文件即可：
```bash
python cartpole_sb3.py
```

### 高级可视化和分析
训练完成后，可以使用可视化工具进行更深入的分析：
```bash
python dqn_visualization.py
```
这个脚本提供了以下功能：
1. 可视化模型表现（多回合）
2. 评估模型性能
3. 录制模型表现视频（需要安装ffmpeg）

## 代码功能

训练脚本(`cartpole_sb3.py`)主要包含以下功能：

1. **环境创建**：创建一个CartPole-v1环境用于训练。
2. **模型定义与训练**：使用DQN算法训练模型，并保存训练日志。
3. **模型评估**：评估训练好的模型在多个回合中的平均表现。
4. **可视化训练结果**：以图形界面展示训练后的模型控制CartPole的效果。
5. **TensorBoard监控**：记录训练过程中的各项指标，可通过TensorBoard查看。

可视化工具(`dqn_visualization.py`)主要功能：

1. **多回合可视化**：观察模型在多个回合中的表现。
2. **性能评估**：统计模型在多个回合中的平均表现。
3. **视频录制**：将模型的表现保存为视频文件。
4. **模型比较**：比较不同超参数训练出的模型之间的性能差异。

## 查看训练曲线

运行完成后，可以通过TensorBoard查看训练曲线：
```bash
tensorboard --logdir ./logs/dqn_cartpole_tensorboard
```
然后在浏览器中打开显示的地址（通常是`http://localhost:6006/`）。

## 超参数实验

训练脚本末尾提供了不同学习率对训练结果影响的实验函数`run_experiment_with_different_lr()`。如果想运行此实验，请取消相应注释行。该实验将：

1. 尝试三种不同的学习率：1e-5, 1e-4, 1e-3
2. 对每种学习率训练一个模型
3. 评估每个模型的性能
4. 绘制对比图表并保存

## 修改建议

您可以尝试修改以下参数来观察其对训练效果的影响：

1. **学习率(learning_rate)**：控制每次更新的步长大小。
2. **缓冲区大小(buffer_size)**：经验回放缓冲区的容量。
3. **批量大小(batch_size)**：每次训练使用的样本数量。
4. **探索参数(exploration_fraction, exploration_final_eps)**：控制探索与利用的平衡。
5. **折扣因子(gamma)**：控制未来奖励的重要性。

## 预期结果

成功训练后的模型应该能够在CartPole环境中取得接近500的平均奖励（CartPole-v1的最大步数）。可视化结果应展示小车能够长时间保持平衡杆直立。

## 理解DQN关键组件

1. **经验回放(Experience Replay)**：打破样本之间的相关性，提高训练稳定性。
2. **目标网络(Target Network)**：减少训练目标的波动，提高训练稳定性。
3. **探索策略(ε-greedy)**：在训练初期鼓励探索，后期逐渐转向利用已学到的知识。

## 常见问题解答

1. **训练过程中奖励不增长？**
   - 尝试增加训练步数（`total_timesteps`）
   - 尝试调整学习率
   - 检查探索参数设置

2. **可视化时模型表现不佳？**
   - 确保使用了正确的模型路径
   - 检查模型是否完成了足够的训练步数
   - 观察TensorBoard中的奖励曲线是否收敛

3. **TensorBoard无法显示？**
   - 确保已安装TensorBoard：`pip install tensorboard`
   - 确保使用了正确的日志目录路径 