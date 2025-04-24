import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env # 用于创建向量化环境
from stable_baselines3.common.evaluation import evaluate_policy # 用于评估模型
import numpy as np
import os
import time
import matplotlib.pyplot as plt

# 创建日志目录用于TensorBoard
log_dir = "./logs/dqn_cartpole_tensorboard"
os.makedirs(log_dir, exist_ok=True)

# 1. 创建环境 (可以是单个环境，或多个并行环境以加速训练)
# 训练环境
vec_env = make_vec_env("CartPole-v1", n_envs=1) # 创建向量化环境，可以改为更多并行环境

# 2. 定义模型 (选择算法，指定策略网络类型，传入环境)
# "MlpPolicy": 使用多层感知机 (MLP) 作为 Q 网络
model = DQN("MlpPolicy", vec_env, verbose=1, # verbose=1 打印训练信息
            learning_rate=1e-4,
            buffer_size=100000, # 经验回放缓冲区大小
            learning_starts=1000, # 多少步后开始学习
            batch_size=32,
            tau=1.0, # Target network update rate
            gamma=0.99,
            train_freq=4, # 每多少步训练一次
            gradient_steps=1,
            target_update_interval=1000, # Target network 更新频率
            exploration_fraction=0.1, # 探索率衰减的总步数比例
            exploration_final_eps=0.05, # 最终探索率
            tensorboard_log=log_dir, # 添加TensorBoard日志
           )

# 3. 训练模型
print("开始训练模型...")
start_time = time.time()
# total_timesteps: 总的训练步数
model.learn(total_timesteps=100000, log_interval=10) # log_interval 控制打印频率
end_time = time.time()
print(f"训练完成，耗时 {end_time - start_time:.2f} 秒")

# 4. 保存模型
model_path = os.path.join(log_dir, "dqn_cartpole")
model.save(model_path)
print(f"模型已保存到 {model_path}")

# 5. 评估训练好的模型
print("正在评估训练好的模型...")
# 创建一个单独的评估环境
eval_env = gym.make("CartPole-v1")
# n_eval_episodes: 评估多少个回合
# deterministic=True: 使用贪心策略进行评估
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print(f"评估结果: 平均奖励 = {mean_reward:.2f} ± {std_reward:.2f}")

# 6. 可视化训练结果
print("正在可视化训练结果...")
# 创建一个新的环境用于可视化
vis_env = gym.make("CartPole-v1", render_mode="human")
obs, info = vis_env.reset()
terminated = False
truncated = False
total_reward_vis = 0

# 可视化一个回合
while not (terminated or truncated):
    action, _states = model.predict(obs, deterministic=True)  # 使用确定性策略
    obs, reward, terminated, truncated, info = vis_env.step(action)
    total_reward_vis += reward
    vis_env.render()  # 渲染环境
    time.sleep(0.01)  # 减慢渲染速度，便于观察

print(f"可视化完成。总奖励: {total_reward_vis}")

# 关闭所有环境
vec_env.close()
eval_env.close()
vis_env.close()

# 提示如何查看训练曲线
print(f"要查看训练曲线，请在终端运行: tensorboard --logdir {log_dir}")
print("然后在浏览器中打开显示的地址（通常是 http://localhost:6006/）")

# 7. 绘制评估结果图表
# 可选：如果你想展示不同超参数对结果的影响，可以在这里添加代码
# 例如：运行多次不同学习率的实验，并绘制结果对比图
def run_experiment_with_different_lr():
    learning_rates = [1e-5, 1e-4, 1e-3]
    mean_rewards = []
    std_rewards = []
    
    for lr in learning_rates:
        print(f"\n尝试学习率: {lr}")
        model = DQN("MlpPolicy", vec_env, verbose=0,
                    learning_rate=lr,
                    buffer_size=100000,
                    learning_starts=1000,
                    batch_size=32,
                    tau=1.0,
                    gamma=0.99,
                    train_freq=4,
                    gradient_steps=1,
                    target_update_interval=1000,
                    exploration_fraction=0.1,
                    exploration_final_eps=0.05)
        
        model.learn(total_timesteps=50000)  # 减少步数以加快实验
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
        mean_rewards.append(mean_reward)
        std_rewards.append(std_reward)
        print(f"学习率 {lr} 的评估结果: 平均奖励 = {mean_reward:.2f} ± {std_reward:.2f}")
    
    # 绘制结果对比图
    plt.figure(figsize=(10, 6))
    plt.bar([str(lr) for lr in learning_rates], mean_rewards, yerr=std_rewards, capsize=10)
    plt.xlabel('学习率')
    plt.ylabel('平均奖励')
    plt.title('不同学习率下模型的性能对比')
    plt.savefig(os.path.join(log_dir, 'learning_rate_comparison.png'))
    plt.show()

# 取消注释下面一行代码以运行学习率实验（警告：这将额外运行多次训练）
# run_experiment_with_different_lr()