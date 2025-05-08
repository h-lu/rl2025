import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import time
import os

# 创建日志目录
log_dir = "/tmp/gym_a2c/"
os.makedirs(log_dir, exist_ok=True)

# 1. 创建环境 (A2C 通常需要向量化环境)
vec_env = make_vec_env("CartPole-v1", n_envs=8) # A2C 通常使用更多并行环境

# 2. 定义 A2C 模型
# A2C 使用 "MlpPolicy" 或 "CnnPolicy"
# 关键超参数:
# n_steps: 每个环境在更新前运行多少步 (影响 TD 估计的长度)
# vf_coef: 值函数损失的系数 (Critic loss weight)
# ent_coef: 熵正则化系数 (鼓励探索)
model = A2C("MlpPolicy", vec_env, verbose=1,
            gamma=0.99,             # 折扣因子
            n_steps=5,              # 每个环境更新前运行 5 步
            vf_coef=0.5,            # 值函数损失系数
            ent_coef=0.0,           # 熵正则化系数 (CartPole 通常不需要太多探索)
            learning_rate=7e-4,     # 学习率 (A2C 通常用稍高一点的学习率)
            tensorboard_log=log_dir
           )

# 3. 训练模型
print("Starting A2C training on CartPole...")
start_time = time.time()
model.learn(total_timesteps=100000, log_interval=50) # 训练步数与 DQN 保持一致
end_time = time.time()
print(f"Training finished in {end_time - start_time:.2f} seconds.")

# 4. 保存模型
model_path = os.path.join(log_dir, "a2c_cartpole_sb3")
model.save(model_path)
print(f"Model saved to {model_path}.zip")

# 5. 评估训练好的模型
print("Evaluating trained A2C model...")
eval_env = gym.make("CartPole-v1")
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20, deterministic=True)
print(f"Evaluation results (A2C): Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")

vec_env.close()
eval_env.close()

print(f"To view training logs, run: tensorboard --logdir {log_dir}")

# --- (可选) 运行 A2C on Pendulum-v1 ---
# print("\nStarting A2C training on Pendulum...")
# log_dir_pendulum = "/tmp/gym_a2c_pendulum/"
# os.makedirs(log_dir_pendulum, exist_ok=True)
# vec_env_pendulum = make_vec_env("Pendulum-v1", n_envs=8)
# model_pendulum = A2C("MlpPolicy", vec_env_pendulum, verbose=1,
#                      gamma=0.99,
#                      n_steps=5,
#                      vf_coef=0.5,
#                      ent_coef=0.0, # Pendulum 可能需要一点熵正则化
#                      learning_rate=7e-4,
#                      tensorboard_log=log_dir_pendulum
#                     )
# start_time = time.time()
# model_pendulum.learn(total_timesteps=200000, log_interval=50) # Pendulum 可能需要更多步数
# end_time = time.time()
# print(f"Pendulum training finished in {end_time - start_time:.2f} seconds.")
# model_path_pendulum = os.path.join(log_dir_pendulum, "a2c_pendulum_sb3")
# model_pendulum.save(model_path_pendulum)
# print(f"Pendulum model saved to {model_path_pendulum}.zip")

# print("Evaluating trained A2C model on Pendulum...")
# eval_env_pendulum = gym.make("Pendulum-v1")
# mean_reward_p, std_reward_p = evaluate_policy(model_pendulum, eval_env_pendulum, n_eval_episodes=10, deterministic=True)
# print(f"Evaluation results (A2C on Pendulum): Mean reward = {mean_reward_p:.2f} +/- {std_reward_p:.2f}")
# vec_env_pendulum.close()
# eval_env_pendulum.close()
# print(f"To view Pendulum training logs, run: tensorboard --logdir {log_dir_pendulum}")