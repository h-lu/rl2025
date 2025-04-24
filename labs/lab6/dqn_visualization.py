"""
DQN 模型可视化和分析工具

这个脚本用于加载已训练好的DQN模型，进行可视化展示和性能分析。
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

# 默认模型路径
DEFAULT_MODEL_PATH = "./logs/dqn_cartpole_tensorboard/dqn_cartpole.zip"

def visualize_model(model_path=DEFAULT_MODEL_PATH, episodes=3, render_mode="human", sleep_time=0.01):
    """
    可视化模型在环境中的表现
    
    参数:
        model_path: 模型保存路径
        episodes: 可视化的回合数
        render_mode: 渲染模式，可选 "human" 或 "rgb_array"
        sleep_time: 每步之间的暂停时间
    """
    # 加载模型
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 {model_path} 不存在")
        return
    
    model = DQN.load(model_path)
    print(f"成功加载模型: {model_path}")
    
    # 创建环境
    env = gym.make("CartPole-v1", render_mode=render_mode)
    
    for episode in range(episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        steps = 0
        
        print(f"\n开始回合 {episode+1}/{episodes}")
        
        while not (terminated or truncated):
            # 使用模型预测动作
            action, _states = model.predict(obs, deterministic=True)
            
            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # 渲染环境
            env.render()
            time.sleep(sleep_time)
        
        print(f"回合 {episode+1} 完成 - 总奖励: {total_reward}, 步数: {steps}")
    
    env.close()

def evaluate_trained_model(model_path=DEFAULT_MODEL_PATH, eval_episodes=10):
    """
    评估训练好的模型性能
    
    参数:
        model_path: 模型保存路径
        eval_episodes: 评估的回合数
    
    返回:
        mean_reward: 平均奖励
        std_reward: 奖励标准差
    """
    # 加载模型
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 {model_path} 不存在")
        return None, None
    
    model = DQN.load(model_path)
    
    # 创建评估环境
    eval_env = gym.make("CartPole-v1")
    
    # 评估模型
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=eval_episodes, deterministic=True)
    
    print(f"评估结果 ({eval_episodes} 回合):")
    print(f"平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")
    
    eval_env.close()
    return mean_reward, std_reward

def record_episode(model_path=DEFAULT_MODEL_PATH, video_dir="./videos", episode_length=500):
    """
    记录模型表现的视频（需要安装 ffmpeg）
    
    参数:
        model_path: 模型保存路径
        video_dir: 视频保存目录
        episode_length: 最大步数
    """
    try:
        from gymnasium.wrappers import RecordVideo
    except ImportError:
        print("错误: 无法导入 RecordVideo，请确保 gymnasium 已正确安装")
        return
    
    # 创建视频保存目录
    os.makedirs(video_dir, exist_ok=True)
    
    # 加载模型
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 {model_path} 不存在")
        return
    
    model = DQN.load(model_path)
    
    # 创建录制环境
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = RecordVideo(env, video_dir)
    
    obs, info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0
    
    # 运行一个回合
    for step in range(episode_length):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    print(f"视频录制完成，总奖励: {total_reward}")
    print(f"视频保存在: {video_dir}")
    
    env.close()

def compare_models(model_paths, model_names=None, eval_episodes=10):
    """
    比较多个模型的性能
    
    参数:
        model_paths: 模型路径列表
        model_names: 模型名称列表（用于图表展示）
        eval_episodes: 每个模型评估的回合数
    """
    if model_names is None:
        model_names = [f"模型 {i+1}" for i in range(len(model_paths))]
    
    mean_rewards = []
    std_rewards = []
    
    for path, name in zip(model_paths, model_names):
        if not os.path.exists(path):
            print(f"警告: 模型文件 {path} 不存在，跳过")
            continue
        
        print(f"\n评估模型: {name}")
        mean_reward, std_reward = evaluate_trained_model(path, eval_episodes)
        
        if mean_reward is not None:
            mean_rewards.append(mean_reward)
            std_rewards.append(std_reward)
    
    # 绘制比较图
    if mean_rewards:
        plt.figure(figsize=(10, 6))
        plt.bar(model_names, mean_rewards, yerr=std_rewards, capsize=10)
        plt.xlabel("模型")
        plt.ylabel("平均奖励")
        plt.title("模型性能比较")
        plt.savefig("model_comparison.png")
        plt.show()

def main():
    """主函数：提供交互式选项"""
    print("DQN 模型可视化和分析工具")
    print("=" * 40)
    print("请选择要执行的操作:")
    print("1. 可视化模型表现")
    print("2. 评估模型性能")
    print("3. 录制模型表现视频")
    print("4. 退出")
    
    choice = input("\n请输入选项 (1-4): ")
    
    if choice == "1":
        model_path = input(f"请输入模型路径 (默认: {DEFAULT_MODEL_PATH}): ") or DEFAULT_MODEL_PATH
        episodes = int(input("请输入可视化回合数 (默认: 3): ") or 3)
        visualize_model(model_path, episodes)
    elif choice == "2":
        model_path = input(f"请输入模型路径 (默认: {DEFAULT_MODEL_PATH}): ") or DEFAULT_MODEL_PATH
        eval_episodes = int(input("请输入评估回合数 (默认: 10): ") or 10)
        evaluate_trained_model(model_path, eval_episodes)
    elif choice == "3":
        model_path = input(f"请输入模型路径 (默认: {DEFAULT_MODEL_PATH}): ") or DEFAULT_MODEL_PATH
        video_dir = input("请输入视频保存目录 (默认: ./videos): ") or "./videos"
        record_episode(model_path, video_dir)
    elif choice == "4":
        print("退出程序")
        return
    else:
        print("无效选项，请重新运行程序")

if __name__ == "__main__":
    main() 