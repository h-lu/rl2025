"""
复杂迷宫环境示例 - 展示不同动画速度设置
"""

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import time
from complex_maze_env import ComplexMazeEnv

def run_random_policy(env, episodes=1, max_steps=100):
    """运行随机策略并记录总奖励"""
    for episode in range(episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False
        truncated = False
        step_count = 0
        
        start_time = time.time()
        
        while not (done or truncated) and step_count < max_steps:
            action = env.action_space.sample()  # 随机选择动作
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
        
        end_time = time.time()
        print(f"Episode {episode+1}: 总步数={step_count}, 总奖励={total_reward:.2f}, 用时={end_time-start_time:.2f}秒")
    
    env.close()

def main():
    print("示例1: 默认渲染设置")
    env1 = ComplexMazeEnv(render_mode="human", size=15, moving_traps=True, fog_of_war=True)
    run_random_policy(env1, episodes=1, max_steps=200)
    
    print("\n示例2: 使用动画速度参数")
    env2 = ComplexMazeEnv(render_mode="human", size=15, moving_traps=True, fog_of_war=True, 
                         animation_speed=5.0)
    run_random_policy(env2, episodes=1, max_steps=200)
    
    print("\n示例3: 降低渲染频率")
    env3 = ComplexMazeEnv(render_mode="human", size=15, moving_traps=True, fog_of_war=True, 
                         animation_speed=5.0, render_every=10)  # 每10步渲染一次
    run_random_policy(env3, episodes=1, max_steps=200)
    
    print("\n示例4: 快速模式（无动画）")
    env4 = ComplexMazeEnv(render_mode="human", size=15, moving_traps=True, fog_of_war=True, 
                         fast_mode=True)  # 完全跳过渲染
    run_random_policy(env4, episodes=1, max_steps=200)

if __name__ == "__main__":
    main() 