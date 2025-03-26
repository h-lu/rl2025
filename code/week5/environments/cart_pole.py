"""
CartPole环境封装类

封装了OpenAI Gym的CartPole环境，提供统一的接口和可视化功能
"""

import gymnasium as gym
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from matplotlib import animation
import base64
from IPython.display import HTML

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

class CartPoleEnv:
    """
    CartPole环境封装类
    
    封装了gym环境，提供统一的接口和额外功能
    """
    
    def __init__(self, env_name=config.ENV_NAME, render_mode=config.RENDER_MODE, seed=config.SEED):
        """
        初始化CartPole环境
        
        参数:
            env_name (str): 环境名称
            render_mode (str): 渲染模式，可以是None, 'human', 'rgb_array'
            seed (int): 随机种子
        """
        self.env = gym.make(env_name, render_mode=render_mode)
        self.env.reset(seed=seed)
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.seed = seed
        self.episode_frames = []  # 用于存储回合的帧，以便后续生成动画
        
    def reset(self):
        """
        重置环境
        
        返回:
            np.array: 初始状态
        """
        state, _ = self.env.reset(seed=self.seed)
        self.episode_frames = []
        if self.env.render_mode == 'rgb_array':
            self.episode_frames.append(self.env.render())
        return state
    
    def step(self, action):
        """
        执行动作
        
        参数:
            action (int): 动作
            
        返回:
            tuple: (next_state, reward, done, info)
        """
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        # 如果设置了渲染模式为rgb_array，则保存帧
        if self.env.render_mode == 'rgb_array' and not done:
            self.episode_frames.append(self.env.render())
            
        return next_state, reward, done, info
    
    def close(self):
        """
        关闭环境
        """
        self.env.close()
    
    def render(self):
        """
        渲染环境
        
        返回:
            ndarray: 环境的RGB图像
        """
        return self.env.render()
    
    def get_episode_animation(self, interval=50):
        """
        获取回合的动画
        
        参数:
            interval (int): 帧间隔（毫秒）
            
        返回:
            HTML: 可以在Jupyter Notebook中显示的HTML对象
        """
        if not self.episode_frames:
            return None
        
        plt.figure(figsize=(8, 6))
        patch = plt.imshow(self.episode_frames[0])
        plt.axis('off')
        
        def animate(i):
            patch.set_data(self.episode_frames[i])
            return [patch]
        
        anim = animation.FuncAnimation(
            plt.gcf(), animate, frames=len(self.episode_frames),
            interval=interval, blit=True
        )
        
        # 保存为HTML5视频
        html = anim.to_html5_video()
        plt.close()
        
        return HTML(html)
    
    def save_episode_animation(self, filename, fps=30):
        """
        保存回合动画为GIF文件
        
        参数:
            filename (str): 文件名
            fps (int): 每秒帧数
        """
        if not self.episode_frames:
            return
        
        # 使用matplotlib创建动画
        fig = plt.figure(figsize=(8, 6))
        patch = plt.imshow(self.episode_frames[0])
        plt.axis('off')
        
        def animate(i):
            patch.set_data(self.episode_frames[i])
            return [patch]
        
        anim = animation.FuncAnimation(
            fig, animate, frames=len(self.episode_frames),
            interval=1000/fps, blit=True
        )
        
        # 保存为GIF
        anim.save(filename, writer='pillow', fps=fps)
        plt.close()
    
    def get_state_description(self):
        """
        获取状态空间的描述
        
        返回:
            dict: 状态空间的描述
        """
        return {
            'Cart Position': '小车在轨道上的位置，范围为[-4.8, 4.8]',
            'Cart Velocity': '小车的速度，范围为[-∞, ∞]',
            'Pole Angle': '杆与垂直方向的夹角，范围为[-0.418, 0.418]弧度（约-24至24度）',
            'Pole Angular Velocity': '杆的角速度，范围为[-∞, ∞]'
        }
    
    def get_action_description(self):
        """
        获取动作空间的描述
        
        返回:
            dict: 动作空间的描述
        """
        return {
            0: '向左推小车',
            1: '向右推小车'
        }
    
    def get_env_info(self):
        """
        获取环境信息
        
        返回:
            dict: 环境信息
        """
        return {
            'name': self.env.unwrapped.spec.id,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'max_episode_steps': self.env.unwrapped.spec.max_episode_steps,
            'reward_threshold': self.env.unwrapped.spec.reward_threshold
        } 