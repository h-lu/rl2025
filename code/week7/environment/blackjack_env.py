import numpy as np
import gymnasium as gym

class BlackjackWrapper(gym.Wrapper):
    """
    对21点环境的包装，将观察空间转换为适合DQN处理的格式
    """
    def __init__(self, env):
        super().__init__(env)
        
    def reset(self, **kwargs):
        """重置环境并返回初始状态"""
        observation, info = self.env.reset(**kwargs)
        return np.array(observation, dtype=np.float32), info
    
    def step(self, action):
        """执行动作并返回下一个状态、奖励等"""
        observation, reward, terminated, truncated, info = self.env.step(action)
        return np.array(observation, dtype=np.float32), reward, terminated, truncated, info
