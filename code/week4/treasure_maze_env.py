"""
迷宫寻宝环境 - 用于演示Q-Learning优化技巧
基于Gymnasium框架开发的自定义环境
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

class TreasureMazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, render_mode=None, size=5, dense_reward=False, treasure_reward=10):
        super().__init__()
        
        # 环境参数
        self.size = size  # 迷宫大小 (size x size)
        self.dense_reward = dense_reward  # 是否使用密集奖励
        self.treasure_reward = treasure_reward  # 找到宝藏的奖励值
        
        # 动作空间: 0=上, 1=右, 2=下, 3=左
        self.action_space = spaces.Discrete(4)
        
        # 观察空间: 智能体在迷宫中的位置 (row, col)
        self.observation_space = spaces.Box(low=0, high=size-1, shape=(2,), dtype=np.int32)
        
        # 渲染设置
        self.render_mode = render_mode
        self.window = None
        self.fig = None
        self.ax = None
        
        # 初始化迷宫
        self._generate_maze()
    
    def _generate_maze(self):
        """生成迷宫，包括墙壁、陷阱和宝藏"""
        # 初始化迷宫 (0=空地, 1=墙, 2=陷阱, 3=宝藏)
        self.maze = np.zeros((self.size, self.size), dtype=np.int32)
        
        # 设置外墙
        self.maze[0, :] = 1
        self.maze[self.size-1, :] = 1
        self.maze[:, 0] = 1
        self.maze[:, self.size-1] = 1
        
        # 创建一些内部墙壁（随机）
        rng = np.random.RandomState(42)  # 固定随机种子，确保迷宫一致
        for _ in range(self.size):
            r, c = rng.randint(1, self.size-1), rng.randint(1, self.size-1)
            if (r, c) != (1, 1):  # 避免在起点放置墙壁
                self.maze[r, c] = 1
        
        # 放置陷阱
        for _ in range(self.size-2):
            r, c = rng.randint(1, self.size-1), rng.randint(1, self.size-1)
            if self.maze[r, c] == 0 and (r, c) != (1, 1):
                self.maze[r, c] = 2
        
        # 放置宝藏在迷宫的某个角落
        corners = [(self.size-2, self.size-2), (self.size-2, 1), (1, self.size-2)]
        treasure_pos = corners[rng.randint(0, len(corners))]
        self.maze[treasure_pos] = 3
        self.treasure_pos = treasure_pos
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 智能体从左上角开始
        self.agent_pos = np.array([1, 1])
        
        # 重置步数
        self.steps = 0
        
        # 计算到宝藏的曼哈顿距离
        self.distance_to_treasure = abs(self.agent_pos[0] - self.treasure_pos[0]) + abs(self.agent_pos[1] - self.treasure_pos[1])
        
        if self.render_mode == "human":
            self._render_frame()
        
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        # 增加步数
        self.steps += 1
        
        # 根据动作移动智能体
        old_pos = self.agent_pos.copy()
        if action == 0:  # 上
            self.agent_pos[0] = max(1, self.agent_pos[0] - 1)
        elif action == 1:  # 右
            self.agent_pos[1] = min(self.size - 2, self.agent_pos[1] + 1)
        elif action == 2:  # 下
            self.agent_pos[0] = min(self.size - 2, self.agent_pos[0] + 1)
        elif action == 3:  # 左
            self.agent_pos[1] = max(1, self.agent_pos[1] - 1)
        
        # 如果撞墙，回到原位
        if self.maze[tuple(self.agent_pos)] == 1:
            self.agent_pos = old_pos
        
        # 计算新的到宝藏的距离
        new_distance = abs(self.agent_pos[0] - self.treasure_pos[0]) + abs(self.agent_pos[1] - self.treasure_pos[1])
        
        # 设置奖励
        reward = 0
        terminated = False
        
        # 检查是否找到宝藏
        if self.maze[tuple(self.agent_pos)] == 3:
            reward = self.treasure_reward
            terminated = True
        # 检查是否掉入陷阱
        elif self.maze[tuple(self.agent_pos)] == 2:
            reward = -self.treasure_reward
            terminated = True
        # 密集奖励模式
        elif self.dense_reward:
            # 每一步的小惩罚
            reward -= 0.1
            
            # 接近宝藏的奖励
            if new_distance < self.distance_to_treasure:
                reward += 0.5
            elif new_distance > self.distance_to_treasure:
                reward -= 0.2
        # 稀疏奖励模式
        else:
            reward = -0.1  # 每一步只有很小的负奖励
        
        # 更新距离
        self.distance_to_treasure = new_distance
        
        # 最大步数限制
        truncated = self.steps >= self.size * self.size * 2
        
        if self.render_mode == "human":
            self._render_frame()
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def _get_obs(self):
        """返回观察结果（智能体的位置）"""
        return self.agent_pos.copy()
    
    def _get_info(self):
        """返回额外信息"""
        return {
            "distance": self.distance_to_treasure,
            "steps": self.steps
        }
    
    def render(self):
        """渲染当前状态"""
        if self.render_mode == "rgb_array":
            return self._render_frame()
    
    def _render_frame(self):
        """渲染单帧画面"""
        if self.fig is None:
            plt.ion()  # 交互模式
            self.fig, self.ax = plt.subplots(figsize=(7, 7))
            plt.show(block=False)
        
        self.ax.clear()
        
        # 创建迷宫颜色映射
        cmap = mcolors.ListedColormap(['white', 'gray', 'red', 'gold'])
        bounds = [0, 1, 2, 3, 4]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        
        # 绘制迷宫
        self.ax.imshow(self.maze, cmap=cmap, norm=norm)
        
        # 绘制智能体
        self.ax.add_patch(Rectangle((self.agent_pos[1] - 0.5, self.agent_pos[0] - 0.5), 
                                    1, 1, fill=True, color='blue', alpha=0.7))
        
        # 添加网格
        self.ax.grid(True, which='both', color='black', linewidth=1)
        self.ax.set_xticks(np.arange(-0.5, self.size, 1))
        self.ax.set_yticks(np.arange(-0.5, self.size, 1))
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        
        # 标题显示步数和奖励类型（使用英文避免字体问题）
        reward_type = "Dense Reward" if self.dense_reward else "Sparse Reward"
        self.ax.set_title(f"Treasure Maze - Steps: {self.steps}, {reward_type}")
        
        plt.draw()
        plt.pause(0.1)
        
        # 获取图像数据
        self.fig.canvas.draw()
        try:
            # 修复图像数据重塑问题
            w, h = self.fig.canvas.get_width_height()
            data = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            expected_size = w * h * 3
            
            # 确保数组大小匹配预期尺寸
            if len(data) != expected_size:
                # 使用替代方法获取图像数据
                buf = self.fig.canvas.buffer_rgba()
                image_data = np.asarray(buf)
                return image_data
            
            data = data.reshape((h, w, 3))
            return data
            
        except ValueError as e:
            print(f"渲染警告: {e}")
            # 返回一个空白图像作为备选
            return np.zeros((700, 700, 3), dtype=np.uint8)
    
    def close(self):
        """关闭环境"""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None 