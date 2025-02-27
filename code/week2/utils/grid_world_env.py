import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import matplotlib.pyplot as plt

class GridWorldEnv(gym.Env):
    """
    Grid World 环境
    一个简单的迷宫环境，智能体需要从起点到达目标，同时避开障碍物
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5, map_type="default"):
        super().__init__()
        self.size = size  # Grid world 大小
        self.window_size = 512  # PyGame 窗口大小

        # 动作空间：上下左右
        self.action_space = spaces.Discrete(4)
        # 观测空间：智能体位置
        self.observation_space = spaces.Discrete(size * size)

        # 根据地图类型选择不同的网格地图
        self._grid_map = self._create_grid_map(map_type)

        # 初始化位置信息
        self._target_location = None
        self._trap_locations = []
        self._agent_start_location = None
        self._agent_location = None
        self._init_locations()

        # 渲染模式
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        # 初始化渲染
        if render_mode is not None:
            self._init_render()
    
    def _create_grid_map(self, map_type):
        """创建指定类型的网格地图"""
        if map_type == "default":
            # 默认的5x5网格地图，修改为目标不被墙完全包围
            return np.array([
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 2, 0, 0],  # 打开右侧的墙，目标可以到达
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0]
            ])
        elif map_type == "maze":
            # 迷宫地图
            return np.array([
                [0, 1, 0, 0, 0],
                [0, 1, 0, 1, 0],
                [0, 0, 0, 1, 0],
                [1, 1, 0, 1, 0],
                [2, 0, 0, 0, 0]
            ])
        elif map_type == "traps":
            # 含有陷阱的地图，修改确保目标可达
            return np.array([
                [0, 0, 0, 0, 0],
                [0, 1, 3, 1, 0],
                [0, 3, 2, 0, 0],  # 打开右侧的墙，目标可以到达
                [0, 1, 3, 1, 0],
                [0, 0, 0, 0, 0]
            ])
        else:
            # 默认地图，修改确保目标可达
            return np.array([
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 2, 0, 0],  # 打开右侧的墙，目标可以到达
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0]
            ])
    
    def _init_locations(self):
        """初始化位置信息"""
        # 查找目标位置
        target_positions = np.where(self._grid_map == 2)
        if len(target_positions[0]) > 0:
            self._target_location = np.array([target_positions[0][0], target_positions[1][0]])
        else:
            # 如果没有设置目标，默认为右下角
            self._target_location = np.array([self.size - 1, self.size - 1])
        
        # 查找陷阱位置
        trap_positions = np.where(self._grid_map == 3)
        for i in range(len(trap_positions[0])):
            self._trap_locations.append(np.array([trap_positions[0][i], trap_positions[1][i]]))
        
        # 设置智能体初始位置
        self._agent_start_location = np.array([0, 0])
        self._agent_location = np.copy(self._agent_start_location)
    
    def _init_render(self):
        """初始化渲染环境"""
        try:
            import pygame
            pygame.init()
            pygame.font.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Grid World")
            self.clock = pygame.time.Clock()
        except ImportError:
            self.render_mode = None
            print("警告: PyGame 未安装，无法以人类可读方式渲染。")
    
    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        
        # 重置智能体位置
        self._agent_location = np.copy(self._agent_start_location)
        
        # 将智能体位置转换为离散观测空间的索引
        observation = self._get_obs()
        
        # 如果需要渲染，渲染当前帧
        if self.render_mode == "human":
            self.render()
        
        return observation, {}
    
    def step(self, action):
        """
        执行一步动作，返回下一个状态、奖励、是否终止等信息
        
        参数:
            action: 动作，0=上, 1=右, 2=下, 3=左
            
        返回:
            observation: 新的状态
            reward: 获得的奖励
            terminated: 是否到达终止状态（目标或陷阱）
            truncated: 是否达到最大步数
            info: 额外信息
        """
        # 动作映射到方向变化 (行,列)
        direction = {
            0: (-1, 0),  # 上
            1: (0, 1),   # 右
            2: (1, 0),   # 下
            3: (0, -1)   # 左
        }
        
        # 计算新位置
        delta_row, delta_col = direction[action]
        new_position = self._agent_location + np.array([delta_row, delta_col])
        
        # 检查是否越界或撞墙
        if (
            0 <= new_position[0] < self.size 
            and 0 <= new_position[1] < self.size 
            and self._grid_map[new_position[0], new_position[1]] != 1
        ):
            self._agent_location = new_position
        
        # 获取当前位置的单元格类型
        current_cell = self._grid_map[self._agent_location[0], self._agent_location[1]]
        
        # 初始化奖励和终止状态
        reward = -0.1  # 每一步的小惩罚，鼓励快速到达目标
        terminated = False
        truncated = False
        
        # 根据当前位置计算奖励和是否终止
        if np.array_equal(self._agent_location, self._target_location):
            # 到达目标
            reward = 1.0
            terminated = True
        elif current_cell == 3 or any(np.array_equal(self._agent_location, trap) for trap in self._trap_locations):
            # 掉入陷阱
            reward = -1.0
            terminated = True
        
        # 获取观测
        observation = self._get_obs()
        
        # 如果需要渲染，渲染当前帧
        if self.render_mode == "human":
            self.render()
        
        return observation, reward, terminated, truncated, {}
    
    def _get_obs(self):
        """获取观测"""
        # 将智能体位置转换为离散观测空间的索引
        return self._agent_location[0] * self.size + self._agent_location[1]
    
    def render(self):
        """渲染环境"""
        if self.render_mode is None:
            return
        
        if self.render_mode == "rgb_array":
            return self._render_frame()
        
        if self.window is None:
            self._init_render()
        
        # 渲染帧
        canvas = self._render_frame()
        
        # 转换为PyGame表面
        pygame_surface = pygame.surfarray.make_surface(canvas)
        self.window.blit(pygame_surface, (0, 0))
        pygame.event.pump()
        pygame.display.update()
        
        # 控制渲染帧率
        self.clock.tick(self.metadata["render_fps"])
    
    def _render_frame(self):
        """渲染单帧"""
        # 初始化画布
        canvas = np.zeros((self.window_size, self.window_size, 3), dtype=np.uint8)
        
        # 计算每个网格单元的大小
        pix_square_size = self.window_size // self.size
        
        # 绘制网格
        for i in range(self.size):
            for j in range(self.size):
                cell_type = self._grid_map[i, j]
                rect = pygame.Rect(
                    j * pix_square_size,
                    i * pix_square_size,
                    pix_square_size,
                    pix_square_size
                )
                
                # 根据单元格类型设置颜色
                if cell_type == 0:  # 空白单元格
                    color = (255, 255, 255)  # 白色
                elif cell_type == 1:  # 墙壁
                    color = (0, 0, 0)  # 黑色
                elif cell_type == 2:  # 目标
                    color = (255, 215, 0)  # 金色
                elif cell_type == 3:  # 陷阱
                    color = (255, 0, 0)  # 红色
                
                # 填充矩形
                pygame.draw.rect(
                    self.window,
                    color,
                    rect,
                )
        
        # 绘制网格线
        for i in range(self.size + 1):
            # 横线
            pygame.draw.line(
                self.window,
                (0, 0, 0),
                (0, i * pix_square_size),
                (self.window_size, i * pix_square_size),
                width=2,
            )
            # 竖线
            pygame.draw.line(
                self.window,
                (0, 0, 0),
                (i * pix_square_size, 0),
                (i * pix_square_size, self.window_size),
                width=2,
            )
        
        # 绘制智能体
        pygame.draw.circle(
            self.window,
            (0, 0, 255),  # 蓝色
            (self._agent_location[1] * pix_square_size + pix_square_size // 2,
             self._agent_location[0] * pix_square_size + pix_square_size // 2),
            pix_square_size // 3,
        )
        
        # 获取渲染结果
        if self.render_mode == "rgb_array":
            pygame.pixelcopy.surface_to_array(canvas, self.window)
            return canvas
    
    def close(self):
        """关闭环境"""
        if self.window is not None:
            pygame.quit()
            self.window = None

def create_random_grid_world(size=5, obstacle_prob=0.3):
    """
    创建随机的Grid World环境
    
    参数:
        size: 网格大小
        obstacle_prob: 障碍物出现的概率
    
    返回:
        随机生成的网格地图
    """
    # 初始化网格地图
    grid_map = np.zeros((size, size), dtype=int)
    
    # 随机放置障碍物
    for i in range(size):
        for j in range(size):
            # 左上角和右下角不放置障碍物
            if (i == 0 and j == 0) or (i == size-1 and j == size-1):
                continue
            
            # 按照概率放置障碍物
            if np.random.random() < obstacle_prob:
                grid_map[i, j] = 1
    
    # 放置目标
    grid_map[size-1, size-1] = 2
    
    return grid_map 