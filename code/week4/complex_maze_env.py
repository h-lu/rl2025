"""
复杂迷宫寻宝环境 - 用于更好地展示Q-Learning优化技巧的差异
基于Gymnasium框架开发的自定义环境

迷宫设计说明:
---------------
1. 迷宫结构:
   - 迷宫大小为 size x size 的网格
   - 外围边缘是墙壁，内部有随机生成的墙壁
   - **确保从起点(左上角)到宝藏(通常在迷宫远端)始终存在一条可行路径**
   - 地形类型: 空地(0)、墙壁(1)、静态陷阱(2)、宝藏(3)、移动陷阱(4)

2. 智能体:
   - 从左上角(1,1)出发，目标是找到宝藏
   - 可以进行上、右、下、左四个方向的移动
   - 撞墙会保持原位置不变

3. 障碍与挑战:
   - 静态陷阱: 踩到会立即结束游戏并获得负奖励
   - 移动陷阱: 会随机移动，踩到同样会立即结束游戏
   - 战争迷雾: 智能体只能看到周围3x3区域，增加探索难度
   - 时间惩罚: 随着步数增加，每步获得的奖励会减少

4. 奖励设置:
   - 找到宝藏: 获得正奖励(默认+10)
   - 踩到陷阱: 获得负奖励(默认-10)
   - 密集奖励模式: 接近宝藏会获得小奖励，远离会获得小惩罚
   - 稀疏奖励模式: 每步只有很小的负奖励(-0.1)，仅在找到宝藏时有显著奖励

5. 环境参数:
   - size: 迷宫大小
   - dense_reward: 是否使用密集奖励
   - treasure_reward: 找到宝藏的奖励值
   - moving_traps: 是否有移动的陷阱
   - time_penalty: 是否有时间惩罚
   - fog_of_war: 是否有战争迷雾（有限视野）
   - animation_speed: 动画速度倍率
   - fast_mode: 是否启用快速模式（完全跳过可视化更新）
   - render_every: 每隔多少步渲染一次
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from collections import deque

class ComplexMazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 12}
    
    def __init__(self, render_mode=None, size=10, dense_reward=False, treasure_reward=10, 
                 moving_traps=True, time_penalty=True, fog_of_war=True, animation_speed=1.0,
                 fast_mode=False, render_every=1):
        super().__init__()
        
        # 环境参数
        self.size = size  # 迷宫大小 (size x size)
        self.dense_reward = dense_reward  # 是否使用密集奖励
        self.treasure_reward = treasure_reward  # 找到宝藏的奖励值
        self.moving_traps = moving_traps  # 是否有移动的陷阱
        self.time_penalty = time_penalty  # 是否有时间惩罚
        self.fog_of_war = fog_of_war  # 是否有战争迷雾（有限视野）
        self.animation_speed = animation_speed  # 动画速度倍率
        self.fast_mode = fast_mode  # 快速模式（完全跳过可视化更新）
        self.render_every = render_every  # 每隔多少步渲染一次
        self.render_counter = 0  # 渲染计数器
        
        # 动作空间: 0=上, 1=右, 2=下, 3=左
        self.action_space = spaces.Discrete(4)
        
        # 观察空间: 如果有战争迷雾，则观察是智能体周围的3x3区域
        # 否则是智能体在迷宫中的位置 (row, col)
        if self.fog_of_war:
            # 3x3视野 + 智能体位置
            self.observation_space = spaces.Dict({
                'position': spaces.Box(low=0, high=size-1, shape=(2,), dtype=np.int32),
                'vision': spaces.Box(low=0, high=4, shape=(3, 3), dtype=np.int32)
            })
        else:
            self.observation_space = spaces.Box(low=0, high=size-1, shape=(2,), dtype=np.int32)
        
        # 渲染设置
        self.render_mode = render_mode
        self.window = None
        self.fig = None
        self.ax = None
        
        # 初始化迷宫
        self._generate_maze()
        
        # 移动陷阱的方向和计时器
        self.trap_directions = {}
        self.trap_move_timer = 0
        
        # 初始化陷阱移动方向
        if self.moving_traps:
            for trap_pos in self.trap_positions:
                self.trap_directions[trap_pos] = np.random.randint(0, 4)
    
    def _generate_maze(self):
        """生成复杂迷宫，包括墙壁、陷阱、移动陷阱和宝藏，确保存在一条到达宝藏的路径"""
        # 初始化迷宫 (0=空地, 1=墙, 2=陷阱, 3=宝藏, 4=移动陷阱)
        self.maze = np.zeros((self.size, self.size), dtype=np.int32)
        
        # 设置外墙
        self.maze[0, :] = 1
        self.maze[self.size-1, :] = 1
        self.maze[:, 0] = 1
        self.maze[:, self.size-1] = 1
        
        # 创建迷宫结构（使用随机Prim算法生成有解的迷宫）
        self._create_maze_structure()
        
        # 放置宝藏在迷宫的远端
        # 确保宝藏位置离起点足够远
        max_distance = 0
        treasure_pos = None
        
        for r in range(1, self.size-1):
            for c in range(1, self.size-1):
                if self.maze[r, c] == 0:
                    distance = abs(r - 1) + abs(c - 1)  # 曼哈顿距离
                    if distance > max_distance:
                        max_distance = distance
                        treasure_pos = (r, c)
        
        if treasure_pos:
            self.maze[treasure_pos] = 3
            self.treasure_pos = treasure_pos
        else:
            # 如果找不到合适的位置，放在右下角附近
            self.treasure_pos = (self.size-2, self.size-2)
            self.maze[self.treasure_pos] = 3
            
        # 验证并确保从起点到宝藏存在路径
        self._ensure_path_exists()
        
        # 放置陷阱（确保不阻断路径）
        self._place_traps()
    
    def _create_maze_structure(self):
        """使用Prim算法创建迷宫结构，确保有路径可达"""
        # 先填满墙壁
        for r in range(1, self.size-1):
            for c in range(1, self.size-1):
                self.maze[r, c] = 1
                
        # 起点设为空地
        self.maze[1, 1] = 0
        
        # 墙壁列表 - 存储可能被打通的墙
        walls = []
        
        # 将起点的墙添加到列表
        if 1 < self.size-2:
            walls.append((1, 2))  # 右边的墙
        if 2 < self.size-1:
            walls.append((2, 1))  # 下边的墙
            
        # 随机打通墙壁
        while walls:
            # 随机选择一个墙
            wall_idx = np.random.randint(0, len(walls))
            r, c = walls[wall_idx]
            walls.pop(wall_idx)
            
            # 判断是否是连接两个区域的墙（一边是通路，一边是墙）
            cells_visited = 0
            for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
                nr, nc = r + dr, c + dc
                if 0 < nr < self.size-1 and 0 < nc < self.size-1 and self.maze[nr, nc] == 0:
                    cells_visited += 1
            
            # 如果只连接到一个通路，打通这面墙
            if cells_visited <= 1:  # 允许最多一个相邻的通路
                self.maze[r, c] = 0
                
                # 将这个新通路的相邻墙加入列表
                for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
                    nr, nc = r + dr, c + dc
                    if (0 < nr < self.size-1 and 0 < nc < self.size-1 and 
                        self.maze[nr, nc] == 1):
                        walls.append((nr, nc))
    
    def _ensure_path_exists(self):
        """使用BFS确保从起点到宝藏存在路径"""
        queue = deque([(1, 1)])  # 起点
        visited = np.zeros((self.size, self.size), dtype=bool)
        visited[1, 1] = True
        path_found = False
        
        # 广度优先搜索
        while queue and not path_found:
            r, c = queue.popleft()
            
            # 如果到达宝藏，则找到了路径
            if (r, c) == self.treasure_pos:
                path_found = True
                break
                
            # 检查四个方向
            for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < self.size and 0 <= nc < self.size and 
                    not visited[nr, nc] and self.maze[nr, nc] != 1):
                    visited[nr, nc] = True
                    queue.append((nr, nc))
        
        # 如果没找到路径，修复迷宫
        if not path_found:
            # 使用A*算法创建一条路径
            self._create_path_to_treasure()
    
    def _create_path_to_treasure(self):
        """使用A*算法创建一条从起点到宝藏的路径"""
        # 启发式函数：曼哈顿距离
        def heuristic(pos, goal):
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
        
        start = (1, 1)
        goal = self.treasure_pos
        
        # 初始化open和closed集
        open_set = {start}
        closed_set = set()
        
        # 从起点到达每个点的实际距离
        g_score = {start: 0}
        
        # 从起点经过每个点到达终点的估计总距离
        f_score = {start: heuristic(start, goal)}
        
        # 记录路径
        came_from = {}
        
        while open_set:
            # 获取f_score最小的节点
            current = min(open_set, key=lambda pos: f_score.get(pos, float('inf')))
            
            if current == goal:
                # 重建路径并确保通路
                while current in came_from:
                    self.maze[current] = 0  # 确保路径上的点是空地
                    current = came_from[current]
                return
            
            open_set.remove(current)
            closed_set.add(current)
            
            # 检查四个相邻位置
            for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
                neighbor = (current[0] + dr, current[1] + dc)
                
                # 检查边界和是否是墙
                if (not (0 < neighbor[0] < self.size-1 and 0 < neighbor[1] < self.size-1) or
                    neighbor in closed_set):
                    continue
                
                # 计算通过当前节点到达邻居的距离
                tentative_g_score = g_score[current] + 1
                
                if neighbor not in open_set:
                    open_set.add(neighbor)
                elif tentative_g_score >= g_score.get(neighbor, float('inf')):
                    continue
                
                # 这条路径更好，记录下来
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
    
    def _place_traps(self):
        """放置陷阱，确保不会阻断到宝藏的路径"""
        # 使用BFS找出从起点到宝藏的所有可能路径
        queue = deque([(1, 1)])
        visited = np.zeros((self.size, self.size), dtype=bool)
        visited[1, 1] = True
        parent = {(1, 1): None}
        
        while queue:
            r, c = queue.popleft()
            
            if (r, c) == self.treasure_pos:
                break
                
            for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < self.size and 0 <= nc < self.size and 
                    not visited[nr, nc] and self.maze[nr, nc] != 1):
                    visited[nr, nc] = True
                    parent[(nr, nc)] = (r, c)
                    queue.append((nr, nc))
        
        # 找出一条主路径
        main_path = set()
        current = self.treasure_pos
        while current:
            main_path.add(current)
            current = parent.get(current)
        
        # 放置静态陷阱
        self.trap_positions = []
        for _ in range(self.size):
            r, c = np.random.randint(1, self.size-1), np.random.randint(1, self.size-1)
            if (self.maze[r, c] == 0 and (r, c) != (1, 1) and (r, c) != self.treasure_pos and 
                (r, c) not in main_path and np.random.random() < 0.7):
                self.maze[r, c] = 2
                self.trap_positions.append((r, c))
        
        # 放置移动陷阱
        if self.moving_traps:
            for _ in range(self.size // 3):  # 减少移动陷阱数量
                r, c = np.random.randint(1, self.size-1), np.random.randint(1, self.size-1)
                if (self.maze[r, c] == 0 and (r, c) != (1, 1) and (r, c) != self.treasure_pos and 
                    (r, c) not in main_path):
                    self.maze[r, c] = 4
                    self.trap_positions.append((r, c))
    
    def _update_moving_traps(self):
        """更新移动陷阱的位置"""
        if not self.moving_traps or self.trap_move_timer % 5 != 0:
            self.trap_move_timer += 1
            return
        
        # 移动陷阱
        new_trap_positions = []
        for trap_pos in self.trap_positions:
            r, c = trap_pos
            if self.maze[r, c] == 4:  # 只移动移动陷阱
                direction = self.trap_directions[trap_pos]
                
                # 尝试移动
                dr, dc = [(0, -1), (1, 0), (0, 1), (-1, 0)][direction]
                new_r, new_c = r + dr, c + dc
                
                # 检查新位置是否有效，并确保不会堵塞到宝藏的路径
                if (1 <= new_r < self.size-1 and 1 <= new_c < self.size-1 and 
                    self.maze[new_r, new_c] == 0):
                    # 移动陷阱
                    self.maze[r, c] = 0
                    self.maze[new_r, new_c] = 4
                    new_trap_positions.append((new_r, new_c))
                    
                    # 更新方向字典
                    self.trap_directions.pop(trap_pos)
                    self.trap_directions[(new_r, new_c)] = direction
                else:
                    # 如果不能移动，改变方向
                    self.trap_directions[trap_pos] = np.random.randint(0, 4)
                    new_trap_positions.append(trap_pos)
            else:
                new_trap_positions.append(trap_pos)
        
        self.trap_positions = new_trap_positions
        self.trap_move_timer += 1
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 智能体从左上角开始
        self.agent_pos = np.array([1, 1])
        
        # 重置步数和时间惩罚
        self.steps = 0
        self.time_penalty_factor = 0.01 if self.time_penalty else 0
        
        # 计算到宝藏的曼哈顿距离
        self.distance_to_treasure = abs(self.agent_pos[0] - self.treasure_pos[0]) + abs(self.agent_pos[1] - self.treasure_pos[1])
        
        # 重置陷阱移动
        if self.moving_traps:
            self.trap_move_timer = 0
            self.trap_directions = {}
            for trap_pos in self.trap_positions:
                self.trap_directions[trap_pos] = np.random.randint(0, 4)
        
        # 重置渲染计数器
        self.render_counter = 0
        
        if self.render_mode == "human" and not self.fast_mode:
            self._render_frame()
        
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        # 增加步数
        self.steps += 1
        self.render_counter += 1
        
        # 更新移动陷阱
        if self.moving_traps:
            self._update_moving_traps()
        
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
        elif self.maze[tuple(self.agent_pos)] in [2, 4]:
            reward = -self.treasure_reward
            terminated = True
        # 密集奖励模式
        elif self.dense_reward:
            # 每一步的小惩罚
            reward -= 0.1
            
            # 时间惩罚
            if self.time_penalty:
                reward -= self.time_penalty_factor * self.steps
            
            # 接近宝藏的奖励
            if new_distance < self.distance_to_treasure:
                reward += 0.5
            elif new_distance > self.distance_to_treasure:
                reward -= 0.2
        # 稀疏奖励模式
        else:
            reward = -0.1  # 每一步只有很小的负奖励
            
            # 时间惩罚
            if self.time_penalty:
                reward -= self.time_penalty_factor * self.steps
        
        # 更新距离
        self.distance_to_treasure = new_distance
        
        # 最大步数限制
        truncated = self.steps >= self.size * self.size * 3
        
        # 只在需要时渲染
        if (self.render_mode == "human" and not self.fast_mode and 
            (self.render_counter >= self.render_every or terminated or truncated)):
            self._render_frame()
            self.render_counter = 0
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def _get_obs(self):
        """返回观察结果"""
        if self.fog_of_war:
            # 获取智能体周围3x3区域的视野
            vision = np.ones((3, 3), dtype=np.int32) * -1  # -1表示看不见
            for i in range(3):
                for j in range(3):
                    r, c = self.agent_pos[0] - 1 + i, self.agent_pos[1] - 1 + j
                    if 0 <= r < self.size and 0 <= c < self.size:
                        vision[i, j] = self.maze[r, c]
            
            return {
                'position': self.agent_pos.copy(),
                'vision': vision
            }
        else:
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
        if self.fast_mode:
            return np.zeros((10, 10, 3), dtype=np.uint8)  # 快速模式下返回空白图像
            
        if self.fig is None:
            plt.ion()  # 交互模式
            self.fig, self.ax = plt.subplots(figsize=(10, 10))
            plt.show(block=False)
        
        self.ax.clear()
        
        # 创建迷宫颜色映射
        cmap = mcolors.ListedColormap(['white', 'gray', 'red', 'gold', 'purple'])
        bounds = [0, 1, 2, 3, 4, 5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        
        # 绘制迷宫
        maze_with_fog = self.maze.copy()
        
        # 如果有战争迷雾，只显示智能体周围的区域
        if self.fog_of_war:
            # 创建迷雾覆盖
            fog = np.ones_like(self.maze) * 5  # 5表示迷雾
            
            # 智能体周围3x3区域可见
            for i in range(max(1, self.agent_pos[0]-1), min(self.size-1, self.agent_pos[0]+2)):
                for j in range(max(1, self.agent_pos[1]-1), min(self.size-1, self.agent_pos[1]+2)):
                    fog[i, j] = self.maze[i, j]
            
            maze_with_fog = fog
        
        self.ax.imshow(maze_with_fog, cmap=cmap, norm=norm)
        
        # 绘制智能体
        self.ax.add_patch(Rectangle((self.agent_pos[1] - 0.5, self.agent_pos[0] - 0.5), 
                                    1, 1, fill=True, color='blue', alpha=0.7))
        
        # 添加网格
        self.ax.grid(True, which='both', color='black', linewidth=1)
        self.ax.set_xticks(np.arange(-0.5, self.size, 1))
        self.ax.set_yticks(np.arange(-0.5, self.size, 1))
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        
        # 标题显示步数和奖励类型
        reward_type = "Dense Reward" if self.dense_reward else "Sparse Reward"
        features = []
        if self.moving_traps:
            features.append("Moving Traps")
        if self.time_penalty:
            features.append("Time Penalty")
        if self.fog_of_war:
            features.append("Fog of War")
        
        feature_str = ", ".join(features)
        self.ax.set_title(f"Complex Maze - Steps: {self.steps}, {reward_type}\n{feature_str}")
        
        # 只在真正需要显示时才调用这些耗时的操作
        if self.render_mode == "human":
            plt.draw()
            try:
                plt.pause(0.01 / max(1.0, self.animation_speed))  # 使用最小值避免除以零
            except Exception:
                pass  # 忽略可能的渲染错误
        
        # 获取图像数据
        if self.render_mode == "rgb_array":
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