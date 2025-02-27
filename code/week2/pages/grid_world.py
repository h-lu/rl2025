import streamlit as st
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import gymnasium as gym

# 添加父目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.visualizations import plot_grid_world
from utils.grid_world_env import GridWorldEnv, create_random_grid_world

def show():
    """显示Grid World环境页面"""
    st.title("Grid World 环境")
    
    st.info("""
    Grid World 是强化学习中的一个经典环境，通常用于演示和教学。它是一个简单的二维网格世界，智能体需要在网格中导航，
    从起点到达目标，同时避开障碍物或陷阱。
    """)
    
    # Grid World环境介绍
    st.subheader("Grid World 环境介绍")
    
    st.markdown("""
    在 Grid World 环境中：
    
    - **状态**：智能体在网格中的位置（行、列坐标）
    - **动作**：通常有4个基本动作：上、右、下、左
    - **奖励**：
        - 到达目标：正奖励（如+1）
        - 每一步的移动：小的负奖励（如-0.1），鼓励智能体尽快到达目标
        - 撞墙：惩罚或无效动作（不移动）
        - 掉入陷阱：负奖励（如-1）
    
    Grid World 的优势在于其简单性和可视化的便利性，使其成为学习和测试强化学习算法的理想环境。
    """)
    
    # 交互式演示：自定义 Grid World 环境
    st.subheader("交互式演示：Grid World 环境")
    
    # 选择环境类型
    env_type = st.selectbox(
        "选择环境类型",
        ["默认环境", "迷宫环境", "带陷阱的环境", "随机环境"]
    )
    
    # 设置网格大小
    grid_size = st.slider("网格大小", 3, 10, 5, 1, key="grid_size")
    
    # 如果是随机环境，添加障碍物概率滑块
    obstacle_prob = 0.3
    if env_type == "随机环境":
        obstacle_prob = st.slider("障碍物概率", 0.0, 0.5, 0.3, 0.05, key="obstacle_prob")
    
    # 创建Grid World环境
    if env_type == "默认环境":
        map_type = "default"
    elif env_type == "迷宫环境":
        map_type = "maze"
    elif env_type == "带陷阱的环境":
        map_type = "traps"
    else:  # 随机环境
        grid_map = create_random_grid_world(grid_size, obstacle_prob)
        env = GridWorldEnv(render_mode=None, size=grid_size)
        env._grid_map = grid_map
        env._init_locations()
    
    if env_type != "随机环境":
        env = GridWorldEnv(render_mode=None, size=grid_size, map_type=map_type)
    
    # 显示环境状态
    st.markdown("### 当前环境状态")
    
    # 获取网格地图和智能体位置
    grid_map = env._grid_map
    agent_pos = env._agent_location
    target_pos = env._target_location
    
    # 绘制网格地图
    fig = plot_grid_world(grid_map, agent_pos, target_pos)
    st.pyplot(fig)
    
    # 显示环境信息
    st.markdown("### 环境信息")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        - **网格大小**: {grid_size} x {grid_size}
        - **智能体位置**: ({agent_pos[0]}, {agent_pos[1]})
        - **目标位置**: ({target_pos[0]}, {target_pos[1]})
        """)
    
    with col2:
        # 统计环境元素
        empty_count = np.sum(grid_map == 0)
        wall_count = np.sum(grid_map == 1)
        goal_count = np.sum(grid_map == 2)
        trap_count = np.sum(grid_map == 3)
        
        st.markdown(f"""
        - **空白单元格数量**: {empty_count}
        - **墙壁数量**: {wall_count}
        - **目标数量**: {goal_count}
        - **陷阱数量**: {trap_count}
        """)
    
    # 如何使用Gymnasium自定义环境
    st.subheader("如何使用Gymnasium实现Grid World环境")
    
    st.markdown("""
    Gymnasium (以前称为 Gym) 是OpenAI开发的强化学习环境库，提供了标准的API接口和多种预定义环境。
    下面是创建自定义Grid World环境的关键步骤：
    """)
    
    # 代码解释标签页
    tab1, tab2, tab3, tab4 = st.tabs(["环境初始化", "动作执行", "环境渲染", "使用环境"])
    
    with tab1:
        st.markdown("""
        ### 环境初始化
        
        创建一个继承自 `gym.Env` 的类，并定义必要的属性和方法：
        
        ```python
        import gymnasium as gym
        from gymnasium import spaces
        import numpy as np
        
        class GridWorldEnv(gym.Env):
            metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
            
            def __init__(self, render_mode=None, size=5):
                super().__init__()
                self.size = size  # Grid world 大小
                self.window_size = 512  # PyGame 窗口大小
                
                # 动作空间：上下左右
                self.action_space = spaces.Discrete(4)
                
                # 观测空间：智能体位置
                self.observation_space = spaces.Discrete(size * size)
                
                # Grid world 地图
                self._grid_map = np.array([
                    [0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 2, 1, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0]
                ])
                
                # 初始化位置信息
                self._target_location = np.array([2, 2])
                self._agent_location = np.array([0, 0])
                
                # 初始化渲染
                self.render_mode = render_mode
                # ...
        ```
        
        这里，我们定义了：
        - 动作空间：4个离散动作（上、右、下、左）
        - 观测空间：离散状态，表示网格位置
        - 网格地图：0=空地，1=墙壁，2=目标，3=陷阱
        - 智能体和目标的初始位置
        """)
    
    with tab2:
        st.code('''
def step(self, action):
    """
    执行一步动作，返回下一个状态、奖励、是否终止等信息
    
    参数:
        action: 动作，0=上, 1=右, 2=下, 3=左
        
    返回:
        observation: 新的状态
        reward: 获得的奖励
        terminated: 是否到达终止状态(目标或陷阱)
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
        0 <= new_position[0] < self.size and 
        0 <= new_position[1] < self.size and 
        self._grid_map[new_position[0], new_position[1]] != 1
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
    elif current_cell == 3:
        # 掉入陷阱
        reward = -1.0
        terminated = True
    
    # 将智能体位置转换为离散观测空间的索引
    observation = self._agent_location[0] * self.size + self._agent_location[1]
    
    return observation, reward, terminated, truncated, {}
        ''', language="python")
        
        st.markdown("""
        关键步骤：
        1. 根据动作计算新位置
        2. 检查边界和障碍物，更新智能体位置
        3. 根据当前位置计算奖励和游戏终止状态
        4. 返回新的观测、奖励和其他信息
        """)
    
    with tab3:
        st.code('''
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
    
    # 绘制智能体
    pygame.draw.circle(
        self.window,
        (0, 0, 255),  # 蓝色
        (self._agent_location[1] * pix_square_size + pix_square_size // 2,
         self._agent_location[0] * pix_square_size + pix_square_size // 2),
        pix_square_size // 3,
    )
    
    # 获取渲染结果
    pygame.pixelcopy.surface_to_array(canvas, self.window)
    return canvas
        ''', language="python")
        
        st.markdown("""
        PyGame是常用的渲染工具，可以：
        1. 绘制网格地图和各种元素
        2. 使用不同颜色表示不同类型的单元格
        3. 实时显示智能体位置
        """)
    
    with tab4:
        st.markdown("""
        ### 使用环境
        
        如何使用自定义的Grid World环境：
        
        ```python
        # 创建环境
        env = GridWorldEnv(render_mode="human")
        
        # 重置环境
        observation, info = env.reset()
        
        # 交互循环
        for _ in range(1000):
            # 随机选择动作
            action = env.action_space.sample()
            
            # 执行动作
            observation, reward, terminated, truncated, info = env.step(action)
            
            # 渲染
            env.render()
            
            # 检查是否结束
            if terminated or truncated:
                observation, info = env.reset()
        
        # 关闭环境
        env.close()
        ```
        
        这个例子展示了：
        1. 如何创建环境实例
        2. 如何重置环境
        3. 如何执行动作并获取反馈
        4. 如何在环境中实现一个完整的交互循环
        """)
    
    # Grid World的扩展与变种
    st.subheader("Grid World 的扩展与变种")
    
    st.markdown("""
    Grid World 是一个高度可扩展的环境，可以通过以下方式进行扩展：
    
    1. **增加网格大小**：创建更大、更复杂的迷宫
    
    2. **添加更多元素**：
       - 移动的障碍物
       - 多个目标，每个具有不同的奖励
       - 钥匙和门：智能体必须先拾取钥匙才能通过门
       
    3. **修改动作空间**：
       - 添加斜向移动
       - 引入概率性动作（有一定概率执行不同于预期的动作）
       
    4. **修改奖励函数**：
       - 基于到目标的距离设计奖励
       - 添加时间惩罚，鼓励更快到达目标
       - 设计形状奖励 (shaped rewards)，引导学习过程
    
    5. **修改观测空间**：
       - 部分可观测：智能体只能看到周围的单元格
       - 添加噪声：观测中包含一定的不确定性
    """)
    
    # 课后作业提示
    st.subheader("课后作业提示")
    
    st.markdown("""
    为完成小组项目一：迷宫寻宝 (Grid World) 环境搭建，考虑以下建议：
    
    1. **从基础开始**：先确保基本的Grid World环境正常工作
    
    2. **逐步扩展**：
       - 首先实现基本功能（移动、碰撞检测、目标检测）
       - 然后添加额外元素（陷阱、多个目标等）
       - 最后优化渲染和用户交互
    
    3. **团队协作**：
       - 分工明确：不同队员负责不同模块
       - 定期沟通：及时解决问题和同步进度
       - 代码审查：相互检查代码，保证质量
    
    4. **测试与调试**：
       - 单元测试：测试各个功能模块
       - 集成测试：测试整个环境的交互
       - 边缘情况：测试特殊情况和极限条件
    
    5. **使用AI辅助工具**：
       - GitHub Copilot：帮助生成常规代码
       - Tabnine：提供智能代码补全
       - 但记得：理解生成的代码并进行必要修改
    """)

if __name__ == "__main__":
    show() 