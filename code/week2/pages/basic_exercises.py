import streamlit as st
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import gymnasium as gym

# 添加父目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.visualizations import plot_grid_world, plot_q_values, plot_learning_curve
from utils.grid_world_env import GridWorldEnv, create_random_grid_world

def show():
    """显示基础练习页面"""
    st.title("基础练习")
    
    st.info("""
    本节包含一些基础练习，帮助你熟悉强化学习的基本概念和Grid World环境的使用。
    通过这些练习，你将学习如何创建环境、实现简单的策略，以及评估智能体的表现。
    """)
    
    # 练习1：创建自定义Grid World环境
    st.subheader("练习1：创建自定义Grid World环境")
    
    st.markdown("""
    **目标**：创建一个简单的Grid World环境，并实现基本的交互功能。
    
    **步骤**：
    1. 定义网格地图（障碍物、目标、陷阱等）
    2. 实现智能体的移动逻辑
    3. 设计奖励函数
    4. 测试环境功能
    
    下面是一个基本框架，你可以在此基础上进行修改和扩展：
    ```python
    import numpy as np
    import gymnasium as gym
    from gymnasium import spaces
    
    class MyGridWorldEnv(gym.Env):
        def __init__(self, size=5):
            super().__init__()
            self.size = size
            
            # 定义动作空间和观测空间
            self.action_space = spaces.Discrete(4)  # 上、右、下、左
            self.observation_space = spaces.Discrete(size * size)
            
            # 创建网格地图 (0=空地, 1=墙壁, 2=目标, 3=陷阱)
            self.grid_map = np.zeros((size, size), dtype=int)
            # 在这里添加障碍物、目标和陷阱
            
            # 设置智能体初始位置
            self.agent_pos = np.array([0, 0])
            
            # 设置目标位置
            self.target_pos = np.array([size-1, size-1])
        
        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            # 重置智能体位置
            self.agent_pos = np.array([0, 0])
            # 返回初始观测
            observation = self._get_obs()
            return observation, {}
        
        def step(self, action):
            # 根据动作移动智能体
            # 计算奖励
            # 检查是否终止
            # 返回结果
            pass
        
        def _get_obs(self):
            # 将智能体位置转换为观测
            return self.agent_pos[0] * self.size + self.agent_pos[1]
    ```
    
    **思考题**：
    - 如何设计更有趣的网格地图？
    - 如何调整奖励函数来引导智能体更快地找到目标？
    """)
    
    # 练习2：实现随机策略
    st.subheader("练习2：实现随机策略")
    
    st.markdown("""
    **目标**：实现一个随机策略，并在Grid World环境中进行测试。
    
    **随机策略代码**：
    ```python
    def random_policy(observation, env):
        # 随机选择一个动作
        return env.action_space.sample()
    ```
    
    **使用策略进行交互**：
    ```python
    # 创建环境
    env = MyGridWorldEnv()
    
    # 重置环境
    observation, info = env.reset()
    
    # 交互循环
    total_reward = 0
    done = False
    
    while not done:
        # 使用随机策略选择动作
        action = random_policy(observation, env)
        
        # 执行动作
        observation, reward, terminated, truncated, info = env.step(action)
        
        # 累积奖励
        total_reward += reward
        
        # 检查是否结束
        done = terminated or truncated
    
    print(f"总奖励: {total_reward}")
    ```
    """)
    
    # 随机策略演示
    if st.button("运行随机策略演示", key="random_policy_demo"):
        # 创建环境
        env = GridWorldEnv(render_mode=None, size=5)
        
        # 可视化演示结果
        episodes = 10
        steps_list = []
        rewards_list = []
        
        for i in range(episodes):
            observation, info = env.reset()
            episode_reward = 0
            steps = 0
            terminated = False
            truncated = False
            
            while not (terminated or truncated):
                # 随机选择动作
                action = env.action_space.sample()
                
                # 执行动作
                observation, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                steps += 1
                
                # 设置最大步数
                if steps >= 100:
                    truncated = True
            
            steps_list.append(steps)
            rewards_list.append(episode_reward)
        
        # 创建DataFrame用于展示结果
        results_df = pd.DataFrame({
            'Episode': range(1, episodes+1),
            'Steps': steps_list,
            'Reward': rewards_list
        })
        
        st.dataframe(results_df)
        
        # 显示平均结果
        st.markdown(f"""
        **随机策略结果**:
        - 平均步数: {np.mean(steps_list):.2f}
        - 平均奖励: {np.mean(rewards_list):.2f}
        - 成功率: {sum(1 for r in rewards_list if r > 0) / episodes:.2f}
        """)
        
        # 绘制步数和奖励
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.bar(range(1, episodes+1), steps_list, color='blue', alpha=0.7)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Steps')
        ax1.set_title('Steps per Episode')
        
        ax2.bar(range(1, episodes+1), rewards_list, color='green', alpha=0.7)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Reward')
        ax2.set_title('Reward per Episode')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # 练习3：实现简单的$\epsilon$-greedy策略
    st.subheader("练习3：实现$\epsilon$-greedy策略")
    
    st.markdown("""
    **目标**：实现一个简单的$\epsilon$-greedy策略，平衡探索与利用。
    
    **$\epsilon$-greedy策略代码**：
    ```python
    def epsilon_greedy_policy(q_values, epsilon):
        # 以概率epsilon进行探索（随机选择动作）
        if np.random.random() < epsilon:
            return np.random.randint(len(q_values))
        # 以概率1-epsilon进行利用（选择Q值最大的动作）
        else:
            return np.argmax(q_values)
    
    # 创建简单的Q表
    q_table = np.zeros((state_count, action_count))
    
    # 在每一步使用epsilon-greedy策略
    def select_action(state, epsilon):
        return epsilon_greedy_policy(q_table[state], epsilon)
    ```
    
    **思考题**：
    - $\epsilon$值如何影响智能体的学习过程？
    - 如何随着学习的进行调整$\epsilon$值？
    """)
    
    # 交互式$\epsilon$-greedy演示
    st.subheader("交互式$\epsilon$-greedy演示")
    
    # 设置参数
    epsilon = st.slider("$\epsilon$ 值", 0.0, 1.0, 0.1, 0.1, key="epsilon_slider")
    
    # Q表初始化方法选择
    q_init_method = st.radio(
        "Q表初始化方法",
        ["零初始化", "随机乐观初始化", "基于启发式规则初始化"],
        help="选择不同的Q表初始化方法会影响智能体的初始探索行为"
    )
    
    # 状态和动作空间
    state_count = 25  # 5x5网格
    action_count = 4  # 上右下左
    
    # 根据不同方法初始化Q表
    if q_init_method == "零初始化":
        q_table = np.zeros((state_count, action_count))
        st.info("零初始化是最简单的方法，但可能导致智能体在开始时缺乏探索动力，因为所有动作的Q值都相同。")
    elif q_init_method == "随机乐观初始化":
        q_table = np.random.uniform(0.5, 1.0, (state_count, action_count))
        st.info("乐观初始化设置较高的初始Q值，会鼓励智能体在初始阶段进行更多探索，因为未尝试过的动作Q值较高。")
    else:  # 基于启发式规则初始化
        # 创建一个简单的启发式Q表 - 偏好向目标方向移动
        q_table = np.zeros((state_count, action_count))
        grid_size = 5
        target_pos = np.array([2, 2])  # 假设目标在中心
        
        # 为每个状态设置一个启发式值
        for s in range(state_count):
            row = s // grid_size
            col = s % grid_size
            
            # 计算到目标的方向
            d_row = target_pos[0] - row
            d_col = target_pos[1] - col
            
            # 偏好向目标方向移动的动作
            if d_row < 0:  # 目标在上方
                q_table[s, 0] = 0.5  # 上
            elif d_row > 0:  # 目标在下方
                q_table[s, 2] = 0.5  # 下
            
            if d_col < 0:  # 目标在左边
                q_table[s, 3] = 0.5  # 左
            elif d_col > 0:  # 目标在右边
                q_table[s, 1] = 0.5  # 右
        
        st.info("启发式初始化根据问题的先验知识设置初始Q值，这里简单地偏好向目标方向移动。")
    
    # 显示Q表构建指导
    with st.expander("Q表构建指导"):
        st.markdown("""
        ### 如何构建Q表

        Q表是强化学习中用于存储状态-动作对价值的数据结构。以下是构建Q表的基本步骤：

        1. **确定状态和动作空间**：
           - 状态空间：在Grid World中，通常是智能体的位置坐标
           - 动作空间：通常是上、右、下、左四个基本动作

        2. **选择合适的初始化方法**：
           - **零初始化**：所有Q值初始为0，简单但可能导致初期探索不足
           - **乐观初始化**：初始Q值设置为较高的值，鼓励初期探索
           - **基于问题的初始化**：利用对问题的先验知识设置初始Q值
        
        3. **更新Q值的方法** (后续课程将详细介绍)：
           - Q-learning: Q(s,a) ← Q(s,a) + α[r + γ·max<sub>a'</sub>Q(s',a') - Q(s,a)]
           - SARSA: Q(s,a) ← Q(s,a) + α[r + γ·Q(s',a') - Q(s,a)]

        4. **实现技巧**：
           - 对于小规模问题，可以使用多维数组或嵌套字典
           - 对于大规模问题，考虑使用函数逼近方法（如神经网络）

        在这个练习中，我们提供了三种初始化方法供你尝试，观察不同初始化方法对智能体行为的影响。
        """)
    
    # 执行$\epsilon$-greedy策略
    if st.button("执行$\epsilon$-greedy策略", key="epsilon_greedy_demo"):
        # 创建Grid World环境用于演示
        env = GridWorldEnv(render_mode=None, size=5)
        
        # 设置episode数
        episodes = 10
        max_steps = 100
        
        steps_list = []
        rewards_list = []
        
        # 运行多个episode
        for episode in range(episodes):
            observation, info = env.reset()
            total_reward = 0
            steps = 0
            terminated = False
            truncated = False
            
            while not (terminated or truncated):
                # 使用$\epsilon$-greedy策略选择动作
                if np.random.random() < epsilon:
                    action = env.action_space.sample()  # 探索
                else:
                    action = np.argmax(q_table[observation])  # 利用
                
                # 执行动作
                observation, reward, terminated, truncated, info = env.step(action)
                
                total_reward += reward
                steps += 1
                
                # 设置最大步数
                if steps >= max_steps:
                    truncated = True
            
            steps_list.append(steps)
            rewards_list.append(total_reward)
        
        # 创建DataFrame用于展示结果
        results_df = pd.DataFrame({
            'Episode': range(1, episodes+1),
            'Steps': steps_list,
            'Reward': rewards_list
        })
        
        st.dataframe(results_df)
        
        # 显示平均结果
        st.markdown(f"""
        **$\epsilon$-greedy策略结果 (ε={epsilon}, 初始化方法={q_init_method})**:
        - 平均步数: {np.mean(steps_list):.2f}
        - 平均奖励: {np.mean(rewards_list):.2f}
        - 成功率: {sum(1 for r in rewards_list if r > 0) / episodes:.2f}
        """)
        
        # 绘制Q值表可视化
        st.subheader("Q值表可视化")
        fig = plot_q_values(q_table, grid_size=5)
        st.pyplot(fig)
        
        # 添加Q值可视化解释
        with st.expander("如何理解Q值可视化图"):
            st.markdown("""
            ### Q值可视化图解读指南
            
            Q值可视化图展示了每个状态(格子)中不同动作的Q值大小和方向：
            
            1. **箭头方向**：表示动作的方向（上、右、下、左）
            2. **箭头长度**：表示该动作的Q值大小，箭头越长表示Q值越大
            3. **颜色深浅**：越深的颜色表示Q值越大，越浅的颜色表示Q值越小
            4. **最佳动作**：每个格子中箭头最长的方向通常是当前策略认为的最佳动作
            
            #### 典型模式解读：
            
            - **目标周围**：通常会形成指向目标的箭头模式，表示智能体已经学会了如何到达目标
            - **无明显方向偏好**：如果所有方向箭头长度相似，说明智能体对该状态下的动作选择没有明显偏好
            - **墙壁边缘**：通常会有避开墙壁的箭头模式，表示智能体学会了避免无效动作
            
            #### 不同初始化方法的表现：
            - **零初始化**：初期所有箭头长度相等，随着学习逐渐形成明显的方向性
            - **乐观初始化**：初期所有箭头都较长，随着学习过程中，无效或次优动作的箭头会变短
            - **启发式初始化**：初期就有指向目标的倾向，学习过程会进一步强化这种模式
            
            观察Q值的分布和变化可以帮助理解强化学习算法的收敛过程和智能体的决策偏好。
            """)
    
    # 练习4：设计多样化的奖励函数
    st.subheader("练习4：设计多样化的奖励函数")
    
    st.markdown("""
    **目标**：探索不同的奖励函数设计，观察它们对智能体行为的影响。
    
    **常见的奖励函数类型**：
    
    1. **稀疏奖励**：
       ```python
       # 只在到达目标时给予奖励
       if np.array_equal(agent_pos, target_pos):
           reward = 1.0
       else:
           reward = 0.0
       ```
    
    2. **稠密奖励**：
       ```python
       # 基础奖励
       reward = -0.1  # 每一步的小惩罚
       
       # 到达目标奖励
       if np.array_equal(agent_pos, target_pos):
           reward += 1.0
       
       # 基于距离的奖励
       prev_distance = np.linalg.norm(prev_pos - target_pos)
       new_distance = np.linalg.norm(agent_pos - target_pos)
       reward += 0.1 * (prev_distance - new_distance)  # 如果靠近目标，给予正奖励
       ```
    
    3. **惩罚与奖励结合**：
       ```python
       # 基础奖励
       reward = -0.1  # 时间成本
       
       # 到达目标
       if np.array_equal(agent_pos, target_pos):
           reward += 1.0
       
       # 触碰陷阱
       elif grid_map[agent_pos[0], agent_pos[1]] == 3:
           reward += -1.0
       
       # 无效动作（撞墙）
       elif np.array_equal(agent_pos, prev_pos) and not np.array_equal(agent_pos, initial_pos):
           reward += -0.2  # 额外惩罚无效动作
       ```
    
    **思考题**：
    - 稀疏奖励和稠密奖励各有什么优缺点？
    - 如何设计奖励函数来引导智能体在复杂环境中学习特定行为？
    - 不良的奖励设计可能导致什么问题？
    """)
    
    # 奖励函数对比演示
    st.subheader("奖励函数对比演示")
    
    reward_type = st.selectbox(
        "选择奖励函数类型",
        ["稀疏奖励", "稠密奖励", "惩罚与奖励结合"]
    )
    
    if st.button("运行奖励函数演示", key="reward_demo"):
        # 创建环境
        env = GridWorldEnv(render_mode=None, size=5)
        
        # 设置episode数
        episodes = 100
        max_steps = 100
        rewards_history = []
        
        # 自定义step函数，根据选择使用不同的奖励计算
        def custom_step(env, action, reward_type):
            # 保存之前的位置
            prev_pos = env._agent_location.copy()
            
            # 动作映射到方向变化 (行,列)
            direction = {
                0: (-1, 0),  # 上
                1: (0, 1),   # 右
                2: (1, 0),   # 下
                3: (0, -1)   # 左
            }
            
            # 计算新位置
            delta_row, delta_col = direction[action]
            new_position = env._agent_location + np.array([delta_row, delta_col])
            
            # 检查是否越界或撞墙
            valid_move = (
                0 <= new_position[0] < env.size and
                0 <= new_position[1] < env.size and
                env._grid_map[new_position[0], new_position[1]] != 1
            )
            
            if valid_move:
                env._agent_location = new_position
            
            # 获取当前位置的单元格类型
            current_cell = env._grid_map[env._agent_location[0], env._agent_location[1]]
            
            # 根据奖励类型计算奖励
            if reward_type == "稀疏奖励":
                # 只在到达目标时给予奖励
                if np.array_equal(env._agent_location, env._target_location):
                    reward = 1.0
                else:
                    reward = 0.0
                    
            elif reward_type == "稠密奖励":
                # 基础奖励
                reward = -0.1  # 每一步的小惩罚
                
                # 到达目标奖励
                if np.array_equal(env._agent_location, env._target_location):
                    reward += 1.0
                
                # 基于距离的奖励
                prev_distance = np.linalg.norm(prev_pos - env._target_location)
                new_distance = np.linalg.norm(env._agent_location - env._target_location)
                reward += 0.1 * (prev_distance - new_distance)  # 如果靠近目标，给予正奖励
                
            else:  # "惩罚与奖励结合"
                # 基础奖励
                reward = -0.1  # 时间成本
                
                # 到达目标
                if np.array_equal(env._agent_location, env._target_location):
                    reward += 1.0
                
                # 触碰陷阱（假设3是陷阱）
                elif current_cell == 3:
                    reward += -1.0
                
                # 无效动作（撞墙）
                elif not valid_move:
                    reward += -0.2  # 额外惩罚无效动作
            
            # 判断是否终止
            terminated = np.array_equal(env._agent_location, env._target_location) or current_cell == 3
            truncated = False
            
            # 获取观测
            observation = env._agent_location[0] * env.size + env._agent_location[1]
            
            return observation, reward, terminated, truncated, {}
        
        # 使用随机策略运行多个episode
        for episode in range(episodes):
            env.reset()
            episode_reward = 0
            
            for step in range(max_steps):
                # 随机选择动作
                action = env.action_space.sample()
                
                # 使用自定义step函数
                _, reward, terminated, truncated, _ = custom_step(env, action, reward_type)
                
                episode_reward += reward
                
                if terminated or truncated:
                    break
            
            rewards_history.append(episode_reward)
        
        # 展示学习曲线
        st.subheader(f"{reward_type}学习曲线")
        fig = plot_learning_curve(rewards_history, window=10)
        st.pyplot(fig)
        
        # 添加学习曲线解释
        with st.expander("如何理解学习曲线"):
            st.markdown("""
            ### 学习曲线解读指南
            
            学习曲线展示了智能体在训练过程中累积奖励的变化趋势，是评估强化学习效果的重要指标：
            
            1. **蓝色散点**：每个episode的实际累积奖励值，显示智能体在单次尝试中的表现
            2. **橙色曲线**：移动平均线，平滑了短期波动，更清晰地展示学习趋势
            3. **红色趋势线**：整体学习趋势，斜率为正表示学习效果在提升
            
            #### 不同阶段的典型表现：
            
            - **初始阶段**：通常奖励较低且波动较大，表示智能体在随机探索环境
            - **学习阶段**：曲线开始上升，波动逐渐减小，表示智能体开始学习有效策略
            - **收敛阶段**：曲线趋于平稳且处于较高水平，表示策略已基本稳定
            
            #### 不同奖励函数下的学习曲线差异：
            
            - **稀疏奖励**：初期学习较慢，曲线上升缓慢，但最终策略通常更纯粹
            - **稠密奖励**：学习通常更快，曲线上升更陡，但可能导致次优策略
            - **惩罚与奖励结合**：初期可能波动较大，但随着学习进行，通常能形成较为合理的策略
            
            观察学习曲线可以帮助判断算法是否收敛、学习效率如何，以及是否需要调整学习参数或奖励函数设计。
            """)
        
        # 显示统计信息
        st.markdown(f"""
        **{reward_type}统计信息**:
        - 平均奖励: {np.mean(rewards_history):.2f}
        - 最大奖励: {np.max(rewards_history):.2f}
        - 最小奖励: {np.min(rewards_history):.2f}
        - 标准差: {np.std(rewards_history):.2f}
        """)
    
    # 小结与扩展练习
    st.subheader("小结与扩展练习")
    
    st.markdown("""
    通过上述基础练习，你应该了解了：
    
    1. 如何创建和使用Grid World环境
    2. 如何实现简单的策略（随机策略、$\epsilon$-greedy策略）
    3. 不同类型的奖励函数设计及其影响
    
    **扩展练习**：
    
    1. 尝试实现一个更大的Grid World环境（如10x10）并添加更多障碍物和陷阱
    
    2. 实现一个衰减的$\epsilon$-greedy策略，随着训练的进行逐渐减小$\epsilon$值
    
    3. 设计一个更复杂的奖励函数，如考虑到距离目标的距离、障碍物的分布等
    
    4. 尝试比较不同策略（随机策略、$\epsilon$-greedy策略、贪婪策略）的性能
    
    5. 尝试实现一个简单的Q-learning算法，在Grid World环境中学习最优策略
    """)

if __name__ == "__main__":
    show() 