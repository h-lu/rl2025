import streamlit as st
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import time
import gymnasium as gym

# 添加父目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.visualizations import plot_grid_world, plot_q_values, plot_learning_curve
from utils.grid_world_env import GridWorldEnv, create_random_grid_world

def show():
    """显示进阶练习页面"""
    st.title("进阶练习")
    
    st.info("""
    本节包含一些进阶练习，帮助你深入理解强化学习算法（特别是Q-learning）以及如何处理更复杂的Grid World环境。
    这些练习将为你后续学习更高级的强化学习算法打下基础。
    """)
    
    # 进阶练习1：Q-learning算法实现
    st.subheader("进阶练习1：Q-learning算法实现")
    
    st.markdown("""
    **目标**：实现Q-learning算法并在Grid World环境中训练智能体。
    
    **Q-learning简介**：
    
    Q-learning是一种基于值函数的强化学习算法，它学习动作-状态对的价值（Q值），使智能体能够做出最优决策。
    
    **关键概念**：
    
    1. **Q表**：存储每个状态-动作对的估计价值
    2. **贝尔曼方程**：$Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$
    3. **超参数**：
       - $\alpha$（学习率）：控制新信息的更新速度
       - $\gamma$（折扣因子）：控制未来奖励的重要性
       - $\epsilon$（探索率）：控制探索与利用的平衡
    
    **算法步骤**：
    
    1. 初始化Q表
    2. 对于每个episode：
       a. 初始化状态 $s$
       b. 对于每个步骤：
          i. 使用$\epsilon$-greedy策略选择动作 $a$
          ii. 执行动作 $a$，观测奖励 $r$ 和新状态 $s'$
          iii. 更新Q值：$Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$
          iv. $s \leftarrow s'$
          v. 如果 $s$ 是终止状态，结束本轮episode
    
    **Q-learning算法代码**：
    ```python
    def q_learning(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1, epsilon_decay=0.99, min_epsilon=0.01):
        # 初始化Q表
        state_size = env.observation_space.n
        action_size = env.action_space.n
        q_table = np.zeros((state_size, action_size))
        
        # 记录每个episode的奖励
        rewards = []
        
        for episode in range(episodes):
            # 重置环境
            state, _ = env.reset()
            total_reward = 0
            done = False
            
            # 衰减epsilon
            current_epsilon = max(epsilon * (epsilon_decay ** episode), min_epsilon)
            
            while not done:
                # 选择动作（epsilon-greedy策略）
                if np.random.random() < current_epsilon:
                    action = env.action_space.sample()  # 探索
                else:
                    action = np.argmax(q_table[state])  # 利用
                
                # 执行动作
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # 更新Q值
                best_next_action = np.argmax(q_table[next_state])
                td_target = reward + gamma * q_table[next_state, best_next_action] * (not done)
                td_error = td_target - q_table[state, action]
                q_table[state, action] += alpha * td_error
                
                # 更新状态和累积奖励
                state = next_state
                total_reward += reward
            
            rewards.append(total_reward)
        
        return q_table, rewards
    ```
    """)
    
    # 交互式Q-learning演示
    st.subheader("交互式Q-learning演示")
    
    # 设置Q-learning参数
    col1, col2, col3 = st.columns(3)
    
    with col1:
        episodes = st.number_input("训练episodes数", 10, 1000, 200, step=10)
        alpha = st.number_input("学习率 α", 0.01, 1.0, 0.1, step=0.01)
        
    with col2:
        gamma = st.number_input("折扣因子 γ", 0.5, 1.0, 0.99, step=0.01)
        epsilon = st.number_input("初始探索率 ε", 0.01, 1.0, 0.1, step=0.01)
        
    with col3:
        epsilon_decay = st.number_input("探索率衰减", 0.9, 0.999, 0.99, step=0.001)
        min_epsilon = st.number_input("最小探索率", 0.001, 0.1, 0.01, step=0.001)
    
    # 选择环境类型
    env_type = st.selectbox(
        "选择环境类型",
        ["默认环境", "迷宫环境", "带陷阱的环境", "随机环境"]
    )
    
    if st.button("运行Q-learning", key="run_qlearning"):
        # 显示进度条
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 创建环境
        if env_type == "默认环境":
            env = GridWorldEnv(render_mode=None, map_type="default")
        elif env_type == "迷宫环境":
            env = GridWorldEnv(render_mode=None, map_type="maze")
        elif env_type == "带陷阱的环境":
            env = GridWorldEnv(render_mode=None, map_type="traps")
        else:  # 随机环境
            grid_map = create_random_grid_world(5, 0.3)
            env = GridWorldEnv(render_mode=None)
            env._grid_map = grid_map
            env._init_locations()
        
        # 显示环境
        grid_map = env._grid_map
        agent_pos = env._agent_location
        target_pos = env._target_location
        
        st.subheader("训练环境")
        fig = plot_grid_world(grid_map, agent_pos, target_pos)
        st.pyplot(fig)
        
        # 实现Q-learning算法
        def q_learning(env, episodes, alpha, gamma, epsilon, epsilon_decay, min_epsilon, progress_bar, status_text):
            # 初始化Q表
            state_size = env.observation_space.n
            action_size = env.action_space.n
            q_table = np.zeros((state_size, action_size))
            
            # 记录每个episode的奖励和步数
            rewards = []
            steps_list = []
            
            for episode in range(episodes):
                # 更新进度条
                progress = (episode + 1) / episodes
                progress_bar.progress(progress)
                status_text.text(f"训练中... Episode {episode+1}/{episodes}")
                
                # 重置环境
                state, _ = env.reset()
                total_reward = 0
                steps = 0
                terminated = False
                truncated = False
                
                # 衰减epsilon
                current_epsilon = max(epsilon * (epsilon_decay ** episode), min_epsilon)
                
                while not (terminated or truncated):
                    # 选择动作（epsilon-greedy策略）
                    if np.random.random() < current_epsilon:
                        action = env.action_space.sample()  # 探索
                    else:
                        action = np.argmax(q_table[state])  # 利用
                    
                    # 执行动作
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    
                    # 更新Q值
                    best_next_action = np.argmax(q_table[next_state])
                    td_target = reward + gamma * q_table[next_state, best_next_action] * (not (terminated or truncated))
                    td_error = td_target - q_table[state, action]
                    q_table[state, action] += alpha * td_error
                    
                    # 更新状态和累积奖励
                    state = next_state
                    total_reward += reward
                    steps += 1
                    
                    # 设置最大步数
                    if steps >= 100:
                        truncated = True
                
                rewards.append(total_reward)
                steps_list.append(steps)
            
            return q_table, rewards, steps_list
        
        # 运行Q-learning
        start_time = time.time()
        q_table, rewards, steps_list = q_learning(env, episodes, alpha, gamma, epsilon, epsilon_decay, min_epsilon, progress_bar, status_text)
        end_time = time.time()
        
        # 显示训练结果
        status_text.text(f"训练完成! 用时: {end_time - start_time:.2f}秒")
        
        # 显示结果
        st.subheader("Q-learning学习结果")
        
        # 展示最终的Q值表
        st.text("最终的Q值表可视化")
        q_value_fig = plot_q_values(q_table, env.size)
        st.pyplot(q_value_fig)
        
        # 添加Q-learning结果解释
        with st.expander("如何解读Q-learning学习结果"):
            st.markdown("""
            ### Q-learning学习结果解读指南
            
            Q-learning学习完成后，可以从多个方面分析结果：
            
            #### 1. Q值表可视化
            
            * **箭头方向**：表示每个状态的最优动作方向
            * **箭头长度**：表示该动作的Q值大小，越长表示Q值越大
            * **路径形成**：观察箭头是否形成了从起点到目标的清晰路径
            * **障碍物影响**：观察智能体是否学会了避开墙壁和陷阱
            
            #### 2. 学习曲线分析
            
            * **上升趋势**：曲线整体呈上升趋势表明智能体学习效果良好
            * **收敛速度**：曲线达到稳定所需的episode数量反映了学习速度
            * **最终性能**：稳定后的奖励水平表示策略的最终质量
            * **波动性**：波动大小反映了策略的稳定性和环境的随机性
            
            #### 3. 算法参数影响
            
            * **学习率(α)**：影响学习速度和稳定性，太大可能导致不收敛，太小可能学习太慢
            * **折扣因子(γ)**：控制短期与长期奖励的平衡，越接近1越重视长期奖励
            * **探索率(ε)**：控制探索与利用的平衡，影响最终策略的质量
            * **ε衰减速率**：控制从探索转向利用的速度，影响学习过程中的策略演变
            
            #### 4. 不同环境下的表现差异
            
            * **简单环境**：通常收敛较快，Q值分布清晰
            * **迷宫环境**：需要更多探索才能找到路径，Q值分布可能更复杂
            * **带陷阱环境**：负奖励区域会形成明显的"避开"模式
            * **随机环境**：每次训练结果可能有差异，体现算法的鲁棒性
            
            通过综合分析这些因素，可以更全面地理解Q-learning算法的学习过程和结果。
            """)
        
        # 展示学习曲线
        st.text("学习曲线")
        learning_curve_fig = plot_learning_curve(rewards, window=20)
        st.pyplot(learning_curve_fig)
        
        # 绘制步数变化
        st.text("每个Episode的步数")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(1, episodes+1), steps_list, 'b-', alpha=0.6)
        
        # 添加移动平均线
        window = min(10, len(steps_list))
        steps_moving_avg = np.convolve(steps_list, np.ones(window)/window, mode='valid')
        ax.plot(range(window, episodes+1), steps_moving_avg, 'r-', linewidth=2)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Steps')
        ax.set_title('Steps per Episode')
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)
        
        # 添加步数变化解释
        with st.expander("如何理解步数变化"):
            st.markdown("""
            ### 步数变化解读指南
            
            步数变化图展示了智能体在每个训练回合(episode)中完成任务所需的步数，是评估策略效率的重要指标：
            
            #### 关键特征：
            
            * **蓝线**：展示每个episode的实际步数，波动反映了环境的随机性和策略的探索
            * **红线**：移动平均线，反映整体趋势，平滑了短期波动
            
            #### 典型模式：
            
            * **下降趋势**：步数逐渐减少表明智能体学到了更高效的策略
            * **稳定阶段**：曲线最终趋于平稳表示策略已经收敛
            * **突然下降**：可能表示智能体发现了新的捷径或更优策略
            * **周期性波动**：可能表明智能体在多个次优策略之间摇摆
            
            #### 与奖励曲线的关系：
            
            * 通常步数减少会伴随着奖励增加
            * 在有负奖励（如每步惩罚）的环境中，步数与奖励的关系尤为紧密
            * 两个图表结合分析可提供更全面的学习过程评估
            
            通过观察步数变化，可以判断智能体是否学会了高效完成任务，以及策略优化的程度。
            """)
        
        # 测试学习到的策略
        st.subheader("测试学习到的策略")
        
        # 创建一个新的环境实例用于测试
        test_env = GridWorldEnv(render_mode=None)
        test_env._grid_map = grid_map.copy()
        test_env._init_locations()
        
        test_episodes = 10
        test_rewards = []
        test_steps = []
        success_count = 0
        
        for _ in range(test_episodes):
            state, _ = test_env.reset()
            total_reward = 0
            steps = 0
            terminated = False
            truncated = False
            
            while not (terminated or truncated):
                # 使用学习到的策略（贪婪策略）
                action = np.argmax(q_table[state])
                
                # 执行动作
                state, reward, terminated, truncated, _ = test_env.step(action)
                
                total_reward += reward
                steps += 1
                
                # 设置最大步数
                if steps >= 100:
                    truncated = True
            
            test_rewards.append(total_reward)
            test_steps.append(steps)
            
            if total_reward > 0:  # 成功到达目标
                success_count += 1
        
        # 显示测试结果
        st.markdown(f"""
        **测试结果**:
        - 平均步数: {np.mean(test_steps):.2f}
        - 平均奖励: {np.mean(test_rewards):.2f}
        - 成功率: {success_count / test_episodes:.2f}
        """)
        
        # 显示智能体策略
        st.subheader("智能体策略可视化")
        
        # 创建策略矩阵
        policy = np.zeros((env.size, env.size), dtype=int)
        
        for i in range(env.size):
            for j in range(env.size):
                state = i * env.size + j
                policy[i, j] = np.argmax(q_table[state])
        
        # 绘制策略
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 绘制网格地图
        cmap = plt.cm.colors.ListedColormap(['white', 'black', 'gold', 'red'])
        ax.imshow(grid_map, cmap=cmap)
        
        # 添加网格线
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
        ax.set_xticks(np.arange(-0.5, env.size, 1))
        ax.set_yticks(np.arange(-0.5, env.size, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        # 绘制策略箭头
        for i in range(env.size):
            for j in range(env.size):
                if grid_map[i, j] in [1, 2, 3]:  # 跳过墙壁、目标和陷阱
                    continue
                
                action = policy[i, j]
                
                # 箭头方向
                if action == 0:  # 上
                    dx, dy = 0, -0.4
                elif action == 1:  # 右
                    dx, dy = 0.4, 0
                elif action == 2:  # 下
                    dx, dy = 0, 0.4
                elif action == 3:  # 左
                    dx, dy = -0.4, 0
                
                ax.arrow(j, i, dx, dy, head_width=0.2, head_length=0.2, fc='blue', ec='blue')
        
        ax.set_title("智能体策略 (箭头表示最优动作方向)")
        st.pyplot(fig)
    
    # 进阶练习2：复杂Grid World设计
    st.subheader("进阶练习2：复杂Grid World设计")
    
    st.markdown("""
    **目标**：设计一个更复杂的Grid World环境，增加额外的元素和机制。
    
    **扩展思路**：
    
    1. **多目标环境**：
       - 添加多个具有不同奖励值的目标
       - 设计任务顺序（需要先收集某些目标才能获得最终奖励）
    
    2. **动态障碍物**：
       - 实现移动的障碍物，增加环境的复杂性
       - 添加随机出现或消失的障碍物
    
    3. **钥匙和门机制**：
       - 智能体需要先拾取钥匙，才能开门通过
       - 多个钥匙对应不同的门
    
    4. **能量管理**：
       - 添加能量限制，每一步消耗能量
       - 放置可以恢复能量的物品
    
    5. **风场效果**：
       - 某些区域有"风"，会使智能体额外移动一步
       - 不同区域风向不同
    
    **实现示例**：
    ```python
    class ComplexGridWorld(gym.Env):
        def __init__(self, size=8):
            super().__init__()
            self.size = size
            self.action_space = spaces.Discrete(4)
            self.observation_space = spaces.Dict({
                'position': spaces.Box(0, size-1, shape=(2,), dtype=int),
                'keys': spaces.MultiBinary(3),  # 3种钥匙
                'energy': spaces.Box(0, 100, shape=(1,), dtype=float)
            })
            
            # 初始化网格地图
            self.grid_map = self._create_complex_map()
            
            # 初始化智能体状态
            self.agent_pos = np.array([0, 0])
            self.keys = np.zeros(3, dtype=int)
            self.energy = 100.0
            
        def _create_complex_map(self):
            # 创建复杂地图
            grid_map = np.zeros((self.size, self.size), dtype=int)
            
            # 添加墙壁、目标、陷阱、钥匙、门、能量等
            # ...
            
            return grid_map
            
        def step(self, action):
            # 执行动作
            # 处理特殊区域效果（如风场）
            # 更新智能体状态（位置、钥匙、能量等）
            # 计算奖励
            # ...
            
            return observation, reward, terminated, truncated, info
    ```
    
    **设计考虑**：
    
    1. **观测空间**：需要扩展以包含额外的状态信息
    2. **状态表示**：如何有效地表示复杂环境中的状态？
    3. **奖励设计**：如何设计奖励函数引导智能体完成复杂任务？
    4. **学习策略**：更复杂的环境可能需要更高级的强化学习算法
    """)
    
    # 多目标Grid World演示
    st.subheader("多目标Grid World演示")
    
    if st.button("生成多目标环境", key="generate_multi_goal"):
        # 创建一个8x8的多目标环境
        size = 8
        multi_goal_map = np.zeros((size, size), dtype=int)
        
        # 添加墙壁
        wall_positions = [(1, 1), (1, 2), (1, 3), (1, 5), (1, 6), 
                          (3, 1), (3, 3), (3, 5), (3, 6), 
                          (5, 1), (5, 2), (5, 3), (5, 5)]
        
        for i, j in wall_positions:
            multi_goal_map[i, j] = 1
        
        # 添加多个目标
        goal_positions = [(2, 6), (6, 2), (6, 6)]
        goal_values = [0.5, 0.7, 1.0]  # 不同目标具有不同奖励值
        
        for idx, (i, j) in enumerate(goal_positions):
            multi_goal_map[i, j] = 2  # 所有目标使用相同的值，但可以用额外的数据结构存储奖励值
        
        # 添加陷阱
        trap_positions = [(2, 2), (4, 4), (6, 4)]
        for i, j in trap_positions:
            multi_goal_map[i, j] = 3
        
        # 设置智能体初始位置
        agent_pos = np.array([0, 0])
        
        # 可视化环境
        fig = plot_grid_world(multi_goal_map, agent_pos)
        st.pyplot(fig)
        
        # 显示环境说明
        st.markdown("""
        **多目标环境说明**:
        
        - 智能体起始位置位于左上角 (0,0)
        - 有三个目标（黄色单元格），具有不同的奖励值:
          - 目标1: 奖励 0.5
          - 目标2: 奖励 0.7
          - 目标3: 奖励 1.0
        - 陷阱（红色单元格）会给予负奖励
        - 墙壁（黑色单元格）不可穿越
        
        **学习任务**:
        
        1. 简单任务：到达任意一个目标即可获得相应奖励
        2. 进阶任务：必须按照奖励值从低到高的顺序访问目标
        3. 高级任务：在有限步数内最大化总奖励（需要优化路径）
        """)
        
        # 添加多目标环境解释
        with st.expander("多目标环境和智能体学习挑战"):
            st.markdown("""
            ### 多目标强化学习环境
            
            多目标环境是研究复杂决策问题的重要工具，它比单目标环境更贴近现实世界的挑战。
            
            #### 环境特性与智能体面临的挑战
            
            1. **状态表示复杂性**
               - 在多目标环境中，状态需要包含更多信息，如已访问的目标、目标价值、当前位置等
               - 状态空间显著增大，可能需要更高效的表示方法（如函数逼近）
            
            2. **探索-利用困境**
               - 智能体需要平衡短期奖励（就近目标）和长期收益（高价值远距离目标）
               - 最优路径规划变得更加复杂，可能需要更复杂的探索策略
            
            3. **信用分配问题**
               - 确定哪些行为导致获得高奖励变得更加困难
               - 延迟奖励使得Q值更新更加困难
            
            #### 学习方法适应
            
            1. **基本Q-learning的局限性**
               - 在复杂多目标环境中，标准Q-learning可能收敛缓慢
               - 可能需要调整学习参数（较低的学习率、较高的探索率）
            
            2. **适应性解决方案**
               - **分层强化学习**：将任务分解为访问不同目标的子任务
               - **内在动机**：添加好奇心驱动的额外奖励以促进探索
               - **记忆增强**：使用经验回放或优先经验回放提高样本效率
            
            #### 实际应用场景
            
            多目标环境模拟了许多现实问题，如：
            - 机器人在工厂中完成多步骤任务
            - 游戏AI需要收集多种资源并完成多个任务
            - 自动驾驶车辆在城市环境中规划路径，兼顾时间、安全和舒适性
            
            通过研究多目标环境中的强化学习，可以开发出更适用于复杂现实问题的智能决策系统。
            """)
    
    # 进阶练习3：Q-learning与动态规划比较
    st.subheader("进阶练习3：Q-learning与动态规划比较")
    
    st.markdown("""
    **目标**：实现价值迭代（动态规划方法）并与Q-learning（强化学习方法）进行比较。
    
    **价值迭代算法**：
    
    价值迭代是一种动态规划方法，它通过不断更新状态价值函数来找到最优策略。
    
    ```python
    def value_iteration(env, theta=0.0001, gamma=0.99):
        # 初始化价值函数
        V = np.zeros(env.observation_space.n)
        
        while True:
            delta = 0
            
            # 对每个状态更新价值函数
            for s in range(env.observation_space.n):
                v = V[s]
                
                # 计算在此状态下所有动作的价值
                action_values = []
                
                for a in range(env.action_space.n):
                    next_s, reward, done = env_model(s, a)  # 环境模型
                    action_values.append(reward + gamma * V[next_s] * (not done))
                
                # 更新状态价值为最大动作价值
                V[s] = max(action_values)
                
                # 计算变化量
                delta = max(delta, abs(v - V[s]))
            
            # 收敛检查
            if delta < theta:
                break
        
        # 从价值函数导出策略
        policy = np.zeros(env.observation_space.n, dtype=int)
        
        for s in range(env.observation_space.n):
            action_values = []
            
            for a in range(env.action_space.n):
                next_s, reward, done = env_model(s, a)
                action_values.append(reward + gamma * V[next_s] * (not done))
            
            policy[s] = np.argmax(action_values)
        
        return policy, V
    ```
    
    **比较分析**：
    
    1. **前提条件**：
       - 动态规划方法需要完整的环境模型（转移概率和奖励函数）
       - 强化学习方法可以通过与环境交互直接学习，不需要环境模型
    
    2. **收敛速度**：
       - 动态规划在小型环境中通常收敛更快
       - 强化学习在大型环境中更实用，但可能需要更多的交互次数
    
    3. **适用场景**：
       - 动态规划：环境模型已知且计算资源足够
       - 强化学习：环境模型未知或过于复杂
    
    **思考题**：
    
    - 在什么情况下Q-learning比价值迭代更有优势？
    - 如何将模型的先验知识整合到强化学习算法中？
    - 探索与利用的平衡在Q-learning中有何作用？在价值迭代中有类似概念吗？
    """)
    
    # 进阶练习4：结合神经网络的强化学习（DQN简介）
    st.subheader("进阶练习4：结合神经网络的强化学习")
    
    st.markdown("""
    **目标**：了解如何将Q-learning与神经网络结合，实现深度Q网络（DQN）。
    
    **DQN简介**：
    
    当状态空间或动作空间很大时，使用表格表示Q值变得不切实际。深度Q网络（DQN）使用神经网络来近似Q函数。
    
    **关键创新**：
    
    1. **经验回放**：存储和重用过去的经验，打破样本之间的相关性
    2. **目标网络**：使用单独的目标网络来计算TD目标，提高学习稳定性
    
    **基本框架**：
    
    ```python
    import tensorflow as tf
    
    # 构建Q网络
    def build_q_network(state_size, action_size):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(state_size,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(action_size)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                     loss='mse')
        return model
    
    # 经验回放缓冲区
    class ReplayBuffer:
        def __init__(self, capacity=10000):
            self.buffer = []
            self.capacity = capacity
            self.position = 0
            
        def add(self, state, action, reward, next_state, done):
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
            self.buffer[self.position] = (state, action, reward, next_state, done)
            self.position = (self.position + 1) % self.capacity
            
        def sample(self, batch_size):
            batch = random.sample(self.buffer, batch_size)
            states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
            return states, actions, rewards, next_states, dones
            
        def __len__(self):
            return len(self.buffer)
    
    # DQN算法
    def dqn(env, episodes=500):
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        
        # 创建Q网络和目标网络
        q_network = build_q_network(state_size, action_size)
        target_network = build_q_network(state_size, action_size)
        target_network.set_weights(q_network.get_weights())
        
        # 创建经验回放缓冲区
        replay_buffer = ReplayBuffer()
        
        # 训练参数
        batch_size = 32
        gamma = 0.99
        epsilon = 1.0
        epsilon_min = 0.01
        epsilon_decay = 0.995
        update_target_every = 100
        
        for episode in range(episodes):
            state, _ = env.reset()
            state = np.reshape(state, [1, state_size])
            total_reward = 0
            
            while True:
                # 选择动作
                if np.random.rand() <= epsilon:
                    action = env.action_space.sample()
                else:
                    q_values = q_network.predict(state, verbose=0)
                    action = np.argmax(q_values[0])
                
                # 执行动作
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                next_state = np.reshape(next_state, [1, state_size])
                
                # 存储经验
                replay_buffer.add(state[0], action, reward, next_state[0], done)
                
                # 从经验回放缓冲区采样并训练
                if len(replay_buffer) > batch_size:
                    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                    
                    targets = q_network.predict(states, verbose=0)
                    next_q_values = target_network.predict(next_states, verbose=0)
                    
                    for i in range(batch_size):
                        if dones[i]:
                            targets[i][actions[i]] = rewards[i]
                        else:
                            targets[i][actions[i]] = rewards[i] + gamma * np.max(next_q_values[i])
                    
                    q_network.fit(states, targets, epochs=1, verbose=0)
                
                # 更新状态和累积奖励
                state = next_state
                total_reward += reward
                
                # 在每episode结束时更新目标网络
                if done:
                    if episode % update_target_every == 0:
                        target_network.set_weights(q_network.get_weights())
                    break
            
            # 衰减epsilon
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay
    ```
    
    **DQN与传统Q-learning的比较**：
    
    1. **表示能力**：DQN可以处理连续或高维状态空间
    2. **泛化能力**：神经网络能够泛化到未见过的状态
    3. **样本效率**：经验回放提高了样本利用效率
    4. **稳定性**：目标网络增强了训练稳定性
    
    **思考题**：
    
    - 在什么情况下应该考虑使用DQN而不是表格Q-learning？
    - 经验回放和目标网络各自解决了什么问题？
    - DQN还存在哪些局限性？如何进一步改进？
    """)
    
    # 进阶练习总结
    st.subheader("进阶练习总结")
    
    st.markdown("""
    通过这些进阶练习，你应该对以下内容有了更深入的理解：
    
    1. **Q-learning算法**的原理、实现和调参技巧
    2. 如何设计**复杂的强化学习环境**，增加额外元素和机制
    3. **动态规划**和**强化学习**方法的比较
    4. **深度强化学习**的基本概念和框架
    
    **后续学习建议**：
    
    1. 深入学习其他强化学习算法，如SARSA、Actor-Critic等
    2. 探索策略梯度方法，特别是在连续动作空间中
    3. 研究更复杂的深度强化学习算法，如PPO、DDPG、SAC等
    4. 将强化学习应用于实际问题，如机器人控制、游戏AI等
    
    继续探索和实践，强化学习是一个深度和广度都极为丰富的领域！
    """)

if __name__ == "__main__":
    show()