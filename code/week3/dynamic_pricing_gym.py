import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class DynamicPricingGymEnv(gym.Env):
    """
    动态定价环境，实现了强化学习Gymnasium接口，综合考虑市场状态、库存管理和定价决策。
    
    该环境模拟了一个电子商务平台的动态定价场景，包含以下核心组件：
    - 状态空间：市场状态（从低迷到火爆）和库存水平的组合
    - 动作空间：可选择的价格水平
    - 转移动态：市场状态随时间的变化和库存随销售/补货的变化
    - 奖励函数：销售产生的利润
    
    该环境结合了价格弹性需求模型，库存管理的成本与约束，以及市场状态变化的不确定性，
    为学习最优定价策略提供了一个综合性的场景。
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, config=None):
        """
        初始化动态定价环境。
        
        参数:
        - config (dict): 环境配置参数，包含以下键值对：
            - base_price (float): 产品基础价格，默认为100
            - cost_price (float): 产品成本价格，默认为70
            - price_range (float): 价格上下浮动范围（百分比），默认为0.3
            - price_levels (int): 可选价格水平数量，默认为5
            - elasticity (float): 价格弹性系数，默认为1.5
            - noise (float): 需求随机噪声幅度，默认为0.2
            - market_transition_stability (float): 市场状态稳定性，即保持当前状态的概率，默认为0.6
            - market_transition_jump_prob (float): 市场状态跳变概率，允许非相邻状态间的跳转，默认为0.05
            - stockout_penalty (float): 缺货惩罚系数，默认为50
            - holding_cost_rate (float): 库存持有成本率，默认为0.01
            - initial_inventory (int): 初始库存，默认为500
            - max_inventory (int): 最大库存，默认为1000
            - inventory_init_range (tuple): 初始库存的随机范围（相对于最大库存的比例），默认为(0.2, 0.9)
            - restock_threshold (float): 补货阈值（相对于最大库存的比例），默认为0.2
            - restock_amount (int): 补货数量，默认为200
            - restock_randomness (float): 补货数量的随机波动范围，默认为0.3
            - max_steps (int): 每个回合的最大步数，默认为365
        """
        super().__init__()
        
        # 设置默认配置
        self.default_config = {
            'base_price': 100.0,
            'cost_price': 70.0,
            'price_range': 0.3,
            'price_levels': 5,
            'elasticity': 1.5,
            'noise': 0.2,
            'market_transition_stability': 0.6,
            'market_transition_jump_prob': 0.05,
            'stockout_penalty': 50.0, 
            'holding_cost_rate': 0.01,
            'initial_inventory': 500,
            'max_inventory': 1000,
            'inventory_init_range': (0.2, 0.9),
            'restock_threshold': 0.2,
            'restock_amount': 200,
            'restock_randomness': 0.3,
            'max_steps': 365
        }
        
        # 用传入的配置覆盖默认配置
        self.config = self.default_config.copy()
        if config is not None:
            self.config.update(config)
        
        # 定价相关属性
        self.base_price = self.config['base_price']
        self.cost_price = self.config['cost_price']
        self.price_range = self.config['price_range']
        self.price_levels = self.config['price_levels']
        self.elasticity = self.config['elasticity']
        self.noise = self.config['noise']
        
        # 分档价格作为动作空间
        price_step = 2 * self.price_range / (self.price_levels - 1) if self.price_levels > 1 else 0
        self.price_actions = [round(self.base_price * (1 - self.price_range + i * price_step), 2) 
                             for i in range(self.price_levels)]
        self.action_space = spaces.Discrete(self.price_levels)
        
        # 市场状态
        self.market_states = ["极度低迷", "低迷", "平稳", "活跃", "火爆"]
        self.n_market_states = len(self.market_states)
        
        # 库存相关属性
        self.max_inventory = self.config['max_inventory']
        self.initial_inventory = self.config['initial_inventory']
        self.inventory_init_range = self.config['inventory_init_range']
        self.restock_threshold = self.config['restock_threshold']
        self.restock_amount = self.config['restock_amount']
        self.restock_randomness = self.config['restock_randomness']
        self.stockout_penalty = self.config['stockout_penalty']
        self.holding_cost_rate = self.config['holding_cost_rate']
        
        # 库存水平（离散化为10个水平）
        self.n_inventory_levels = 10
        self.inventory_discretization = self.max_inventory / self.n_inventory_levels
        self.inventory_levels = [int(i * self.inventory_discretization) for i in range(self.n_inventory_levels+1)]
        
        # 观察空间：市场状态和库存水平
        self.observation_space = spaces.Dict({
            "market_state": spaces.Discrete(self.n_market_states),
            "inventory_level": spaces.Discrete(self.n_inventory_levels)
        })
        
        # 市场状态转移矩阵
        self.market_transition_stability = self.config['market_transition_stability']
        self.market_transition_jump_prob = self.config['market_transition_jump_prob']
        self.market_transition_matrix = self._create_enhanced_transition_matrix()
        
        # 回合相关属性
        self.max_steps = self.config['max_steps']
        self.current_step = 0
        self.current_market_state = None
        self.current_inventory = None
        self.current_price = None
        
        # 渲染模式
        self.render_mode = None
        self.window = None
        self.clock = None
        
        # 初始化状态
        self.reset()
    
    def _create_enhanced_transition_matrix(self):
        """
        创建增强版状态转移矩阵，支持非相邻状态间的跳转。
        
        该矩阵基于以下原则：
        1. 主对角线：当前状态保持不变的概率 (stability)
        2. 次对角线：向相邻状态转移的概率
        3. 其他位置：非相邻状态间跳变的概率，通过jump_prob控制
        
        返回:
            np.ndarray: 形状为(n_market_states, n_market_states)的转移概率矩阵
        """
        n = self.n_market_states
        stability = self.market_transition_stability
        jump_prob = self.market_transition_jump_prob
        
        # 初始化矩阵
        matrix = np.zeros((n, n))
        
        # 设置保持当前状态的概率（对角线）
        np.fill_diagonal(matrix, stability)
        
        # 计算相邻转移的概率
        adjacent_prob = (1.0 - stability - jump_prob * (n - 3)) / 2 if n > 2 else (1.0 - stability)
        
        # 设置相邻状态转移概率
        for i in range(n):
            if i > 0:  # 可以向左转移
                matrix[i, i-1] = adjacent_prob
            if i < n-1:  # 可以向右转移
                matrix[i, i+1] = adjacent_prob
        
        # 设置非相邻状态的跳变概率
        for i in range(n):
            for j in range(n):
                if abs(i - j) > 1:  # 非相邻状态
                    matrix[i, j] = jump_prob
        
        # 处理边界情况（第一行和最后一行）
        # 第一行：向右的概率更高
        if n > 1:
            matrix[0, 1] = 1.0 - stability - jump_prob * (n - 2)
            # 最后一行：向左的概率更高
            matrix[n-1, n-2] = 1.0 - stability - jump_prob * (n - 2)
        
        # 确保每行概率和为1
        row_sums = matrix.sum(axis=1)
        for i in range(n):
            if row_sums[i] != 1.0:
                # 调整对角线元素使总和为1
                matrix[i, i] += (1.0 - row_sums[i])
        
        return matrix
    
    def get_state_name(self):
        """获取当前状态的可读名称"""
        market = self.market_states[self.current_market_state]
        inventory = self.current_inventory
        return f"市场:{market}, 库存:{inventory}"
    
    def get_current_state(self):
        """获取当前状态作为观察空间的格式"""
        inventory_level = min(int(self.current_inventory / self.inventory_discretization), 
                              self.n_inventory_levels - 1)
        return {
            "market_state": self.current_market_state,
            "inventory_level": inventory_level
        }
    
    def get_state_index(self):
        """获取用于Q学习的状态索引（市场状态，库存水平）"""
        inventory_level = min(int(self.current_inventory / self.inventory_discretization), 
                              self.n_inventory_levels - 1)
        return (self.current_market_state, inventory_level)
    
    def reset(self, seed=None, options=None):
        """
        重置环境到初始状态。
        
        参数:
            seed (int, optional): 随机种子
            options (dict, optional): 额外的重置选项
            
        返回:
            tuple: (observation, info)
        """
        # 设置随机种子
        super().reset(seed=seed)
        
        # 随机初始市场状态
        self.current_market_state = self.np_random.integers(0, self.n_market_states)
        
        # 随机初始库存（在指定范围内）
        if self.inventory_init_range:
            min_ratio, max_ratio = self.inventory_init_range
            init_ratio = self.np_random.uniform(min_ratio, max_ratio)
            self.current_inventory = int(self.max_inventory * init_ratio)
        else:
            self.current_inventory = self.initial_inventory
        
        # 重置步数计数器
        self.current_step = 0
        
        # 初始观察
        observation = self.get_current_state()
        info = {"state_name": self.get_state_name()}
        
        return observation, info
    
    def step(self, action):
        """
        执行给定动作并更新环境状态。
        
        参数:
            action (int): 价格动作索引
            
        返回:
            tuple: (observation, reward, terminated, truncated, info)
        """
        assert self.action_space.contains(action), f"动作 {action} 不在动作空间中"
        
        # 获取所选价格
        self.current_price = self.price_actions[action]
        
        # 基于价格弹性和随机因素计算需求
        base_demand = 100  # 基础需求量
        price_ratio = self.current_price / self.base_price
        
        # 市场状态对基础需求的影响 (从低迷到火爆)
        market_factor = 0.6 + 0.2 * self.current_market_state  # 范围: [0.6, 1.4]
        
        # 计算价格弹性影响和随机波动
        demand = base_demand * market_factor * (price_ratio ** -self.elasticity)
        demand *= (1 + self.np_random.normal(0, self.noise))  # 添加随机波动
        demand = max(0, int(demand))  # 确保非负整数
        
        # 计算实际销售量（受库存限制）
        sales = min(demand, self.current_inventory)
        
        # 计算利润
        profit = sales * (self.current_price - self.cost_price)
        
        # 缺货惩罚
        unsatisfied_demand = max(0, demand - sales)
        stockout_cost = unsatisfied_demand * self.stockout_penalty
        
        # 库存持有成本
        holding_cost = self.current_inventory * self.cost_price * self.holding_cost_rate
        
        # 计算净利润（作为奖励）
        reward = profit - stockout_cost - holding_cost
        
        # 更新库存
        self.current_inventory -= sales
        
        # 检查是否需要补货
        if self.current_inventory < self.max_inventory * self.restock_threshold:
            # 添加补货随机波动
            if self.restock_randomness > 0:
                random_factor = 1 + self.np_random.uniform(-self.restock_randomness, self.restock_randomness)
                restock_amount = int(self.restock_amount * random_factor)
            else:
                restock_amount = self.restock_amount
                
            # 确保不超过最大库存
            self.current_inventory = min(self.current_inventory + restock_amount, self.max_inventory)
        
        # 更新市场状态
        next_state_probs = self.market_transition_matrix[self.current_market_state]
        self.current_market_state = self.np_random.choice(self.n_market_states, p=next_state_probs)
        
        # 增加步数
        self.current_step += 1
        
        # 检查是否结束
        terminated = False  # 在这个环境中没有自然终止条件
        truncated = self.current_step >= self.max_steps
        
        # 新的观察
        observation = self.get_current_state()
        
        # 准备信息字典
        info = {
            "state_name": self.get_state_name(),
            "price": self.current_price,
            "demand": demand,
            "sales": sales,
            "profit": profit,
            "stockout_cost": stockout_cost,
            "holding_cost": holding_cost,
            "net_profit": reward,
            "inventory": self.current_inventory
        }
        
        return observation, reward, terminated, truncated, info

def q_learning(env, num_episodes, learning_rate=0.1, discount_factor=0.9, 
               initial_epsilon=0.7, epsilon_decay=0.995, min_epsilon=0.05,
               optimistic_init=True, init_value=50.0, periodic_explore=True,
               periodic_interval=100):
    """
    使用Q学习算法训练动态定价环境的策略。
    
    参数:
        env (DynamicPricingGymEnv): 动态定价环境实例
        num_episodes (int): 训练轮数
        learning_rate (float): 学习率，控制更新步长
        discount_factor (float): 折扣因子，控制对未来奖励的重视程度
        initial_epsilon (float): 初始探索率
        epsilon_decay (float): 探索率的衰减系数
        min_epsilon (float): 最小探索率
        optimistic_init (bool): 是否使用乐观初始化
        init_value (float): 乐观初始化的Q值
        periodic_explore (bool): 是否周期性进行完全随机探索
        periodic_interval (int): 周期性探索的间隔轮数
        
    返回:
        tuple: (q_table, history, visit_stats)
    """
    # 初始化Q表
    if optimistic_init:
        # 乐观初始化，鼓励探索
        q_table = np.ones((env.n_market_states, env.n_inventory_levels, env.action_space.n)) * init_value
    else:
        # 默认初始化为零
        q_table = np.zeros((env.n_market_states, env.n_inventory_levels, env.action_space.n))
    
    # 初始化访问计数表
    visit_counts = np.zeros((env.n_market_states, env.n_inventory_levels, env.action_space.n))
    
    # 记录训练历史
    history = []
    
    # 保存所有访问过的状态
    all_visited_states = set()
    
    # 训练循环
    epsilon = initial_epsilon
    
    for episode in range(num_episodes):
        # 是否进行周期性强制探索
        force_explore = periodic_explore and (episode % periodic_interval == 0)
        
        # 重置环境
        observation, info = env.reset()
        state = (observation["market_state"], observation["inventory_level"])
        
        # 记录每个回合的数据
        episode_reward = 0
        episode_visits = []  # 记录访问过的状态-动作对
        
        # 记录是否已经计入全局状态集
        all_visited_states.add(state)
        
        done = False
        truncated = False
        
        while not (done or truncated):
            # 选择动作（epsilon-贪婪策略）
            if force_explore or np.random.random() < epsilon:
                # 探索：随机选择动作
                action = env.action_space.sample()
            else:
                # 利用：选择Q值最大的动作
                state_q_values = q_table[state[0], state[1], :]
                action = np.argmax(state_q_values)
            
            # 记录这次的状态-动作对
            episode_visits.append((state[0], state[1], action))
            
            # 更新访问计数
            visit_counts[state[0], state[1], action] += 1
            
            # 执行动作
            next_observation, reward, done, truncated, info = env.step(action)
            next_state = (next_observation["market_state"], next_observation["inventory_level"])
            
            # 添加到访问过的状态集
            all_visited_states.add(next_state)
            
            # 更新Q值（使用动态学习率）
            visits = visit_counts[state[0], state[1], action]
            dynamic_lr = learning_rate / (1 + 0.1 * visits**0.5)  # 随访问次数衰减
            
            # 标准Q学习更新公式
            best_next_action = np.argmax(q_table[next_state[0], next_state[1], :])
            q_table[state[0], state[1], action] += dynamic_lr * (
                reward + discount_factor * q_table[next_state[0], next_state[1], best_next_action] - 
                q_table[state[0], state[1], action]
            )
            
            # 更新状态和累积奖励
            state = next_state
            episode_reward += reward
        
        # 计算这一回合的状态空间覆盖率
        visited_states = len(np.unique([(s, i) for s, i, _ in episode_visits]))
        total_states = env.n_market_states * env.n_inventory_levels
        episode_coverage_rate = visited_states / total_states
        
        # 计算累积的状态空间覆盖率
        cumulative_coverage_rate = len(all_visited_states) / total_states
        
        # 衰减探索率
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        # 记录本回合历史
        history.append({
            "episode": episode,
            "total_reward": episode_reward,
            "epsilon": epsilon,
            "episode_coverage_rate": episode_coverage_rate,
            "cumulative_coverage_rate": cumulative_coverage_rate
        })
    
    # 计算访问统计数据
    total_visits = np.sum(visit_counts)
    zero_visit_pairs = np.sum(visit_counts == 0)
    non_zero_pairs = total_visits - zero_visit_pairs
    max_visits = np.max(visit_counts)
    min_visits = np.min(visit_counts[visit_counts > 0]) if np.any(visit_counts > 0) else 0
    
    visit_stats = {
        "total_visits": int(total_visits),
        "zero_visit_pairs": int(zero_visit_pairs),
        "non_zero_pairs": int(non_zero_pairs),
        "max_visits": int(max_visits),
        "min_visits": int(min_visits),
        "final_coverage_rate": len(all_visited_states) / total_states,
        "all_visited_states": all_visited_states
    }
    
    return q_table, history, visit_stats

if __name__ == "__main__":
    """测试环境和算法"""
    # 创建环境
    env = DynamicPricingGymEnv()
    
    # 训练模型
    q_table, history, visit_stats = q_learning(
        env, 
        num_episodes=100, 
        learning_rate=0.1, 
        discount_factor=0.9, 
        initial_epsilon=0.7,
        optimistic_init=True,  # 使用乐观初始化
        init_value=50.0,        # 设置较高的初始值促进探索
        periodic_explore=True  # 周期性进行完全随机探索
    )
    
    # 打印历史记录
    history_df = pd.DataFrame(history)
    print(history_df.head())
    
    # 打印访问统计
    print(f"状态-动作对总访问次数: {visit_stats['total_visits']}")
    print(f"最大访问次数: {visit_stats['max_visits']}")
    print(f"最小访问次数: {visit_stats['min_visits']}")
    print(f"未访问的状态-动作对数量: {visit_stats['zero_visit_pairs']}")
    
    # 测试学习到的策略
    observation, _ = env.reset(seed=42)
    done = False
    total_reward = 0
    
    while not done:
        # 获取当前状态
        market_state = observation['market_state']
        inventory_level = observation['inventory_level']
        
        # 选择动作
        action = np.argmax(q_table[market_state, inventory_level])
        
        # 执行动作
        observation, reward, terminated, truncated, info = env.step(action)
        
        # 累计奖励
        total_reward += reward
        
        # 渲染
        env.render()
        
        # 检查是否结束
        done = terminated or truncated
    
    print(f"总奖励: {total_reward:.2f}") 