import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
import torch
import os

from models.base_dqn import DQN
from models.double_dqn import DoubleDQN
from models.dueling_dqn import DuelingDQN
from models.per_dqn import PERDQN
from environment.blackjack_env import BlackjackWrapper
from utils.visualization import render_blackjack_state, plot_training_results, visualize_q_values
from utils.training import train_agent, evaluate_agent

# 设置页面
st.set_page_config(page_title="深入探索DQN——改进、调优与实战", layout="wide")

# 标题和介绍
st.title("深入探索DQN——改进、调优与实战")
st.markdown("""
本交互式课件将演示DQN算法及其改进方法在21点环境中的应用。通过比较不同算法的性能，
帮助你理解各种改进的原理和效果。
""")

# 21点环境介绍
st.header("21点环境介绍")
st.markdown("""
21点(Blackjack)是一个经典的卡牌游戏环境，玩家的目标是使手牌点数接近但不超过21点。
在这个环境中：
- **状态**: 玩家当前点数、庄家明牌点数、是否持有可作为1点或11点的A
- **动作**: 0(停牌)或1(要牌)
- **奖励**: 赢得游戏+1，平局0，输掉游戏-1
""")

# 可视化21点环境
with st.expander("21点环境可视化", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("环境示例")
        env = gym.make('Blackjack-v1')
        state, _ = env.reset()
        render_blackjack_state(state)
    
    with col2:
        st.subheader("随机策略演示")
        if st.button("随机玩一局"):
            env = gym.make('Blackjack-v1')
            state, _ = env.reset()
            render_blackjack_state(state)
            
            st.write("**游戏过程:**")
            done = False
            total_reward = 0
            
            while not done:
                action = np.random.choice([0, 1])  # 随机选择动作
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward
                
                st.write(f"动作: {'要牌' if action == 1 else '停牌'}")
                st.write(f"新状态: 玩家点数={next_state[0]}, 庄家明牌={next_state[1]}, 可用A={'是' if next_state[2] else '否'}")
                
                if done:
                    if reward > 0:
                        st.success(f"游戏结束，玩家获胜! 奖励: {reward}")
                    elif reward < 0:
                        st.error(f"游戏结束，玩家失败! 奖励: {reward}")
                    else:
                        st.info(f"游戏结束，平局! 奖励: {reward}")
                
                state = next_state

# DQN算法选择
st.header("DQN算法训练")

# 模型选择和参数设置
col1, col2 = st.columns(2)

with col1:
    algorithm = st.selectbox(
        "选择算法",
        ["基础DQN", "Double DQN", "Dueling DQN", "优先经验回放DQN"]
    )
    
    num_episodes = st.slider("训练回合数", 100, 5000, 1000, 100)
    
    # 高级参数（可折叠）
    with st.expander("高级参数"):
        learning_rate = st.slider("学习率", 1e-5, 1e-2, 1e-3, format="%.5f")
        gamma = st.slider("折扣因子", 0.8, 0.999, 0.99, 0.001)
        epsilon_start = st.slider("初始探索率", 0.5, 1.0, 1.0, 0.01)
        epsilon_end = st.slider("最终探索率", 0.01, 0.5, 0.1, 0.01)
        epsilon_decay = st.slider("探索率衰减", 0.9, 0.999, 0.995, 0.001)
        buffer_size = st.slider("经验缓冲区大小", 1000, 100000, 10000, 1000)
        batch_size = st.slider("批次大小", 16, 256, 64, 8)
        target_update = st.slider("目标网络更新频率", 1, 1000, 10, 1)

with col2:
    st.subheader("算法说明")
    if algorithm == "基础DQN":
        st.info("""
        **基础DQN**是深度Q学习的基本实现，包含两个主要创新：
        1. 经验回放：存储并随机采样过去的经验，打破数据相关性
        2. 目标网络：使用单独的网络计算目标Q值，提高训练稳定性
        """)
    elif algorithm == "Double DQN":
        st.info("""
        **Double DQN**解决了Q值过高估计问题：
        - 使用在线网络选择动作，目标网络评估该动作的价值
        - 分离动作选择和评估，减少偏差
        """)
    elif algorithm == "Dueling DQN":
        st.info("""
        **Dueling DQN**将Q值分解为状态价值和动作优势：
        - 价值流表示处于某状态的价值（与动作无关）
        - 优势流表示在该状态下采取特定动作的相对优势
        - 特别适合某些状态下大多数动作价值相似的情况
        """)
    elif algorithm == "优先经验回放DQN":
        st.info("""
        **优先经验回放**根据经验的重要性进行采样：
        - 使用TD误差的绝对值作为优先级度量
        - 优先级高的经验被更频繁地采样
        - 通过重要性采样权重修正引入的偏差
        """)

# 训练按钮
if st.button("开始训练"):
    # 创建保存模型和结果的目录
    os.makedirs("code/week7/trained_models", exist_ok=True)
    
    with st.spinner(f"正在训练{algorithm}..."):
        # 创建环境
        env = BlackjackWrapper(gym.make('Blackjack-v1'))
        env = RecordEpisodeStatistics(env)
        
        # 创建相应的智能体
        if algorithm == "基础DQN":
            agent = DQN(
                state_dim=3,  # 黑杰克状态维度：玩家点数、庄家明牌、是否有可用A
                action_dim=env.action_space.n,
                learning_rate=learning_rate,
                gamma=gamma,
                epsilon_start=epsilon_start,
                epsilon_end=epsilon_end,
                epsilon_decay=epsilon_decay,
                buffer_size=buffer_size,
                batch_size=batch_size,
                target_update=target_update
            )
        elif algorithm == "Double DQN":
            agent = DoubleDQN(
                state_dim=3,  # 黑杰克状态维度：玩家点数、庄家明牌、是否有可用A
                action_dim=env.action_space.n,
                learning_rate=learning_rate,
                gamma=gamma,
                epsilon_start=epsilon_start,
                epsilon_end=epsilon_end,
                epsilon_decay=epsilon_decay,
                buffer_size=buffer_size,
                batch_size=batch_size,
                target_update=target_update
            )
        elif algorithm == "Dueling DQN":
            agent = DuelingDQN(
                state_dim=3,  # 黑杰克状态维度：玩家点数、庄家明牌、是否有可用A
                action_dim=env.action_space.n,
                learning_rate=learning_rate,
                gamma=gamma,
                epsilon_start=epsilon_start,
                epsilon_end=epsilon_end,
                epsilon_decay=epsilon_decay,
                buffer_size=buffer_size,
                batch_size=batch_size,
                target_update=target_update
            )
        elif algorithm == "优先经验回放DQN":
            agent = PERDQN(
                state_dim=3,  # 黑杰克状态维度：玩家点数、庄家明牌、是否有可用A
                action_dim=env.action_space.n,
                learning_rate=learning_rate,
                gamma=gamma,
                epsilon_start=epsilon_start,
                epsilon_end=epsilon_end,
                epsilon_decay=epsilon_decay,
                buffer_size=buffer_size,
                batch_size=batch_size,
                target_update=target_update
            )
        
        # 训练智能体
        rewards, losses, epsilons = train_agent(env, agent, num_episodes=num_episodes)
        
        # 保存模型
        model_path = f"code/week7/trained_models/{algorithm.replace(' ', '_').lower()}.pt"
        torch.save(agent.q_network.state_dict(), model_path)
        st.success(f"模型已保存到 {model_path}")
        
        # 可视化训练结果
        plot_training_results(rewards, losses, epsilons)
        
        # 评估智能体
        avg_reward, success_rate = evaluate_agent(env, agent, num_episodes=100)
        
        st.success(f"评估完成! 平均奖励: {avg_reward:.2f}, 成功率: {success_rate*100:.2f}%")
        
        # 可视化Q值和策略
        st.subheader("学习到的策略可视化")
        visualize_q_values(agent, render_type='policy')
        
        # 保存训练结果到会话状态
        st.session_state.trained_agent = agent
        st.session_state.trained_algorithm = algorithm

# 如果已经训练了智能体，提供交互式测试功能
if 'trained_agent' in st.session_state:
    st.header("智能体测试")
    st.subheader(f"使用训练好的{st.session_state.trained_algorithm}进行测试")
    
    if st.button("进行一次游戏测试"):
        env = BlackjackWrapper(gym.make('Blackjack-v1'))
        state, _ = env.reset()
        render_blackjack_state(state)
        
        agent = st.session_state.trained_agent
        done = False
        total_reward = 0
        
        st.write("**游戏过程:**")
        
        while not done:
            action = agent.select_action(state, epsilon=0)  # 使用贪婪策略（不探索）
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            st.write(f"动作: {'要牌' if action == 1 else '停牌'}")
            st.write(f"新状态: 玩家点数={next_state[0]}, 庄家明牌={next_state[1]}, 可用A={'是' if next_state[2] else '否'}")
            
            if done:
                if reward > 0:
                    st.success(f"游戏结束，玩家获胜! 奖励: {reward}")
                elif reward < 0:
                    st.error(f"游戏结束，玩家失败! 奖励: {reward}")
                else:
                    st.info(f"游戏结束，平局! 奖励: {reward}")
            
            state = next_state

# 算法比较部分
st.header("DQN算法比较")

if st.button("比较不同DQN算法"):
    with st.spinner("正在比较不同DQN算法..."):
        # 创建环境
        env = BlackjackWrapper(gym.make('Blackjack-v1'))
        env = RecordEpisodeStatistics(env)
        
        # 创建不同的智能体
        agents = {
            "基础DQN": DQN(
                state_dim=3,  # 黑杰克状态维度：玩家点数、庄家明牌、是否有可用A
                action_dim=env.action_space.n,
                epsilon_decay=0.995
            ),
            "Double DQN": DoubleDQN(
                state_dim=3,  # 黑杰克状态维度：玩家点数、庄家明牌、是否有可用A
                action_dim=env.action_space.n,
                epsilon_decay=0.995
            ),
            "Dueling DQN": DuelingDQN(
                state_dim=3,  # 黑杰克状态维度：玩家点数、庄家明牌、是否有可用A
                action_dim=env.action_space.n,
                epsilon_decay=0.995
            ),
            "优先经验回放DQN": PERDQN(
                state_dim=3,  # 黑杰克状态维度：玩家点数、庄家明牌、是否有可用A
                action_dim=env.action_space.n,
                epsilon_decay=0.995
            )
        }
        
        # 设置训练回合数
        compare_episodes = 1000
        
        # 训练和比较
        results = {}
        
        for name, agent in agents.items():
            st.text(f"正在训练 {name}...")
            rewards, losses, _ = train_agent(env, agent, num_episodes=compare_episodes, log_interval=100)
            avg_reward, success_rate = evaluate_agent(env, agent)
            results[name] = {
                "rewards": rewards,
                "avg_reward": avg_reward,
                "success_rate": success_rate
            }
        
        # 绘制比较图表
        st.subheader("各算法奖励曲线对比")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for name, data in results.items():
            # 计算移动平均
            window_size = 50
            moving_avg = np.convolve(data["rewards"], np.ones(window_size)/window_size, mode='valid')
            ax.plot(moving_avg, label=name)
        
        ax.set_xlabel('回合')
        ax.set_ylabel('平均奖励 (移动平均)')
        ax.set_title('不同DQN算法的回合奖励比较')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        
        st.pyplot(fig)
        
        # 显示最终性能对比
        st.subheader("最终性能对比")
        
        # 创建对比表格
        table_data = {
            "算法": [],
            "平均奖励": [],
            "成功率 (%)": [],
            "最终100回合平均奖励": []
        }
        
        for name, data in results.items():
            table_data["算法"].append(name)
            table_data["平均奖励"].append(f"{data['avg_reward']:.3f}")
            table_data["成功率 (%)"].append(f"{data['success_rate']*100:.1f}")
            table_data["最终100回合平均奖励"].append(f"{np.mean(data['rewards'][-100:]):.3f}")
        
        st.table(table_data)
        
        st.success("算法比较完成!")

# 显示可用的DQN变体
st.markdown("### 可用的DQN变体")
st.markdown("""
- [标准DQN](./): 基础深度Q网络算法
- [Double DQN](1_double_dqn): 解决Q值过高估计问题
- [多步学习DQN](3_nstep_dqn): 加速信息传播，提高学习效率
- [探索策略](4_exploration): 不同的探索方法对比
""")
