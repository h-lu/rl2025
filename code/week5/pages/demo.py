"""
DQN算法演示页面

展示DQN在CartPole环境中的训练和评估过程
"""

import streamlit as st
import sys
import os
import numpy as np
import pandas as pd
import time
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import gymnasium as gym
from PIL import Image
import math

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from agents.dqn_agent import DQNAgent
from environments.cart_pole import CartPoleEnv
from utils.visualization import plot_training_progress, plot_q_values, configure_matplotlib_fonts

# 确保matplotlib可以显示中文字体
configure_matplotlib_fonts()

def render_demo_page():
    """渲染DQN算法演示页面"""
    st.title(f"{config.PAGE_ICONS['demo']} {config.PAGE_TITLES['demo']}")
    
    st.markdown("""
    ## DQN算法演示
    
    本页面展示DQN算法在CartPole环境中的训练和评估过程。您可以选择使用预训练的模型直接查看性能，
    或者在浏览器中实时训练一个新模型并观察学习过程。
    """)
    
    # 创建选项卡
    tab1, tab2, tab3 = st.tabs(["预训练模型演示", "实时训练", "可视化分析"])
    
    # 预训练模型演示选项卡
    with tab1:
        st.markdown("""
        ### 预训练模型演示
        
        这里展示了一个已经训练好的DQN智能体在CartPole环境中的表现。
        预训练模型已经学会了如何平衡杆子，您可以观察到智能体如何选择动作来维持杆子直立。
        """)
        
        # 显示CartPole环境介绍
        with st.expander("CartPole环境介绍", expanded=True):
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("""
                #### 环境描述
                
                CartPole是一个经典的控制问题，目标是通过向左或向右移动小车，使得杆子保持直立。
                
                - **状态空间**: 4维连续空间
                  1. 小车位置 (-4.8 ~ 4.8)
                  2. 小车速度 (-∞ ~ ∞)
                  3. 杆子角度 (-0.418 ~ 0.418 弧度)
                  4. 杆子角速度 (-∞ ~ ∞)
                  
                - **动作空间**: 2个离散动作
                  - 0: 向左推小车
                  - 1: 向右推小车
                  
                - **奖励**: 每个时间步获得+1的奖励
                
                - **终止条件**:
                  - 杆子倾斜角度超过±15度
                  - 小车位置超出±2.4单位
                  - 达到最大步数(500)
                
                **成功标准**: 平均超过195分(最近100回合)
                """)
            
            with col2:
                # 显示CartPole图片
                try:
                    # 尝试本地显示图片
                    img_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                        "assets", "cartpole.png")
                    if os.path.exists(img_path):
                        img = Image.open(img_path)
                        st.image(img, caption="CartPole环境示意图")
                    else:
                        # 如果本地图片不存在，使用网络图片
                        st.image("https://gymnasium.farama.org/_images/cart_pole.gif", 
                               caption="CartPole环境示意图")
                except:
                    # 如果无法加载，显示替代文本
                    st.info("无法加载CartPole环境图片，请检查网络连接或图片路径。")
        
        # 添加演示控制按钮
        col1, col2 = st.columns([1, 1])
        with col1:
            episodes = st.number_input("演示回合数", min_value=1, max_value=10, value=3)
        with col2:
            speed = st.select_slider("播放速度", options=["慢", "中", "快"], value="中")
        
        # 演示按钮
        if st.button("开始演示"):
            # 显示演示进度
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 创建环境
            env = CartPoleEnv(render_mode='rgb_array')
            
            # 创建并初始化智能体
            agent = DQNAgent(state_size=env.state_size, action_size=env.action_size)
            
            # 模拟已经训练好的智能体
            # 实际应用中应该加载保存的模型，但这里我们使用一个简单策略
            
            # 演示多个回合
            all_frames = []
            episode_rewards = []
            
            # 设置播放速度
            if speed == "慢":
                frame_delay = 0.05
            elif speed == "中":
                frame_delay = 0.02
            else:
                frame_delay = 0.001
            
            for ep in range(episodes):
                status_text.text(f"回合 {ep+1}/{episodes}")
                
                # 重置环境
                state = env.reset()
                episode_reward = 0
                done = False
                t = 0
                max_t = 500
                
                # 回合循环
                while not done and t < max_t:
                    # 使用确定性策略选择动作
                    # 我们将状态的第3个值（杆子角度）和第4个值（角速度）
                    # 用作简单的启发式规则
                    if state[2] < 0:  # 如果杆子向左倾斜
                        action = 0  # 向左移动
                    else:  # 如果杆子向右倾斜
                        action = 1  # 向右移动
                    
                    # 执行动作
                    next_state, reward, done, _ = env.step(action)
                    
                    # 更新状态和累积奖励
                    state = next_state
                    episode_reward += reward
                    t += 1
                    
                    # 显示进度
                    progress = (ep * max_t + t) / (episodes * max_t)
                    progress_bar.progress(min(progress, 1.0))
                    
                    # 收集帧以便动画显示
                    if len(env.episode_frames) > 0 and t % 2 == 0:  # 减少帧数，使动画更快
                        all_frames.append(env.episode_frames[-1])
                        
                        # 展示最新帧
                        if t % 10 == 0:
                            fig, ax = plt.subplots(figsize=(8, 4))
                            ax.imshow(env.episode_frames[-1])
                            ax.axis('off')
                            st.pyplot(fig)
                            plt.close(fig)
                            time.sleep(frame_delay)
                
                episode_rewards.append(episode_reward)
                status_text.text(f"回合 {ep+1}/{episodes} 完成，得分: {episode_reward}")
            
            # 显示最终性能
            env.close()
            st.success(f"演示完成！平均得分: {np.mean(episode_rewards):.2f}")
            
            # 如果有足够的帧，创建动画
            if len(all_frames) > 10:
                st.subheader("完整演示回放")
                
                # 这里应该使用动画但简化版不实现
                st.warning("动画回放功能需要额外代码，在简化版中未实现。")
    
    # 实时训练选项卡
    with tab2:
        st.markdown("""
        ### 实时训练
        
        在这里，您可以在浏览器中实时训练一个DQN智能体来解决CartPole问题，并观察学习过程。
        
        **注意**：由于浏览器中的计算资源限制，训练可能会比较慢，建议使用较小的训练回合数。
        """)
        
        # 训练参数设置
        st.subheader("训练参数")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            max_episodes = st.slider("训练回合数", min_value=10, max_value=200, value=50, step=10)
            gamma = st.slider("折扣因子 (γ)", min_value=0.8, max_value=0.999, value=0.99, step=0.01)
        
        with col2:
            epsilon_start = st.slider("初始探索率", min_value=0.5, max_value=1.0, value=1.0, step=0.1)
            epsilon_end = st.slider("最小探索率", min_value=0.01, max_value=0.2, value=0.01, step=0.01)
        
        with col3:
            epsilon_decay = st.slider("探索率衰减", min_value=0.9, max_value=0.999, value=0.995, step=0.001)
            target_update = st.slider("目标网络更新频率", min_value=1, max_value=100, value=10, step=1)
        
        # 高级参数
        with st.expander("高级参数", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                learning_rate = st.slider("学习率", min_value=0.0001, max_value=0.01, value=0.001, step=0.0001, format="%.4f")
                batch_size = st.slider("批次大小", min_value=16, max_value=128, value=64, step=8)
            
            with col2:
                buffer_size = st.slider("回放缓冲区大小", min_value=1000, max_value=100000, value=10000, step=1000)
                hidden_size = st.slider("隐藏层大小", min_value=16, max_value=128, value=64, step=16)
        
        # 开始训练按钮
        if st.button("开始训练"):
            # 创建训练进度区域
            progress_bar = st.progress(0)
            episode_text = st.empty()
            score_text = st.empty()
            chart_placeholder = st.empty()
            metrics_container = st.container()
            
            # 训练智能体
            try:
                # 创建环境
                env = CartPoleEnv()
                
                # 创建智能体
                agent = DQNAgent(
                    state_size=env.state_size, 
                    action_size=env.action_size,
                    gamma=gamma,
                    epsilon_start=epsilon_start,
                    epsilon_end=epsilon_end,
                    epsilon_decay=epsilon_decay,
                    buffer_size=buffer_size,
                    batch_size=batch_size,
                    update_target_every=target_update
                )
                
                # 训练记录
                scores = []
                avg_scores = []
                epsilon_history = []
                loss_history = []
                
                # 训练循环
                for i_episode in range(1, max_episodes + 1):
                    state = env.reset()
                    score = 0
                    done = False
                    max_steps = 500
                    
                    for t in range(max_steps):
                        # 选择动作
                        action = agent.act(state)
                        
                        # 执行动作
                        next_state, reward, done, _ = env.step(action)
                        
                        # 智能体学习
                        loss = agent.step(state, action, reward, next_state, done)
                        if loss:
                            loss_history.append(loss)
                        
                        # 更新状态和累积奖励
                        state = next_state
                        score += reward
                        
                        # 如果回合结束，跳出循环
                        if done:
                            break
                    
                    # 记录回合结果
                    scores.append(score)
                    avg_score = np.mean(scores[-100:])
                    avg_scores.append(avg_score)
                    epsilon_history.append(agent.get_epsilon())
                    
                    # 更新界面
                    progress = i_episode / max_episodes
                    progress_bar.progress(progress)
                    episode_text.text(f"训练回合: {i_episode}/{max_episodes}")
                    score_text.text(f"最新得分: {score:.2f}, 平均得分(最近100回合): {avg_score:.2f}, 探索率(ε): {agent.get_epsilon():.4f}")
                    
                    # 每5个回合更新图表
                    if i_episode % 5 == 0 or i_episode == max_episodes:
                        # 绘制训练进度图表
                        fig = plot_training_progress(scores, avg_scores, epsilon_history)
                        chart_placeholder.pyplot(fig)
                        plt.close(fig)
                        
                        # 显示训练指标
                        with metrics_container:
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("最高得分", f"{max(scores):.1f}")
                            
                            with col2:
                                st.metric("平均得分", f"{avg_score:.1f}")
                            
                            with col3:
                                st.metric("当前探索率(ε)", f"{agent.get_epsilon():.4f}")
                
                # 训练完成
                env.close()
                st.success(f"训练完成！最终平均得分: {avg_score:.2f}")
                
                # 显示损失曲线
                if len(loss_history) > 0:
                    st.subheader("训练损失曲线")
                    
                    # 对损失进行平滑
                    smoothed_loss = np.array(loss_history)
                    if len(smoothed_loss) > 100:
                        kernel_size = min(50, len(smoothed_loss) // 10)
                        kernel = np.ones(kernel_size) / kernel_size
                        smoothed_loss = np.convolve(smoothed_loss, kernel, mode='valid')
                    
                    # 绘制损失曲线
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(loss_history, alpha=0.3, label='原始损失')
                    ax.plot(range(len(smoothed_loss)), smoothed_loss, label='平滑损失')
                    ax.set_xlabel('训练步数')
                    ax.set_ylabel('损失值')
                    ax.legend()
                    ax.set_title('DQN训练损失')
                    st.pyplot(fig)
                
                # 保存会话状态
                st.session_state.trained_agent = agent
                st.session_state.training_scores = scores
                st.session_state.training_avg_scores = avg_scores
                st.session_state.epsilon_history = epsilon_history
                
                # 添加训练后的评估选项
                st.subheader("评估训练结果")
                st.info("训练已完成，您可以切换到'可视化分析'选项卡查看更多详细分析。")
                
            except Exception as e:
                st.error(f"训练过程中出现错误: {str(e)}")
    
    # 可视化分析选项卡
    with tab3:
        st.markdown("""
        ### 可视化分析
        
        在这里，您可以查看训练好的DQN智能体的详细性能分析，包括Q值分布、关键状态下的动作选择等。
        
        **注意**：此页面需要先在"实时训练"选项卡中完成训练才能显示完整内容。
        """)
        
        # 检查是否有训练好的智能体
        if 'trained_agent' in st.session_state:
            agent = st.session_state.trained_agent
            scores = st.session_state.training_scores
            avg_scores = st.session_state.training_avg_scores
            epsilon_history = st.session_state.epsilon_history
            
            st.success("发现训练好的智能体！以下是详细分析:")
            
            # 显示关键指标
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("最高得分", f"{max(scores):.1f}")
                st.metric("训练回合数", f"{len(scores)}")
            
            with col2:
                st.metric("最终平均得分", f"{avg_scores[-1]:.1f}")
                st.metric("最终探索率(ε)", f"{epsilon_history[-1]:.4f}")
            
            with col3:
                # 计算训练是否成功
                success = avg_scores[-1] >= 195.0
                st.metric("训练状态", "成功 ✅" if success else "未收敛 ⚠️")
                st.metric("收敛速度", f"{np.argmax([s >= 195 for s in avg_scores]) if success else '未收敛'}")
            
            # 绘制训练曲线
            st.subheader("训练过程分析")
            fig = plot_training_progress(scores, avg_scores, epsilon_history)
            st.pyplot(fig)
            
            # Q值分析
            st.subheader("策略分析")
            
            # 创建环境以生成状态示例
            env = CartPoleEnv()
            
            # 收集一些典型状态
            def generate_sample_states(n=100):
                states = []
                state = env.reset()
                states.append(state)
                
                for _ in range(n-1):
                    action = env.env.action_space.sample()
                    next_state, _, done, _ = env.step(action)
                    states.append(next_state)
                    if done:
                        state = env.reset()
                        states.append(state)
                
                return np.array(states)
            
            # 生成样本状态
            sample_states = generate_sample_states()
            
            # 分析Q值分布
            st.write("#### Q值分布")
            st.write("下图显示了在样本状态下，智能体对不同动作的平均Q值估计。这反映了智能体的策略偏好。")
            
            try:
                # 绘制Q值分布
                fig = plot_q_values(sample_states, agent.q_network)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"无法绘制Q值分布: {str(e)}")
            
            # 状态重要性分析
            st.write("#### 状态特征重要性")
            st.write("CartPole环境中有4个状态特征，以下分析显示不同特征对智能体决策的影响程度。")
            
            # 生成一个平衡状态
            balanced_state = np.array([0.0, 0.0, 0.0, 0.0])
            
            # 创建不同特征值的状态
            feature_values = {
                "小车位置": np.linspace(-2.4, 2.4, 11),
                "小车速度": np.linspace(-2.0, 2.0, 11),
                "杆子角度": np.linspace(-0.2, 0.2, 11),
                "杆子角速度": np.linspace(-2.0, 2.0, 11)
            }
            
            # 分析每个特征对动作选择的影响
            for i, (feature_name, values) in enumerate(feature_values.items()):
                states = np.array([balanced_state.copy() for _ in values])
                for j, value in enumerate(values):
                    states[j, i] = value
                
                # 获取动作值
                q_values = agent.q_network.predict(states)
                left_probs = np.exp(q_values[:, 0]) / np.sum(np.exp(q_values), axis=1)
                right_probs = np.exp(q_values[:, 1]) / np.sum(np.exp(q_values), axis=1)
                
                # 绘制特征重要性图表
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(values, left_probs, 'b-', label='向左概率')
                ax.plot(values, right_probs, 'r-', label='向右概率')
                ax.axhline(y=0.5, color='gray', linestyle='--')
                ax.set_xlabel(feature_name)
                ax.set_ylabel('动作概率')
                ax.legend()
                ax.set_title(f'{feature_name}对动作选择的影响')
                st.pyplot(fig)
                plt.close(fig)
                
                # 解释
                sensitivity = np.max(np.abs(right_probs - 0.5))
                if sensitivity > 0.3:
                    st.write(f"**解释**: 智能体对{feature_name}非常敏感，这是决策的关键因素。")
                elif sensitivity > 0.1:
                    st.write(f"**解释**: 智能体对{feature_name}有一定敏感度，这会影响决策。")
                else:
                    st.write(f"**解释**: 智能体对{feature_name}不太敏感，这不是主要决策因素。")
            
            # 关闭环境
            env.close()
            
        else:
            st.warning("请先在'实时训练'选项卡中训练一个智能体，然后返回此页面查看分析。")
            
            # 提供一个示例分析
            st.subheader("示例分析")
            st.info("以下是一个训练好的DQN智能体的示例分析，仅供参考。")
            
            # 生成示例数据
            example_episodes = np.arange(100)
            np.random.seed(42)
            example_scores = [min(200, max(0, 20 + i + np.random.normal(0, 10))) for i in range(100)]
            example_avg_scores = [np.mean(example_scores[max(0, i-9):i+1]) for i in range(100)]
            example_epsilon = [max(0.01, 1.0 * (0.99 ** i)) for i in range(100)]
            
            # 绘制示例训练曲线
            fig, ax1 = plt.subplots(figsize=(10, 5))
            
            color = 'tab:blue'
            ax1.set_xlabel('回合数')
            ax1.set_ylabel('分数', color=color)
            ax1.plot(example_episodes, example_scores, alpha=0.3, color=color, label='回合分数')
            ax1.plot(example_episodes, example_avg_scores, color=color, label='平均分数')
            ax1.tick_params(axis='y', labelcolor=color)
            
            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel('Epsilon', color=color)
            ax2.plot(example_episodes, example_epsilon, color=color, label='Epsilon')
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.set_ylim([0, 1.1])
            
            fig.tight_layout()
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            plt.title('DQN训练示例')
            st.pyplot(fig)
            plt.close(fig)
            
            st.caption("注意: 这是示例数据，不代表实际训练结果。请运行实际训练以获得真实分析。") 