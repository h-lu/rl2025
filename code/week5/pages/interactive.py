"""
DQN算法交互式实验页面

允许用户自定义DQN参数进行实验，观察不同参数对性能的影响
"""

import streamlit as st
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import time
import gymnasium as gym

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from agents.dqn_agent import DQNAgent, create_dqn_model
from environments.cart_pole import CartPoleEnv
from utils.visualization import plot_training_progress, create_altair_training_chart, configure_matplotlib_fonts

# 确保matplotlib可以显示中文字体
configure_matplotlib_fonts()

def render_interactive_page():
    """渲染DQN算法交互式实验页面"""
    st.title(f"{config.PAGE_ICONS['interactive']} {config.PAGE_TITLES['interactive']}")
    
    st.markdown("""
    ## 交互式DQN实验
    
    本页面允许您通过调整不同的参数来研究它们对DQN性能的影响。
    您可以创建和比较多个不同配置的DQN智能体，从而深入理解各参数的作用。
    """)
    
    # 创建选项卡
    tab1, tab2, tab3, tab4 = st.tabs(["实验设计", "实验结果", "案例研究", "参数敏感性分析"])
    
    # 实验设计选项卡
    with tab1:
        st.markdown("""
        ### 实验设计
        
        设计您的DQN实验，选择要测试的参数和取值范围。
        您可以创建多组不同的参数配置，系统将训练所有配置并比较它们的性能。
        """)
        
        # 基础环境设置
        st.subheader("环境设置")
        col1, col2 = st.columns(2)
        with col1:
            max_episodes = st.slider("训练回合数", min_value=10, max_value=200, value=50, step=10)
        with col2:
            max_steps = st.slider("每回合最大步数", min_value=100, max_value=500, value=200, step=50)
        
        # 创建实验组
        st.subheader("参数实验组")
        st.markdown("创建多组参数配置进行对比实验")
        
        # 初始化实验组
        if 'experiment_groups' not in st.session_state:
            st.session_state.experiment_groups = []
            st.session_state.next_group_id = 1
        
        # 显示当前实验组
        if st.session_state.experiment_groups:
            st.write("当前实验组:")
            group_data = []
            for i, group in enumerate(st.session_state.experiment_groups):
                group_data.append({
                    "ID": group["id"],
                    "名称": group["name"],
                    "折扣因子(γ)": group["gamma"],
                    "学习率": group["learning_rate"],
                    "探索策略": f"ε-greedy ({group['epsilon_start']} → {group['epsilon_end']})"
                })
            
            # 显示实验组表格
            df = pd.DataFrame(group_data)
            st.dataframe(df)
            
            # 删除实验组按钮
            if st.button("清除所有实验组"):
                st.session_state.experiment_groups = []
                st.rerun()
        
        # 添加新实验组
        st.subheader("添加新实验组")
        
        with st.form("new_experiment_group"):
            group_name = st.text_input("实验组名称", value=f"实验组 {st.session_state.next_group_id}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**核心参数**")
                gamma = st.slider("折扣因子(γ)", min_value=0.8, max_value=0.99, value=0.95, step=0.01, key="gamma_new")
                learning_rate = st.number_input("学习率", min_value=0.0001, max_value=0.01, value=0.001, format="%.4f", key="lr_new")
                
                st.write("**网络结构**")
                hidden_layer_size = st.slider("隐藏层大小", min_value=16, max_value=256, value=64, step=16, key="hidden_new")
            
            with col2:
                st.write("**探索策略参数**")
                epsilon_start = st.slider("初始探索率(ε)", min_value=0.5, max_value=1.0, value=1.0, step=0.1, key="eps_start_new")
                epsilon_end = st.slider("最小探索率(ε)", min_value=0.01, max_value=0.2, value=0.01, step=0.01, key="eps_end_new")
                epsilon_decay = st.slider("探索率衰减", min_value=0.9, max_value=0.99, value=0.95, step=0.01, key="eps_decay_new")
                
                st.write("**记忆回放与目标网络**")
                buffer_size = st.select_slider("回放缓冲区大小", options=[1000, 5000, 10000, 50000], value=10000, key="buffer_new")
                batch_size = st.select_slider("批次大小", options=[16, 32, 64, 128], value=32, key="batch_new")
                target_update_freq = st.slider("目标网络更新频率", min_value=1, max_value=20, value=5, step=1, key="target_update_new")
            
            # 添加实验组按钮
            submit_button = st.form_submit_button("添加实验组")
            
            if submit_button:
                new_group = {
                    "id": st.session_state.next_group_id,
                    "name": group_name,
                    "gamma": gamma,
                    "learning_rate": learning_rate,
                    "epsilon_start": epsilon_start,
                    "epsilon_end": epsilon_end,
                    "epsilon_decay": epsilon_decay,
                    "buffer_size": buffer_size,
                    "batch_size": batch_size,
                    "update_target_every": target_update_freq,
                    "hidden_layer_size": hidden_layer_size
                }
                st.session_state.experiment_groups.append(new_group)
                st.session_state.next_group_id += 1
                
                # 显示成功消息
                st.success(f"已添加实验组: {group_name}")
                st.rerun()
        
        # 运行实验按钮
        if st.session_state.experiment_groups:
            st.subheader("运行实验")
            if st.button("开始运行所有实验", type="primary"):
                # 初始化实验结果
                st.session_state.experiment_results = []
                
                # 创建进度显示区域
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                total_experiments = len(st.session_state.experiment_groups)
                
                for i, group in enumerate(st.session_state.experiment_groups):
                    status_text.text(f"正在训练实验组 {i+1}/{total_experiments}: {group['name']}")
                    
                    # 创建环境
                    env = CartPoleEnv()
                    
                    # 创建智能体
                    agent = DQNAgent(
                        state_size=env.state_size,
                        action_size=env.action_size,
                        gamma=group["gamma"],
                        epsilon_start=group["epsilon_start"],
                        epsilon_end=group["epsilon_end"],
                        epsilon_decay=group["epsilon_decay"],
                        learning_rate=group["learning_rate"],
                        buffer_size=group["buffer_size"],
                        batch_size=group["batch_size"],
                        update_target_every=group["update_target_every"],
                        hidden_layer_size=group["hidden_layer_size"]
                    )
                    
                    # 训练记录
                    scores = []
                    avg_scores = []
                    epsilon_history = []
                    
                    # 训练循环
                    for episode in range(1, max_episodes + 1):
                        state = env.reset()
                        score = 0
                        done = False
                        
                        for step in range(max_steps):
                            # 选择动作
                            action = agent.act(state)
                            
                            # 执行动作
                            next_state, reward, done, _ = env.step(action)
                            
                            # 智能体学习
                            agent.step(state, action, reward, next_state, done)
                            
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
                        
                        # 更新进度
                        progress = (i * max_episodes + episode) / (total_experiments * max_episodes)
                        progress_bar.progress(progress)
                        
                        # 每10个回合显示一次状态
                        if episode % 10 == 0:
                            status_text.text(f"训练 {group['name']}: 回合 {episode}/{max_episodes}, 平均得分: {avg_score:.2f}")
                    
                    # 保存实验结果
                    experiment_result = {
                        "id": group["id"],
                        "name": group["name"],
                        "params": group,
                        "scores": scores,
                        "avg_scores": avg_scores,
                        "epsilon_history": epsilon_history,
                        "final_avg_score": avg_scores[-1],
                        "max_score": max(scores),
                        "convergence_episode": next((i for i, s in enumerate(avg_scores) if s >= 195), None)
                    }
                    
                    st.session_state.experiment_results.append(experiment_result)
                    
                    # 环境清理
                    env.close()
                
                # 更新进度
                progress_bar.progress(1.0)
                status_text.text("所有实验完成！")
                
                # 重新运行以更新UI
                st.rerun()
        else:
            st.info("请添加至少一个实验组来开始实验")
    
    # 实验结果选项卡
    with tab2:
        st.markdown("""
        ### 实验结果
        
        这里显示所有实验组的训练结果和性能比较。
        您可以通过图表直观地比较不同参数配置对DQN性能的影响。
        """)
        
        # 检查是否有实验结果
        if 'experiment_results' in st.session_state and st.session_state.experiment_results:
            # 显示实验结果摘要
            st.subheader("实验摘要")
            
            # 创建结果摘要表格
            summary_data = []
            for result in st.session_state.experiment_results:
                summary_data.append({
                    "实验组": result["name"],
                    "最终平均得分": f"{result['final_avg_score']:.2f}",
                    "最高得分": f"{result['max_score']:.2f}",
                    "收敛回合": result["convergence_episode"] if result["convergence_episode"] is not None else "未收敛",
                    "γ值": result["params"]["gamma"],
                    "学习率": result["params"]["learning_rate"],
                    "ε衰减": result["params"]["epsilon_decay"]
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df)
            
            # 可视化比较
            st.subheader("学习曲线比较")
            
            # 创建多组实验的学习曲线
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for result in st.session_state.experiment_results:
                ax.plot(result["avg_scores"], label=result["name"])
            
            ax.set_xlabel("回合数")
            ax.set_ylabel("平均得分(最近100回合)")
            ax.set_title("不同参数配置的学习曲线比较")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # 添加目标得分线
            ax.axhline(y=195, color='r', linestyle='--', alpha=0.5, label="目标得分(195)")
            
            st.pyplot(fig)
            
            # 单独实验详细结果
            st.subheader("单个实验详细结果")
            
            # 实验选择器
            selected_exp_name = st.selectbox(
                "选择实验组查看详细结果",
                options=[result["name"] for result in st.session_state.experiment_results]
            )
            
            # 显示所选实验的详细结果
            selected_exp = next(result for result in st.session_state.experiment_results if result["name"] == selected_exp_name)
            
            # 显示参数
            st.write("**实验参数:**")
            param_cols = st.columns(3)
            
            with param_cols[0]:
                st.write(f"折扣因子(γ): {selected_exp['params']['gamma']}")
                st.write(f"学习率: {selected_exp['params']['learning_rate']}")
                st.write(f"隐藏层大小: {selected_exp['params']['hidden_layer_size']}")
            
            with param_cols[1]:
                st.write(f"初始探索率(ε): {selected_exp['params']['epsilon_start']}")
                st.write(f"最小探索率(ε): {selected_exp['params']['epsilon_end']}")
                st.write(f"探索率衰减: {selected_exp['params']['epsilon_decay']}")
            
            with param_cols[2]:
                st.write(f"回放缓冲区大小: {selected_exp['params']['buffer_size']}")
                st.write(f"批次大小: {selected_exp['params']['batch_size']}")
                st.write(f"目标网络更新频率: {selected_exp['params']['update_target_every']}")
            
            # 绘制详细学习曲线
            st.write("**学习曲线:**")
            fig = plot_training_progress(
                selected_exp["scores"], 
                selected_exp["avg_scores"], 
                selected_exp["epsilon_history"]
            )
            st.pyplot(fig)
            
            # 性能指标
            metric_cols = st.columns(4)
            
            with metric_cols[0]:
                st.metric("最终平均得分", f"{selected_exp['final_avg_score']:.2f}")
            
            with metric_cols[1]:
                st.metric("最高得分", f"{selected_exp['max_score']:.2f}")
            
            with metric_cols[2]:
                st.metric("收敛回合", selected_exp["convergence_episode"] if selected_exp["convergence_episode"] is not None else "未收敛")
            
            with metric_cols[3]:
                # 计算平均奖励增长率
                if len(selected_exp["avg_scores"]) >= 10:
                    first_10_avg = np.mean(selected_exp["avg_scores"][:10])
                    last_10_avg = np.mean(selected_exp["avg_scores"][-10:])
                    growth_rate = (last_10_avg - first_10_avg) / (len(selected_exp["avg_scores"]) - 10)
                    st.metric("平均增长率", f"{growth_rate:.2f}")
                else:
                    st.metric("平均增长率", "数据不足")
        
        else:
            st.info("尚未运行任何实验。请先在'实验设计'选项卡中设计并运行实验。")
    
    # 案例研究选项卡
    with tab3:
        st.markdown("""
        ### 案例研究
        
        这里展示一些预设的案例研究，比较不同参数对DQN性能的影响，帮助您理解参数调整的重要性。
        """)
        
        # 案例研究列表
        case_studies = [
            {
                "title": "折扣因子(γ)的影响",
                "description": "比较不同折扣因子值对DQN学习性能的影响",
                "param": "gamma",
                "values": [0.9, 0.95, 0.99],
                "explanation": """
                **折扣因子(γ)的影响**
                
                折扣因子决定了未来奖励的重要性:
                - **较小的γ值** (如0.9): 更注重即时奖励，可能导致短视行为
                - **中等的γ值** (如0.95): 平衡考虑即时和未来奖励
                - **较大的γ值** (如0.99): 更重视长期回报，但可能使训练不稳定
                
                在CartPole环境中，较高的γ值通常表现更好，因为杆子保持平衡需要长期规划。
                但过高的γ值可能导致训练不稳定，特别是当环境噪声较大时。
                """
            },
            {
                "title": "学习率的影响",
                "description": "比较不同学习率对DQN收敛速度和稳定性的影响",
                "param": "learning_rate",
                "values": [0.0001, 0.001, 0.01],
                "explanation": """
                **学习率的影响**
                
                学习率控制网络权重调整的幅度:
                - **较小的学习率** (如0.0001): 收敛慢但稳定
                - **中等的学习率** (如0.001): 平衡收敛速度和稳定性
                - **较大的学习率** (如0.01): 收敛快但可能不稳定
                
                学习率设置不当是导致DQN训练失败的常见原因之一。过高的学习率可能导致
                Q值发散，而过低的学习率则导致学习过慢，无法在合理时间内收敛。
                """
            },
            {
                "title": "探索策略的影响",
                "description": "比较不同探索策略参数对DQN探索效率的影响",
                "param": "epsilon_decay",
                "values": [0.9, 0.95, 0.99],
                "explanation": """
                **探索策略的影响**
                
                ε-greedy探索策略中的衰减率影响探索行为:
                - **快速衰减** (如0.9): 快速减少随机探索，更早利用学到的知识
                - **中等衰减** (如0.95): 平衡探索和利用
                - **慢速衰减** (如0.99): 保持长时间的探索行为
                
                在简单环境中，快速衰减可能有利于更快收敛。但在复杂环境中，维持较长的
                探索期可能帮助智能体发现更优策略，避免陷入局部最优。
                """
            },
            {
                "title": "记忆回放缓冲区大小的影响",
                "description": "比较不同回放缓冲区大小对DQN学习效率的影响",
                "param": "buffer_size",
                "values": [1000, 10000, 50000],
                "explanation": """
                **记忆回放缓冲区大小的影响**
                
                回放缓冲区的大小影响样本多样性和内存使用:
                - **小缓冲区** (如1000): 更关注最近的经验，内存效率高
                - **中等缓冲区** (如10000): 平衡新旧经验，适合大多数任务
                - **大缓冲区** (如50000): 保存更多样化的经验，但内存消耗大
                
                缓冲区过小可能导致过于关注最近的经验，影响样本多样性和去相关性。
                而过大的缓冲区可能包含过多旧的、不相关的经验，同时增加内存消耗。
                """
            }
        ]
        
        # 案例研究选择器
        selected_case_index = st.selectbox(
            "选择案例研究",
            options=range(len(case_studies)),
            format_func=lambda i: case_studies[i]["title"]
        )
        
        selected_case = case_studies[selected_case_index]
        
        st.subheader(selected_case["title"])
        st.write(selected_case["description"])
        
        # 显示案例研究的解释
        with st.expander("查看理论解释", expanded=True):
            st.markdown(selected_case["explanation"])
        
        # 运行案例研究
        if st.button(f"运行'{selected_case['title']}'案例研究"):
            # 设置基础参数
            base_params = {
                "gamma": 0.95,
                "learning_rate": 0.001,
                "epsilon_start": 1.0,
                "epsilon_end": 0.01,
                "epsilon_decay": 0.95,
                "buffer_size": 10000,
                "batch_size": 32,
                "update_target_every": 5,
                "hidden_layer_size": 64
            }
            
            # 创建案例研究实验组
            param_name = selected_case["param"]
            param_values = selected_case["values"]
            
            case_groups = []
            for i, value in enumerate(param_values):
                params = base_params.copy()
                params[param_name] = value
                params["name"] = f"{param_name}={value}"
                params["id"] = i + 1
                case_groups.append(params)
            
            # 设置实验组
            st.session_state.experiment_groups = case_groups
            st.session_state.next_group_id = len(case_groups) + 1
            
            # 开始运行实验
            st.info(f"已设置{len(case_groups)}个实验组。请点击'开始运行所有实验'按钮开始运行。")
            st.rerun()
    
    # 参数敏感性分析选项卡
    with tab4:
        st.markdown("""
        ### 参数敏感性分析
        
        这里提供更详细的参数敏感性分析工具，帮助您理解不同参数的重要性
        和它们对DQN性能的影响范围。
        """)
        
        if 'experiment_results' in st.session_state and len(st.session_state.experiment_results) >= 3:
            # 显示参数敏感性分析
            st.subheader("参数敏感性分析")
            
            # 创建参数敏感性数据
            param_names = ["gamma", "learning_rate", "epsilon_decay", "buffer_size", "batch_size", "update_target_every"]
            param_display_names = ["折扣因子(γ)", "学习率", "探索率衰减", "回放缓冲区大小", "批次大小", "目标网络更新频率"]
            
            # 用户选择要分析的性能指标
            metric = st.selectbox(
                "选择性能指标",
                options=["final_avg_score", "max_score", "convergence_episode"],
                format_func=lambda x: {
                    "final_avg_score": "最终平均得分",
                    "max_score": "最高得分",
                    "convergence_episode": "收敛回合数"
                }[x]
            )
            
            # 尝试分析参数敏感性
            try:
                # 为每个参数创建图表
                st.subheader("参数对性能的影响")
                
                for i, (param, display_name) in enumerate(zip(param_names, param_display_names)):
                    # 收集此参数的数据点
                    param_values = []
                    metric_values = []
                    
                    for result in st.session_state.experiment_results:
                        # 检查参数存在
                        if param in result["params"]:
                            param_values.append(result["params"][param])
                            
                            # 对于收敛回合，如果未收敛则使用最大回合数
                            if metric == "convergence_episode" and result[metric] is None:
                                metric_values.append(len(result["scores"]))
                            else:
                                metric_values.append(result[metric])
                    
                    # 如果有至少3个不同的参数值，绘制图表
                    if len(set(param_values)) >= 2:
                        fig, ax = plt.subplots(figsize=(8, 4))
                        
                        # 按参数值排序
                        sorted_data = sorted(zip(param_values, metric_values))
                        sorted_params, sorted_metrics = zip(*sorted_data)
                        
                        # 绘制参数影响图
                        ax.plot(sorted_params, sorted_metrics, 'o-', linewidth=2)
                        ax.set_xlabel(display_name)
                        ax.set_ylabel({
                            "final_avg_score": "最终平均得分",
                            "max_score": "最高得分",
                            "convergence_episode": "收敛回合数"
                        }[metric])
                        
                        # 设置图表标题和网格
                        ax.set_title(f"{display_name}对{ax.get_ylabel()}的影响")
                        ax.grid(True, linestyle='--', alpha=0.7)
                        
                        # 显示图表
                        st.pyplot(fig)
                    else:
                        st.info(f"没有足够的{display_name}变化数据进行分析。至少需要2个不同的参数值。")
            
            except Exception as e:
                st.error(f"参数敏感性分析出错: {str(e)}")
                st.info("可能是由于实验数据不足或参数变化不够。请尝试运行更多不同参数配置的实验。")
        
        else:
            st.info("参数敏感性分析需要至少3个不同的实验结果。请先在'实验设计'选项卡中设计并运行更多实验。")
            
            # 生成示例分析
            st.subheader("参数敏感性分析示例")
            
            # 生成示例数据
            np.random.seed(42)
            param_values = np.linspace(0.9, 0.99, 5)
            scores = [50 + 150 * p + np.random.normal(0, 10) for p in param_values]
            
            # 绘制示例图
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(param_values, scores, 'o-', linewidth=2)
            ax.set_xlabel("折扣因子 (γ)")
            ax.set_ylabel("最终平均得分")
            ax.set_title("折扣因子对性能的影响(示例)")
            ax.grid(True, linestyle='--', alpha=0.7)
            
            st.pyplot(fig)
            st.caption("注意: 这是示例数据，不代表实际结果。请运行实际实验以获得真实分析。") 