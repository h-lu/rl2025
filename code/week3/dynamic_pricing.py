import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from dynamic_pricing_gym import DynamicPricingGymEnv, q_learning

def show():
    """展示动态定价模型的Streamlit界面"""
    
    # 设置页面标题
    st.title("强化学习动态定价模型")
    
    # 用于存储最终批次的Q表，只在训练结束后一次性显示热图
    final_q_table = None
    final_visit_stats = None
    
    st.markdown("""
    ## 模型介绍
    
    这个动态定价环境使用Gymnasium框架实现，它模拟了一个电子商务平台的定价决策过程。
    模型结合了市场状态动态和库存管理，提供了一个更贴近现实的定价环境。
    
    ### 核心模型组件
    
    1. **状态空间**：
       - 市场状态（低迷、平淡、活跃、热门、火爆）
       - 库存水平（极低、较低、适中、较高、极高）
       - 额外特征：实际库存数量、当前天数
    
    2. **动作空间**：
       - 可选价格水平（基于基础价格的调整百分比）
    
    3. **转移动态**：
       - 市场状态根据马尔可夫链平滑转移
       - 库存随销售减少，并在低于阈值时自动补货
    
    4. **奖励函数**：
       - 销售利润 = 收入 - 成本
       - 缺货惩罚 = 未满足需求 × 惩罚系数
       - 库存持有成本 = 库存量 × 单位持有成本率
       - 总奖励 = 销售利润 - 缺货惩罚 - 库存持有成本
    
    ### 应用场景
    
    这种定价模型适用于多种实际业务场景：
    
    - **电商平台**：根据市场热度和库存情况动态调整商品价格
    - **酒店客房**：针对不同季节和入住率调整房价
    - **航空票价**：根据航班预订情况和临近起飞时间调整票价
    - **餐饮行业**：根据用餐高峰期和食材新鲜度调整菜品价格
    - **共享出行**：根据用户需求高峰和车辆供应调整价格
    """)
    
    # 数学模型说明
    with st.expander("展开查看详细数学模型"):
        st.markdown("""
        ## 数学模型
        
        ### 1. 价格弹性需求模型
        
        需求量基于价格弹性模型计算：
        
        Q = Q₀ · (P/P₀)^(-η) · (1 + ε)
        
        其中：
        - Q：预期需求量
        - Q₀：基础需求量，与市场状态相关
        - P：定价
        - P₀：基准价格
        - η：价格弹性系数
        - ε：随机扰动，范围为[-noise, noise]
        
        ### 2. 市场状态转移（马尔可夫过程）
        
        市场状态按照马尔可夫链进行转移：
        
        P(s_{t+1} = s_j | s_t = s_i) = T_{ij}
        
        其中T是状态转移矩阵，主对角线元素表示保持当前状态的概率，非对角线元素表示状态转移的概率。
        
        ### 3. 利润计算
        
        利润 = 销售收入 - 销售成本 - 缺货惩罚 - 库存持有成本
        
        = min(Q, I) · (P - C) - max(0, Q - I) · P_stockout - I · C_holding
        
        其中：
        - I：当前库存量
        - C：单位成本
        - P_stockout：缺货惩罚系数
        - C_holding：单位库存持有成本率
        
        ### 4. Q学习算法
        
        Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]
        
        其中：
        - s, a：当前状态和动作
        - r：获得的奖励
        - s'：下一个状态
        - α：学习率
        - γ：折扣因子
        """)
    
    # 参数设置部分
    st.header("模型参数设置")
    
    # 分成三列来容纳更多参数
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("价格参数")
        base_price = st.slider("基础价格", 50, 200, 100, help="产品的标准定价，单位为元")
        cost_ratio = st.slider("成本价格比例", 0.3, 0.9, 0.7, 0.05, help="成本价格占基础价格的比例")
        price_range = st.slider("价格调整范围", 0.1, 0.5, 0.3, 0.05, help="价格上下浮动的百分比")
        price_levels = st.slider("价格级别数量", 3, 9, 5, help="可选择的不同价格水平数量")
        
    with col2:
        st.subheader("市场参数")
        elasticity = st.slider("价格弹性系数", 0.5, 3.0, 1.5, 0.1, help="需求对价格变化的敏感程度，值越大需求对价格越敏感")
        market_stability = st.slider("市场稳定性", 0.1, 0.9, 0.6, 0.1, help="市场状态保持不变的概率，值越大市场状态变化越慢")
        market_jump_prob = st.slider("市场跳变概率", 0.0, 0.2, 0.05, 0.01, help="非相邻市场状态间直接跳变的概率，提高状态覆盖率")
        noise = st.slider("市场随机波动", 0.0, 0.5, 0.2, 0.05, help="需求随机波动的幅度，模拟现实市场的不确定性")
        stockout_penalty = st.slider("缺货惩罚系数", 10, 100, 50, 10, help="每单位未满足需求的惩罚，反映客户满意度损失")
        
    with col3:
        st.subheader("库存参数")
        initial_inventory = st.slider("初始库存", 100, 1000, 500, 50, help="模拟开始时的初始库存数量")
        max_inventory = st.slider("最大库存", 500, 2000, 1000, 100, help="仓库最大容量，超过此值将无法补货")
        init_inv_range = st.slider("初始库存随机范围", 0.0, 1.0, (0.2, 0.9), 0.1, help="初始库存的随机范围（相对于最大库存的比例），增加状态空间覆盖")
        restock_threshold = st.slider("补货阈值", 0.1, 0.5, 0.2, 0.05, help="当库存低于最大库存的这个比例时触发补货")
        restock_amount = st.slider("补货数量", 50, 500, 200, 50, help="每次补货的固定数量")
        restock_randomness = st.slider("补货随机波动", 0.0, 0.5, 0.3, 0.1, help="补货数量的随机波动范围，增加库存变化的多样性")
        holding_cost_rate = st.slider("库存持有成本率", 0.001, 0.05, 0.01, 0.001, help="每单位库存的持有成本占成本价格的比例")
    
    # 显示生成的价格水平
    cost_price = base_price * cost_ratio
    price_step = 2 * price_range / (price_levels - 1) if price_levels > 1 else 0
    price_actions = [round(base_price * (1 - price_range + i * price_step), 2) for i in range(price_levels)]
    
    with st.expander("查看价格动作空间"):
        st.markdown("### 价格动作空间")
        st.markdown("""
        下表展示了模型可以选择的所有价格水平。在训练过程中，模型将学习在不同状态下选择最优的价格水平。
        注意观察最低和最高价格与基准价格的差异，这直接影响了价格策略的灵活性。
        """)
        price_df = pd.DataFrame({
            "价格级别": [f"价格{i+1}" for i in range(len(price_actions))],
            "实际价格": price_actions,
            "相对基准价格": [f"{(p/base_price - 1)*100:.1f}%" for p in price_actions],
            "利润率": [f"{(p-cost_price)/p*100:.1f}%" for p in price_actions]
        })
        st.table(price_df)
    
    # 学习参数设置
    st.header("学习参数设置")
    
    with st.expander("了解学习参数的影响"):
        st.markdown("""
        ### 学习参数对模型的影响
        
        - **学习率(α)**：控制新信息对已有知识的更新速度。较高的学习率使模型快速适应新情况但可能不稳定，较低的学习率学习较慢但更稳定。
        
        - **折扣因子(γ)**：决定模型对未来奖励的重视程度。值越接近1，模型越重视长期收益；值越小，模型越注重眼前利益。
        
        - **探索率(ε)**：控制模型尝试新策略的概率。较高的探索率有助于发现新的优秀策略，但会牺牲短期收益。
        
        - **探索率衰减**：随着训练进行，逐渐降低探索率，使模型从探索转向利用已学到的知识。
        
        - **最小探索率**：设置一个探索率下限，确保模型始终保持一定程度的探索，避免陷入局部最优。
        
        - **训练轮数**：决定模型学习的总周期。轮数越多，模型有更多机会学习，但也需要更长的训练时间。
        
        - **乐观初始化**：通过给Q表设置较高的初始值，鼓励算法在早期更多地探索各种可能性，有助于发现全局最优策略。
        
        - **周期性强制探索**：定期让模型进行完全随机的探索，帮助访问更多不同的状态，提高状态空间覆盖率。
        """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        learning_rate = st.slider("学习率 (α)", 0.01, 0.5, 0.2, 0.01, help="控制每次更新的幅度，值越大学习越快但可能不稳定")
        discount_factor = st.slider("折扣因子 (γ)", 0.5, 0.99, 0.9, 0.01, help="控制对未来奖励的重视程度，值越大越注重长期收益")
        epsilon = st.slider("初始探索率 (ε)", 0.1, 1.0, 0.7, 0.1, help="初始随机探索的概率，值越大越倾向于尝试新策略")
        
    with col2:
        epsilon_decay = st.slider("探索率衰减", 0.9, 0.999, 0.995, 0.001, help="每轮探索率的衰减系数，值越大衰减越慢")
        min_epsilon = st.slider("最小探索率", 0.01, 0.2, 0.05, 0.01, help="探索率的下限，确保持续探索新策略")
        num_episodes = st.slider("训练轮数", 100, 5000, 2000, 100, help="完整训练的轮数，建议至少1000轮以获得稳定效果")
    
    # 高级参数
    with st.expander("高级学习参数"):
        st.markdown("""
        ### 高级学习参数
        
        这些参数可以进一步调整学习算法的行为，影响状态空间的覆盖率和训练效果。
        """)
        
        use_optimistic_init = st.checkbox("使用乐观初始化", value=True, help="初始化Q表为正值而非零值，鼓励更多探索")
        
        if use_optimistic_init:
            init_value = st.slider("初始Q值", 1.0, 100.0, 50.0, 1.0, help="Q表的初始值，较高的值会促进更多探索")
        else:
            init_value = 0.0
        
        use_dynamic_lr = st.checkbox("使用动态学习率", value=True, help="随着状态-动作对访问次数增加降低学习率")
        
        periodic_explore = st.checkbox("周期性强制探索", value=True, help="每隔一定轮数进行完全随机探索，增加状态空间覆盖率")
        
        batch_size = st.slider("批次大小", 10, 100, 50, 10, help="每批次训练的轮数，影响界面更新频率")
        
        smooth_window = st.slider("奖励平滑窗口", 1, 50, 20, 1, help="奖励曲线的移动平均窗口大小，用于减少噪声")
        
        st.markdown("""
        ### 提高状态空间覆盖率的技巧
        
        如果状态空间覆盖率较低（不到50%），可以尝试以下调整：
        
        1. **提高市场跳变概率**：允许市场状态之间的非连续跳转
        2. **扩大初始库存随机范围**：使训练开始时能够覆盖更多不同的库存水平
        3. **增加补货随机波动**：使库存变化更加多样化
        4. **启用周期性强制探索**：定期进行完全随机的动作选择
        5. **提高最小探索率**：确保在训练后期仍有足够的探索
        6. **增加训练轮数**：给模型更多时间探索状态空间
        """)
    
    # 创建环境配置
    env_config = {
        'base_price': base_price,
        'cost_price': cost_price,
        'price_range': price_range,
        'price_levels': price_levels,
        'elasticity': elasticity,
        'noise': noise,
        'market_transition_stability': market_stability,
        'market_transition_jump_prob': market_jump_prob,
        'stockout_penalty': stockout_penalty,
        'holding_cost_rate': holding_cost_rate,
        'initial_inventory': initial_inventory,
        'max_inventory': max_inventory,
        'inventory_init_range': init_inv_range,
        'restock_threshold': restock_threshold,
        'restock_amount': restock_amount,
        'restock_randomness': restock_randomness,
        'max_steps': 365  # 一年
    }
    
    # 训练按钮
    if st.button("开始训练", help="点击开始使用当前参数设置训练模型"):
        # 创建环境
        env = DynamicPricingGymEnv(config=env_config)
        
        # 进度条
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 创建训练记录占位符
        reward_chart = st.empty()
        coverage_chart = st.empty()
        q_value_container = st.container()  # 使用固定容器而非empty
        visit_stats_chart = st.empty()
        
        # 训练模型并记录进度
        with st.spinner("训练中..."):
            # 模拟分批训练以更新进度
            num_batches = (num_episodes + batch_size - 1) // batch_size
            
            all_history = []
            # 初始化Q表，可选乐观初始化
            if use_optimistic_init:
                q_table = np.ones((env.n_market_states, env.n_inventory_levels, env.action_space.n)) * init_value
            else:
                q_table = np.zeros((env.n_market_states, env.n_inventory_levels, env.action_space.n))
            
            current_epsilon = epsilon
            
            for batch in range(num_batches):
                # 更新进度
                progress = (batch + 1) / num_batches
                progress_bar.progress(progress)
                status_text.text(f"训练进度: {int(progress * 100)}% (第 {batch * batch_size + 1}-{min((batch + 1) * batch_size, num_episodes)} 轮)")
                
                # 计算本批次的轮数
                start_ep = batch * batch_size
                end_ep = min((batch + 1) * batch_size, num_episodes)
                batch_episodes = end_ep - start_ep
                
                # 训练本批次
                current_epsilon = max(min_epsilon, epsilon * (epsilon_decay ** start_ep))
                batch_q_table, batch_history, visit_stats = q_learning(
                    env, 
                    batch_episodes, 
                    learning_rate, 
                    discount_factor, 
                    current_epsilon, 
                    epsilon_decay,
                    min_epsilon,
                    use_optimistic_init,
                    init_value,
                    periodic_explore
                )
                
                # 更新Q表
                q_table = batch_q_table
                
                # 更新历史记录
                for item in batch_history:
                    item["episode"] += start_ep
                all_history.extend(batch_history)
                
                # 计算移动平均奖励
                history_df = pd.DataFrame(all_history)
                if len(history_df) > smooth_window:
                    history_df['smoothed_reward'] = history_df['total_reward'].rolling(window=smooth_window).mean()
                else:
                    history_df['smoothed_reward'] = history_df['total_reward']
                
                # 更新训练曲线
                fig = px.line(
                    history_df, 
                    x="episode", 
                    y=["total_reward", "smoothed_reward"], 
                    title="训练奖励曲线 (包含移动平均线)",
                    labels={"value": "奖励", "variable": "指标"}
                )
                fig.update_layout(
                    xaxis_title="训练轮次",
                    yaxis_title="总奖励",
                    hovermode="x unified",
                    legend=dict(
                        title="",
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    autosize=True,
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                # 添加参考线显示最佳奖励
                max_reward = history_df['total_reward'].max()
                max_reward_episode = history_df.loc[history_df['total_reward'].idxmax(), 'episode']
                fig.add_hline(
                    y=max_reward, 
                    line_dash="dash", 
                    line_color="green",
                    annotation_text=f"最大奖励: {max_reward:.1f} (第{max_reward_episode}轮)",
                    annotation_position="top right"
                )
                reward_chart.plotly_chart(fig, use_container_width=True)
                
                # 更新状态覆盖率曲线
                if 'cumulative_coverage_rate' in history_df.columns:
                    fig = px.line(
                        history_df,
                        x="episode",
                        y=["episode_coverage_rate", "cumulative_coverage_rate"],
                        title="状态空间覆盖率分析",
                        labels={"value": "覆盖率", "variable": "指标类型"}
                    )
                    fig.update_layout(
                        xaxis_title="训练轮次",
                        yaxis_title="覆盖率 (%)",
                        hovermode="x unified",
                        legend=dict(
                            title="",
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        ),
                        autosize=True,
                        margin=dict(l=20, r=20, t=50, b=20)
                    )
                    fig.update_yaxes(range=[0, 1], tickformat=".0%")
                    
                    # 添加目标覆盖率参考线
                    fig.add_hline(
                        y=0.8, 
                        line_dash="dash", 
                        line_color="green",
                        annotation_text="理想覆盖率: 80%",
                        annotation_position="top right"
                    )
                    
                    coverage_chart.plotly_chart(fig, use_container_width=True)
                
                # 显示访问统计
                if batch == num_batches - 1:  # 只在最后一批次显示
                    final_coverage = visit_stats['final_coverage_rate']
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("状态-动作对总访问次数", f"{visit_stats['total_visits']}")
                    col2.metric("最大访问次数", f"{visit_stats['max_visits']}")
                    col3.metric("最小访问次数", f"{visit_stats['min_visits']}")
                    col4.metric("状态空间覆盖率", f"{final_coverage:.1%}", 
                               delta=f"{len(visit_stats['all_visited_states'])}/{env.n_market_states * env.n_inventory_levels}个状态")
                    
                    if final_coverage < 0.5:
                        st.warning("""
                        **状态空间覆盖率低于50%**，这会影响模型学习的全面性。建议：
                        1. 增加市场跳变概率
                        2. 扩大初始库存随机范围
                        3. 增加补货随机波动
                        4. 确保启用周期性强制探索
                        5. 增加训练轮数
                        """)
                    elif final_coverage < 0.8:
                        st.info(f"状态空间覆盖率为{final_coverage:.1%}，还有提升空间。考虑调整市场跳变概率和初始库存随机范围。")
                    else:
                        st.success(f"状态空间覆盖率达到{final_coverage:.1%}，模型能够全面学习不同状态下的策略。")
                
                # 更新Q值热图 (只在最后一个批次显示)
                if batch == num_batches - 1:
                    # 保存最终Q表和访问统计以便后续显示
                    final_q_table = q_table.copy()
                    final_visit_stats = visit_stats.copy()
        
        # 训练完成后展示最优策略
        st.header("最优定价策略")
        st.markdown("""
        训练完成后，我们可以提取学习到的最优定价策略。下面的热图展示了在不同市场状态和库存水平组合下，
        模型认为的最优价格。这个策略表格可以直接指导实际业务中的定价决策。
        """)
        
        # 最优动作矩阵
        optimal_actions = np.zeros((env.n_market_states, env.n_inventory_levels), dtype=int)
        for m in range(env.n_market_states):
            for i in range(env.n_inventory_levels):
                optimal_actions[m, i] = np.argmax(q_table[m, i, :])
        
        # 最优价格矩阵
        optimal_prices = np.array([[env.price_actions[a] for a in row] for row in optimal_actions])
        
        # 创建热图
        opt_df = pd.DataFrame(
            optimal_prices,
            index=env.market_states,
            columns=env.inventory_levels[:-1]  # 使用与Q表形状匹配的索引（去掉最后一个）
        )
        
        fig = px.imshow(
            opt_df,
            title="最优价格策略图",
            text_auto='.2f',
            color_continuous_scale="Viridis",
            labels=dict(x="库存水平", y="市场状态", color="最优价格")
        )
        fig.update_layout(
            xaxis_title="库存水平",
            yaxis_title="市场状态",
            autosize=True,
            margin=dict(l=20, r=20, t=50, b=20),
            title_font=dict(size=20),
            font=dict(size=14)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 策略分析
        st.subheader("策略分析")
        st.markdown("""
        以下是对学习到的最优策略的分析，帮助理解模型如何根据不同条件调整价格：
        """)
        
        # 计算分析数据
        fire_avg = np.mean(optimal_prices[4, :])
        low_avg = np.mean(optimal_prices[0, :])
        market_effect = (fire_avg - low_avg) / 4
        
        high_inv_avg = np.mean(optimal_prices[:, 4])
        low_inv_avg = np.mean(optimal_prices[:, 0])
        inventory_effect = low_inv_avg - high_inv_avg
        
        highest_price_loc = np.unravel_index(np.argmax(optimal_prices), optimal_prices.shape)
        lowest_price_loc = np.unravel_index(np.argmin(optimal_prices), optimal_prices.shape)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            ### 市场状态对价格的影响
            
            - 在火爆市场时，平均最优价格为 **{fire_avg:.2f}**
            - 在低迷市场时，平均最优价格为 **{low_avg:.2f}**
            - 市场状态每提高一级，最优价格平均增加 **{market_effect:.2f}**
            
            这符合经济学理论：需求旺盛时，价格可以提高以获取更高利润。
            """)
            
            st.markdown(f"""
            ### 极端情况分析
            
            - 最高价格 **{np.max(optimal_prices):.2f}** 出现在 **{env.market_states[highest_price_loc[0]]}+{env.inventory_levels[highest_price_loc[1]]}** 情况下
            - 最低价格 **{np.min(optimal_prices):.2f}** 出现在 **{env.market_states[lowest_price_loc[0]]}+{env.inventory_levels[lowest_price_loc[1]]}** 情况下
            
            可以看出市场状态和库存水平的组合对价格决策的综合影响。
            """)
        
        with col2:
            st.markdown(f"""
            ### 库存水平对价格的影响
            
            - 在极高库存时，平均最优价格为 **{high_inv_avg:.2f}**
            - 在极低库存时，平均最优价格为 **{low_inv_avg:.2f}**
            - 库存减少时，价格策略倾向于 **{'提高' if inventory_effect > 0 else '降低'}** (差异: {abs(inventory_effect):.2f})
            
            这说明模型已经学会了 **{'库存不足时提高价格控制销量' if inventory_effect > 0 else '库存过高时降价促进销售'}**。
            """)
            
            st.markdown(f"""
            ### 决策建议
            
            1. **市场繁荣时**：适当提高价格，充分利用消费者的高支付意愿
            2. **市场低迷时**：考虑降价促销，刺激需求
            3. **库存过高时**：适当降价以加速库存周转，减少持有成本
            4. **库存不足时**：根据市场状态调整价格，在热门市场可提高价格减缓销售速度
            5. **综合策略**：应根据市场状态和库存水平综合考虑，找到平衡点
            """)
        
        # 训练质量评估
        st.header("训练质量评估")
        if len(all_history) > 0:
            last_100_rewards = history_df.tail(100)['total_reward']
            avg_reward = last_100_rewards.mean()
            reward_std = last_100_rewards.std()
            max_reward = history_df['total_reward'].max()
            
            st.metric("最后100轮平均奖励", f"{avg_reward:.2f}", delta=f"±{reward_std:.2f}")
            
            stability = 1 - (reward_std / abs(avg_reward)) if avg_reward != 0 else 0
            
            # 获取最终状态空间覆盖率
            final_coverage = visit_stats['final_coverage_rate']
            
            # 综合评分 = 奖励稳定性(40%) + 状态覆盖率(40%) + 非零Q值比例(20%)
            non_zero_q = np.count_nonzero(q_table)
            total_q = q_table.size
            q_coverage = non_zero_q / total_q
            
            overall_score = 0.4 * stability + 0.4 * final_coverage + 0.2 * q_coverage
            
            col1, col2, col3 = st.columns(3)
            col1.metric("奖励稳定性", f"{stability:.2%}")
            col2.metric("状态空间覆盖率", f"{final_coverage:.2%}")
            col3.metric("Q表非零值比例", f"{q_coverage:.2%}")
            
            st.metric("训练质量综合评分", f"{overall_score:.2%}")
            
            if overall_score > 0.8:
                st.success(f"训练结果优秀 (评分: {overall_score:.2%})，模型具有很好的泛化能力和稳定性。")
            elif overall_score > 0.6:
                st.info(f"训练结果良好 (评分: {overall_score:.2%})，但仍有提升空间。")
            else:
                st.warning(f"训练结果一般 (评分: {overall_score:.2%})，建议调整参数后重新训练。")
            
            # 奖励分析
            if stability > 0.8:
                st.success(f"训练已经充分收敛 (稳定性: {stability:.2%})")
            elif stability > 0.5:
                st.info(f"训练基本收敛，但还有波动 (稳定性: {stability:.2%})")
            else:
                st.warning(f"训练尚未充分收敛，建议增加训练轮数 (稳定性: {stability:.2%})")
            
            # 状态覆盖分析
            not_visited = env.n_market_states * env.n_inventory_levels - len(visit_stats['all_visited_states'])
            if not_visited > 0:
                st.info(f"有 {not_visited} 个状态组合未被访问（共 {env.n_market_states * env.n_inventory_levels} 个）。")
                
                # 分析未访问状态
                visited_states = list(visit_stats['all_visited_states'])
                visited_markets = set([s[0] for s in visited_states])
                visited_inventories = set([s[1] for s in visited_states])
                
                missing_markets = [env.market_states[i] for i in range(env.n_market_states) if i not in visited_markets]
                missing_inventories = [str(env.inventory_levels[i]) for i in range(env.n_inventory_levels) if i not in visited_inventories]
                
                if missing_markets:
                    st.warning(f"未访问的市场状态: {', '.join(missing_markets)}")
                if missing_inventories:
                    st.warning(f"未访问的库存水平: {', '.join(missing_inventories)}")
            
            # Q值分布直方图
            fig = px.histogram(
                q_table.flatten(), 
                title="Q值分布直方图",
                labels={'value': 'Q值', 'count': '频次'},
                nbins=50
            )
            fig.update_layout(
                autosize=True,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

        # 展示模拟交易过程
        st.header("模拟交易过程")
        st.markdown("""
        点击下方按钮，使用学习到的策略在新的市场环境中进行一次完整的模拟交易。
        这将展示模型在实际应用中的表现，包括每日的定价决策、销量、库存变化和利润。
        """)
        
        if st.button("运行模拟交易", key="run_simulation"):
            # 创建新环境进行测试
            test_env = DynamicPricingGymEnv(config=env_config)
            observation, info = test_env.reset(seed=42)  # 固定种子以便重现
            
            # 运行一个完整的模拟交易周期
            terminated = False
            truncated = False
            
            # 记录数据
            simulation_data = []
            
            with st.spinner("模拟中..."):
                while not (terminated or truncated):
                    # 获取当前状态
                    market_state = observation["market_state"]
                    inventory_level = observation["inventory_level"]
                    
                    # 使用学习到的策略
                    action = np.argmax(q_table[market_state, inventory_level])
                    
                    # 执行动作
                    next_observation, reward, terminated, truncated, info = test_env.step(action)
                    
                    # 记录数据
                    simulation_data.append({
                        "天数": len(simulation_data) + 1,
                        "市场状态": info["market_state_name"],
                        "库存": observation["inventory"][0],
                        "库存水平": info["inventory_level_name"],
                        "设定价格": info["price"],
                        "需求量": info["demand"],
                        "销量": info["sales"],
                        "是否缺货": info["is_stockout"],
                        "利润": info["profit"]
                    })
                    
                    # 更新观察
                    observation = next_observation
                
                # 创建数据表
                sim_df = pd.DataFrame(simulation_data)
                
                # 利润和库存曲线
                fig1 = px.line(
                    sim_df, 
                    x="天数", 
                    y=["利润", "库存"], 
                    title="日常交易指标"
                )
                fig1.update_layout(
                    xaxis_title="天数",
                    yaxis_title="数值",
                    legend_title="指标",
                    hovermode="x unified"
                )
                st.plotly_chart(fig1, use_container_width=True)
                
                # 价格和销量曲线
                fig2 = px.line(
                    sim_df, 
                    x="天数", 
                    y=["设定价格", "需求量", "销量"],
                    title="价格与销量关系"
                )
                fig2.update_layout(
                    xaxis_title="天数",
                    yaxis_title="数值",
                    legend_title="指标",
                    hovermode="x unified"
                )
                st.plotly_chart(fig2, use_container_width=True)
                
                # 库存和市场状态散点图
                fig3 = px.scatter(
                    sim_df,
                    x="库存",
                    y="设定价格",
                    color="市场状态",
                    size="销量",
                    hover_data=["天数", "利润"],
                    title="库存-价格-市场状态关系"
                )
                fig3.update_layout(
                    xaxis_title="库存",
                    yaxis_title="设定价格",
                    legend_title="市场状态"
                )
                st.plotly_chart(fig3, use_container_width=True)
                
                # 汇总统计
                st.subheader("模拟交易结果统计")
                
                total_profit = sim_df["利润"].sum()
                avg_profit = sim_df["利润"].mean()
                total_sales = sim_df["销量"].sum()
                stockout_days = sim_df["是否缺货"].sum()
                stockout_rate = stockout_days / len(sim_df)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("总利润", f"{total_profit:.2f}")
                col2.metric("平均日利润", f"{avg_profit:.2f}")
                col3.metric("总销量", f"{total_sales:.0f}")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("缺货天数", f"{stockout_days}")
                col2.metric("缺货率", f"{stockout_rate:.2%}")
                col3.metric("模拟总天数", f"{len(sim_df)}")
                
                # 市场状态分析
                st.subheader("不同市场状态下的表现")
                market_stats = sim_df.groupby("市场状态").agg({
                    "利润": ["mean", "sum"],
                    "设定价格": "mean",
                    "销量": ["mean", "sum"],
                    "是否缺货": "mean"
                }).reset_index()
                
                market_stats.columns = ["市场状态", "平均利润", "总利润", "平均价格", "平均销量", "总销量", "缺货率"]
                
                fig4 = px.bar(
                    market_stats,
                    x="市场状态",
                    y="平均利润",
                    color="市场状态",
                    title="不同市场状态下的平均利润"
                )
                st.plotly_chart(fig4, use_container_width=True)
                
                st.table(market_stats)
                
                # 显示完整数据表
                with st.expander("查看完整模拟数据"):
                    st.dataframe(sim_df)
    
    # Gymnasium环境的优势
    with st.expander("了解Gymnasium框架的优势"):
        st.header("Gymnasium环境的优势")
        st.markdown("""
        本实现使用Gymnasium框架重写了动态定价环境，相比传统实现具有以下优势：
        
        1. **标准化接口**：遵循Gymnasium的标准接口（reset、step、render），更易与现有强化学习算法集成
        
        2. **更完善的随机性控制**：通过seed参数实现可重现的随机性，便于实验结果的复现
        
        3. **丰富的观察空间**：使用Dict空间组合多种观察类型，包括离散状态和连续库存值
        
        4. **更严格的动作验证**：通过action_space自动验证动作的合法性
        
        5. **更好的可扩展性**：容易添加新的特性和功能扩展
        
        6. **与先进算法兼容**：可以直接与Stable Baselines3等现代强化学习库集成
        
        7. **更好的调试和可视化支持**：提供标准化的渲染接口，方便监控训练过程
        """)
        
        st.markdown("""
        ### 未来拓展方向
        
        1. **多产品协同定价**：考虑产品间的互补和替代关系
        2. **竞争对手因素**：将竞争对手的定价策略纳入状态空间  
        3. **客户分群差异化定价**：针对不同客户群体采用差异化定价策略
        4. **季节性因素**：增加季节性特征，更好地模拟实际业务场景
        5. **深度强化学习**：使用DQN等深度强化学习算法处理更复杂的状态空间
        """)
    
    # 添加引用信息
    st.markdown("""
    ---
    ### 参考资料
    
    - Gymnasium文档: [https://gymnasium.farama.org/](https://gymnasium.farama.org/)
    - 强化学习理论: [Sutton & Barto, Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)
    - 价格弹性需求模型: [Price Elasticity of Demand](https://en.wikipedia.org/wiki/Price_elasticity_of_demand)
    """)

    # 在训练完成后显示Q值热图
    if final_q_table is not None and final_visit_stats is not None:
        with q_value_container:
            st.subheader("Q值热图 (各市场状态下)")
            st.markdown(f"""
            以下热图展示了在不同市场状态和库存水平下，各价格动作的Q值（预期长期收益）。
            颜色越深表示该动作的预期收益越高，模型会倾向于选择这些动作。
            
            **训练完成**
            - 最终状态空间覆盖率: **{final_visit_stats['final_coverage_rate']:.1%}**
            - 未访问的状态-动作对: **{final_visit_stats['zero_visit_pairs']}个**
            """)
            
            # 计算Q值范围，用于一致的颜色映射
            q_min = np.min(final_q_table)
            q_max = np.max(final_q_table)
            
            # 创建列来水平排列热图
            market_cols = st.columns(3)
            col_index = 0
            
            for m_idx, market in enumerate(env.market_states):
                # 循环使用3列来排列热图
                with market_cols[col_index % 3]:
                    q_values = final_q_table[m_idx]
                    q_df = pd.DataFrame(
                        q_values,
                        index=env.inventory_levels[:-1],  # 使用与Q表形状匹配的索引（去掉最后一个）
                        columns=[f"{p:.1f}" for p in env.price_actions]
                    )
                    
                    fig = px.imshow(
                        q_df,
                        title=f"市场状态: {market}",
                        labels=dict(x="价格", y="库存水平", color="Q值"),
                        color_continuous_scale="RdBu_r",
                        range_color=[q_min, q_max]  # 统一颜色范围
                    )
                    fig.update_layout(
                        autosize=True,
                        margin=dict(l=20, r=20, t=50, b=20),
                        title_font=dict(size=18),
                        font=dict(size=14)
                    )
                    # 显示Q值
                    for i in range(len(env.inventory_levels[:-1])):
                        for j in range(len(env.price_actions)):
                            fig.add_annotation(
                                x=j, y=i,
                                text=f"{q_values[i, j]:.1f}",
                                showarrow=False,
                                font=dict(
                                    color="white" if abs(q_values[i, j]) > (q_max - q_min)/2 + q_min else "black",
                                    size=12
                                )
                            )
                    st.plotly_chart(fig, use_container_width=True)
                    col_index += 1

if __name__ == "__main__":
    show() 