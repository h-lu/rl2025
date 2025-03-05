import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import base64
import os

def get_svg_as_base64(file_path):
    """将SVG文件转换为base64编码的字符串"""
    try:
        # 获取当前文件的绝对路径，然后构建SVG文件的完整路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(current_dir, file_path)
        
        if os.path.exists(full_path):
            with open(full_path, "rb") as svg_file:
                svg_content = svg_file.read()
                b64_content = base64.b64encode(svg_content).decode("utf-8")
                return b64_content
        else:
            return None
    except Exception as e:
        print(f"读取SVG文件时出错: {e}")
        return None

def show():
    st.header("Q-Learning理论介绍")
    
    st.markdown("""
    ## Q-Learning算法基础
    
    Q-Learning是一种无模型（model-free）的强化学习算法，能够学习如何在未知环境中做出最优决策。它通过不断尝试与环境交互，逐步优化决策策略。
    
    ### 核心概念：Q函数与Q表
    
    Q函数表示在某状态下采取某动作的预期累积奖励。形式上，Q(s,a)表示在状态s下执行动作a的"质量"或"价值"。
    
    Q表是Q函数的一种表格表示形式，每一行代表一个状态，每一列代表一个动作，单元格中的值代表相应状态-动作对的Q值。
    """)
    
    # 算法流程图
    st.markdown("### Q-Learning算法流程")
    st.markdown("""
    Q-Learning算法遵循以下步骤：
    
    1. **初始化Q表**：为所有状态-动作对设置初始Q值（通常为0）
    2. **交互循环**：
       - 在当前状态选择动作（通常使用ε-贪婪策略）
       - 执行动作，获得奖励和新状态
       - 更新Q值
       - 转移到新状态
    3. **重复**：直到达到终止条件或训练足够的回合数
    """)
    
    # 使用base64内联SVG
    flow_svg_b64 = get_svg_as_base64("images/q_learning_flow.svg")
    if flow_svg_b64:
        st.markdown(f"""
        <div style="display: flex; justify-content: center; margin: 20px 0;">
            <img src="data:image/svg+xml;base64,{flow_svg_b64}" width="700" alt="Q-Learning算法流程图">
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("无法加载算法流程图，请确保SVG文件存在并可访问。")
    
    # 时序差分更新公式
    st.markdown("""
    ### 时序差分（TD）更新公式
    
    Q-Learning的核心是使用时序差分（TD）更新公式来调整Q值，其基本形式为：
    
    $$Q(s,a) \\leftarrow Q(s,a) + \\alpha [r + \\gamma \\max_{a'} Q(s',a') - Q(s,a)]$$
    
    其中：
    
    * **Q(s,a)** 是当前状态-动作对的Q值
    * **α** 是学习率，控制更新步长（通常为0.1到0.5之间）
    * **r** 是执行动作a后获得的即时奖励
    * **γ** 是折扣因子，决定了未来奖励的重要性（通常为0.9左右）
    * **max Q(s',a')** 是下一状态中所有可能动作的最大Q值
    """)
    
    # 使用base64内联TD更新公式SVG
    td_svg_b64 = get_svg_as_base64("images/td_update.svg")
    if td_svg_b64:
        st.markdown(f"""
        <div style="display: flex; justify-content: center; margin: 20px 0;">
            <img src="data:image/svg+xml;base64,{td_svg_b64}" width="700" alt="TD更新公式图">
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("无法加载TD更新公式图，请确保SVG文件存在并可访问。")
    
    # 使用简化的解释
    st.markdown("""
    ### 公式解释（通俗版）
    
    简单来说，TD更新公式做了以下三件事：
    
    1. **看到了什么**：我们获得了即时奖励 r
    2. **希望能得到什么**：当前奖励 + 未来可能的最大收益（打折后）
    3. **调整期望**：根据现实与期望的差距，按比例调整我们的估计值
    
    这个过程类似于我们在生活中学习经验并调整预期的方式。
    
    ### 实际案例
    
    假设：
    * 你在玩迷宫游戏
    * 当前位置A，选择向右移动到位置B
    * 移动花费1能量（奖励-1）
    * 在B处能看到终点距离更近
    
    即使这一步获得了负奖励，Q-Learning也能识别出这个动作长期来看是值得的，因为它会考虑到未来可能获得的更大奖励。
    
    ### 探索与利用平衡
    
    Q-Learning使用ε-贪婪策略来平衡探索与利用：
    
    * **探索**：概率ε随机选择动作，探索未知可能性
    * **利用**：概率(1-ε)选择当前最优动作，利用已有知识
    
    这种平衡确保算法既能找到新的潜在最优路径，又能稳定地沿当前已知的最优路径行动。
    
    ### 主要优点
    
    * **无需环境模型**：直接从交互中学习
    * **离线学习**：可以学习从任何策略生成的经验
    * **收敛性保证**：在适当条件下，保证收敛到最优策略
    """)

if __name__ == "__main__":
    show() 