"""
单个DQN算法运行脚本

该脚本接受命令行参数来指定要运行的DQN变体，并训练和评估该算法。
示例使用方法:
    python run_single_dqn.py --algo basic
    python run_single_dqn.py --algo double
    python run_single_dqn.py --algo dueling
    python run_single_dqn.py --algo prioritized
"""

import os
import sys
import time
import argparse
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import platform
from collections import deque

# 环境参数
ENV_NAME = 'Blackjack-v1'
STATE_SIZE = 3  # [玩家点数, 庄家牌点, 是否有可用A]
ACTION_SIZE = 2  # [停牌, 要牌]

# 训练参数
TRAIN_EPISODES = 2000  # 训练回合数
EVAL_EPISODES = 1000  # 评估回合数

def set_chinese_font():
    """根据操作系统设置中文字体"""
    system = platform.system()
    try:
        if system == 'Windows':
            plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 黑体
            plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
        elif system == 'Darwin': # macOS
            plt.rcParams['font.sans-serif'] = ['PingFang SC'] # Mac 苹方
            plt.rcParams['axes.unicode_minus'] = False
        else: # Linux 或其他
            plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'Arial Unicode MS']
            plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        print(f"设置中文字体时出错: {e}")

def plot_scores(scores, avg_scores, title="学习曲线", save_path=None):
    """绘制学习曲线"""
    set_chinese_font()  # 设置中文字体
    plt.figure(figsize=(10, 5))
    plt.plot(scores, alpha=0.3, color='blue', label='分数')
    plt.plot(avg_scores, color='red', label='平均分数 (100回合)')
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    plt.xlabel('回合')
    plt.ylabel('分数')
    plt.title(title)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_winrate(winrates, title="获胜率", save_path=None):
    """绘制获胜率曲线"""
    set_chinese_font()  # 设置中文字体
    plt.figure(figsize=(10, 5))
    plt.plot(winrates, color='green', label='获胜率')
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='随机策略')
    plt.xlabel('评估次数')
    plt.ylabel('获胜率')
    plt.title(title)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='运行单个DQN变体')
    parser.add_argument('--algo', type=str, required=True, 
                        choices=['basic', 'double', 'dueling', 'prioritized'],
                        help='要运行的算法: basic, double, dueling, prioritized')
    parser.add_argument('--episodes', type=int, default=TRAIN_EPISODES,
                        help=f'训练回合数 (默认: {TRAIN_EPISODES})')
    parser.add_argument('--no-plot', action='store_true', 
                        help='不显示学习曲线和获胜率图')
    parser.add_argument('--save-dir', type=str, default='results',
                        help='保存结果的目录 (默认: results)')
    
    return parser.parse_args()

def main():
    """主函数：根据命令行参数训练和评估指定的DQN变体"""
    args = parse_args()
    
    # 确保输出目录存在
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # 设置中文字体
    set_chinese_font()
    
    algo_name = args.algo
    algo_display_names = {
        'basic': '基础DQN',
        'double': 'Double DQN',
        'dueling': 'Dueling DQN',
        'prioritized': '优先经验回放DQN'
    }
    
    display_name = algo_display_names[algo_name]
    print(f"\n=============== 运行 {display_name} 算法 ===============")
    print(f"环境: {ENV_NAME}")
    print(f"训练回合数: {args.episodes}")
    
    # 根据算法类型导入相应模块
    if algo_name == 'basic':
        from dqn_mini import DQNAgent, train_agent, evaluate_agent
        agent = DQNAgent(STATE_SIZE, ACTION_SIZE)
    elif algo_name == 'double':
        from double_dqn_mini import DoubleDQNAgent, train_agent, evaluate_agent
        agent = DoubleDQNAgent(STATE_SIZE, ACTION_SIZE)
    elif algo_name == 'dueling':
        from dueling_dqn_mini import DuelingDQNAgent, train_agent, evaluate_agent
        agent = DuelingDQNAgent(STATE_SIZE, ACTION_SIZE)
    elif algo_name == 'prioritized':
        from prioritized_replay_mini import PrioritizedDQNAgent, train_agent, evaluate_agent
        agent = PrioritizedDQNAgent(STATE_SIZE, ACTION_SIZE)
    
    # 训练智能体
    print(f"\n开始训练 {display_name}...")
    start_time = time.time()
    
    # 确保我们使用修改后的train_agent函数，增加额外输出
    print(f"正在初始化训练环境和智能体...")
    if hasattr(agent, 'train_start'):
        print(f"目标回放缓冲区大小: {agent.train_start} 个样本")
    
    scores, avg_scores, winrates = train_agent(agent, ENV_NAME, args.episodes)
    train_time = time.time() - start_time
    print(f"\n训练完成! 总用时: {train_time:.2f}秒")
    
    # 绘制并保存学习曲线
    if not args.no_plot:
        plot_scores(scores, avg_scores, 
                   f"{display_name}在{ENV_NAME}环境中的学习曲线", 
                   f"{args.save_dir}/{algo_name}_learning_curve.png")
        
        # 绘制并保存获胜率曲线
        plot_winrate(winrates, 
                    f"{display_name}在{ENV_NAME}环境中的获胜率", 
                    f"{args.save_dir}/{algo_name}_winrate.png")
    
    # 最终评估
    print("\n最终评估:")
    eval_result = evaluate_agent(agent, ENV_NAME, n_episodes=EVAL_EPISODES)
    
    # 处理不同的返回值格式
    if isinstance(eval_result, tuple) and len(eval_result) == 3:
        win_rate, draw_rate, loss_rate = eval_result
        print(f"平局率: {draw_rate:.3f}")
        print(f"失败率: {loss_rate:.3f}")
    else:
        win_rate = eval_result
    
    # 保存模型
    model_dir = 'models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = f"{model_dir}/{algo_name}_model.h5"
    agent.save(model_path)
    print(f"模型已保存至: {model_path}")
    
    # 保存结果数据
    import pandas as pd
    
    # 训练数据
    train_data = pd.DataFrame({
        '回合': range(1, len(scores)+1),
        '分数': scores,
        '平均分数': avg_scores
    })
    train_data.to_csv(f"{args.save_dir}/{algo_name}_training_data.csv", index=False)
    
    # 总结
    print("\n性能总结:")
    print(f"算法: {display_name}")
    print(f"最终获胜率: {win_rate:.3f}")
    print(f"训练时间: {train_time:.2f}秒")
    print(f"训练回合数: {args.episodes}")
    print(f"结果已保存至: {args.save_dir}")

if __name__ == "__main__":
    main() 