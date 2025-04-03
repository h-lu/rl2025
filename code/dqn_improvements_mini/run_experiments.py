"""
DQN改进算法比较实验

该脚本运行并比较四种DQN改进算法：
1. 基础DQN
2. Double DQN
3. Dueling DQN
4. 优先经验回放DQN

对比实验在Blackjack-v1环境中进行，并保存学习曲线和性能指标。
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import platform
import gymnasium as gym
from collections import defaultdict
import pandas as pd

# 导入各算法
from dqn_mini import DQNAgent as BasicDQN
from double_dqn_mini import DoubleDQNAgent 
from dueling_dqn_mini import DuelingDQNAgent
from prioritized_replay_mini import PrioritizedDQNAgent

# 确保输出目录存在
results_dir = "results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    
models_dir = "models"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# 设置随机种子
SEED = 42
np.random.seed(SEED)

# 环境参数
ENV_NAME = 'Blackjack-v1'
STATE_SIZE = 3  # [玩家点数, 庄家牌点, 是否有可用A]
ACTION_SIZE = 2  # [停牌, 要牌]

# 训练参数
TRAIN_EPISODES = 2000  # 训练回合数，降低以加快测试
EVAL_FREQ = 1000  # 评估频率
EVAL_EPISODES = 1000  # 每次评估的回合数

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

def preprocess_state(state):
    """预处理状态"""
    player_score, dealer_score, usable_ace = state
    return np.array([player_score, dealer_score, int(usable_ace)])

def evaluate_agent(agent, env_name, n_episodes=1000, verbose=True):
    """评估智能体性能"""
    env = gym.make(env_name)
    wins = 0
    draws = 0
    losses = 0
    
    for i in range(n_episodes):
        state, _ = env.reset(seed=SEED+i+10000)
        state = preprocess_state(state)
        done = False
        
        while not done:
            action = agent.act(state, eval_mode=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = preprocess_state(next_state)
            done = terminated or truncated
            state = next_state
        
        if reward > 0:
            wins += 1
        elif reward == 0:
            draws += 1
        else:
            losses += 1
    
    win_rate = wins / n_episodes
    draw_rate = draws / n_episodes
    loss_rate = losses / n_episodes
    
    if verbose:
        print(f'评估 {n_episodes} 回合:')
        print(f'获胜率: {win_rate:.3f}')
        print(f'平局率: {draw_rate:.3f}')
        print(f'失败率: {loss_rate:.3f}')
    
    return win_rate, draw_rate, loss_rate

def train_and_evaluate(agent, agent_name, env_name=ENV_NAME):
    """训练并评估智能体，返回学习曲线和性能指标"""
    print(f"\n开始训练 {agent_name}...")
    start_time = time.time()
    
    print("开始准备训练环境...")
    env = gym.make(env_name)
    scores = []
    avg_scores = []
    winrates = []
    draw_rates = []
    loss_rates = []
    eval_steps = []
    
    print(f"正在初始化智能体和回放缓冲区...")
    if hasattr(agent, 'train_start'):
        print(f"目标回放缓冲区大小: {agent.train_start} 个样本")
        if hasattr(agent, 'memory') and hasattr(agent.memory, '__len__'):
            current_buffer_size = len(agent.memory)
            print(f"当前缓冲区大小: {current_buffer_size} 个样本")
    
    # 训练循环
    for i_episode in range(1, TRAIN_EPISODES+1):
        state, _ = env.reset(seed=SEED+i_episode)
        state = preprocess_state(state)
        done = False
        score = 0
        steps = 0
        
        # 单局游戏循环
        while not done:
            # 选择动作
            action = agent.act(state)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = preprocess_state(next_state)
            done = terminated or truncated
            
            # 学习
            agent.step(state, action, reward, next_state, done)
            
            # 更新状态和分数
            state = next_state
            score += reward
            steps += 1
            
            if done:
                break
        
        # 记录分数
        scores.append(score)
        
        # 计算滑动平均
        window_size = min(100, len(scores))
        avg_score = np.mean(scores[-window_size:])
        avg_scores.append(avg_score)
        
        # 更频繁地打印进度
        if i_episode == 1 or i_episode % 100 == 0 or i_episode == TRAIN_EPISODES:
            elapsed_time = time.time() - start_time
            if hasattr(agent, 'memory') and hasattr(agent.memory, '__len__'):
                current_buffer_size = len(agent.memory)
                buffer_status = f"缓冲区: {current_buffer_size}"
                if hasattr(agent, 'train_start'):
                    buffer_status += f"/{agent.train_start}"
            else:
                buffer_status = ""
                
            agent_status = f"{agent_name}: 回合 {i_episode}/{TRAIN_EPISODES}"
            score_status = f"分数: {score}\t平均: {avg_score:.3f}"
            if hasattr(agent, 'epsilon'):
                agent_status += f"\t探索率: {agent.epsilon:.4f}"
            if buffer_status:
                agent_status += f"\t{buffer_status}"
            
            print(f"{agent_status}\t{score_status}\t用时: {elapsed_time:.1f}秒")
            
        # 定期评估
        if i_episode % EVAL_FREQ == 0:
            print(f"\n开始第{i_episode//EVAL_FREQ}次评估 ({EVAL_FREQ}回合训练后)...")
            # 处理不同算法模块可能返回不同格式的情况
            eval_result = evaluate_agent(agent, env_name, n_episodes=EVAL_EPISODES, verbose=False)
            
            # 如果返回的是三元组 (win_rate, draw_rate, loss_rate)
            if isinstance(eval_result, tuple) and len(eval_result) == 3:
                win_rate, draw_rate, loss_rate = eval_result
            # 如果只返回win_rate
            else:
                win_rate = eval_result
                draw_rate = None
                loss_rate = None
                
            winrates.append(win_rate)
            if draw_rate is not None:
                draw_rates.append(draw_rate)
                loss_rates.append(loss_rate)
            eval_steps.append(i_episode)
            print(f'评估完成: {agent_name} 获胜率: {win_rate:.3f}\n')
    
    # 记录结束时间和总训练时间
    total_time = time.time() - start_time
    print(f"\n{agent_name} 训练完成! 总用时: {total_time:.2f}秒")
    
    # 保存模型
    model_path = os.path.join(models_dir, f"{agent_name.lower().replace(' ', '_')}_model.h5")
    agent.save(model_path)
    print(f"模型已保存至: {model_path}")
    
    # 返回结果
    return {
        'name': agent_name,
        'scores': scores,
        'avg_scores': avg_scores,
        'winrates': winrates,
        'draw_rates': draw_rates if draw_rates else [],
        'loss_rates': loss_rates if loss_rates else [],
        'eval_steps': eval_steps,
        'training_time': total_time
    }

def plot_learning_curves(results, save_path=None):
    """绘制学习曲线比较图"""
    set_chinese_font()
    plt.figure(figsize=(12, 6))
    
    for result in results:
        plt.plot(result['avg_scores'], label=result['name'])
    
    plt.xlabel('回合')
    plt.ylabel('平均分数 (100回合滑动窗口)')
    plt.title('各种DQN算法在Blackjack环境中的学习曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_win_rates(results, save_path=None):
    """绘制获胜率比较图"""
    set_chinese_font()
    plt.figure(figsize=(12, 6))
    
    for result in results:
        plt.plot(result['eval_steps'], result['winrates'], 
                 marker='o', label=result['name'])
    
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='随机策略')
    plt.xlabel('训练回合')
    plt.ylabel('获胜率')
    plt.title('各种DQN算法在Blackjack环境中的获胜率')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def create_performance_table(results):
    """创建性能对比表格"""
    data = {
        '算法': [],
        '最终获胜率': [],
        '最高获胜率': [],
        '训练时间(秒)': []
    }
    
    # 检查是否所有结果都有平局率和失败率
    has_draw_loss_rates = all(len(result['draw_rates']) > 0 for result in results)
    
    if has_draw_loss_rates:
        data['最终平局率'] = []
        data['最终失败率'] = []
    
    for result in results:
        data['算法'].append(result['name'])
        data['最终获胜率'].append(result['winrates'][-1])
        data['最高获胜率'].append(max(result['winrates']))
        data['训练时间(秒)'].append(result['training_time'])
        
        if has_draw_loss_rates:
            data['最终平局率'].append(result['draw_rates'][-1])
            data['最终失败率'].append(result['loss_rates'][-1])
    
    return pd.DataFrame(data)

def save_results_to_csv(results):
    """保存结果到CSV文件"""
    for result in results:
        # 保存学习曲线数据
        learning_data = pd.DataFrame({
            '回合': range(1, len(result['scores'])+1),
            '分数': result['scores'],
            '平均分数': result['avg_scores']
        })
        learning_csv = os.path.join(results_dir, f"{result['name'].lower().replace(' ', '_')}_learning.csv")
        learning_data.to_csv(learning_csv, index=False)
        
        # 保存评估数据
        eval_data = pd.DataFrame({
            '回合': result['eval_steps'],
            '获胜率': result['winrates'],
            '平局率': result['draw_rates'],
            '失败率': result['loss_rates']
        })
        eval_csv = os.path.join(results_dir, f"{result['name'].lower().replace(' ', '_')}_evaluation.csv")
        eval_data.to_csv(eval_csv, index=False)
    
    # 保存性能对比表格
    performance_table = create_performance_table(results)
    performance_csv = os.path.join(results_dir, "performance_comparison.csv")
    performance_table.to_csv(performance_csv, index=False)
    
    print(f"\n结果数据已保存至 {results_dir} 目录")

def main():
    """主函数：训练和比较不同的DQN改进算法"""
    print(f"开始在 {ENV_NAME} 环境上比较DQN改进算法...")
    print(f"状态空间大小: {STATE_SIZE}, 动作空间大小: {ACTION_SIZE}")
    
    # 创建各算法智能体
    agents = [
        (BasicDQN(STATE_SIZE, ACTION_SIZE), "基础DQN"),
        (DoubleDQNAgent(STATE_SIZE, ACTION_SIZE), "Double DQN"),
        (DuelingDQNAgent(STATE_SIZE, ACTION_SIZE), "Dueling DQN"),
        (PrioritizedDQNAgent(STATE_SIZE, ACTION_SIZE), "优先经验回放DQN")
    ]
    
    # 训练和评估所有智能体
    results = []
    for agent, name in agents:
        result = train_and_evaluate(agent, name)
        results.append(result)
    
    # 绘制并保存学习曲线
    plot_learning_curves(results, os.path.join(results_dir, "learning_curves_comparison.png"))
    
    # 绘制并保存获胜率曲线
    plot_win_rates(results, os.path.join(results_dir, "win_rates_comparison.png"))
    
    # 创建并显示性能对比表格
    performance_table = create_performance_table(results)
    print("\n性能对比:")
    print(performance_table)
    
    # 保存结果数据
    save_results_to_csv(results)
    
    print("\n实验完成!")

if __name__ == "__main__":
    main() 