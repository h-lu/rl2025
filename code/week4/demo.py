"""
Q-Learning优化技巧演示脚本
用于展示如何使用迷宫环境和Q-Learning算法
"""

from treasure_maze_env import TreasureMazeEnv
from q_learning_demo import QLearningAgent, compare_exploration_strategies, compare_q_initializations, compare_reward_functions
import traceback
import sys

def main():
    print("Q-Learning优化技巧演示")
    print("=" * 50)
    print("请选择要演示的内容：")
    print("1. 单个智能体学习过程")
    print("2. 比较不同探索策略")
    print("3. 比较不同Q表初始化策略")
    print("4. 比较不同奖励函数")
    print("0. 退出")
    
    choice = input("请输入选项（0-4）：")
    
    try:
        if choice == "1":
            print("\n演示单个智能体学习过程...")
            # 创建环境和智能体
            env = TreasureMazeEnv(render_mode="human", size=7, dense_reward=True)
            agent = QLearningAgent(
                env=env,
                learning_rate=0.1,
                discount_factor=0.99,
                exploration_strategy="epsilon_decay",
                epsilon=0.3,
                epsilon_decay=0.99,
                epsilon_min=0.01,
                q_init_strategy="optimistic",
                q_init_value=5.0
            )
            
            # 训练智能体
            print("开始训练...")
            try:
                agent.train(episodes=100, render=True)
                
                # 绘制训练结果
                agent.plot_training_results()
                
                # 可视化策略
                agent.visualize_policy()
            except Exception as e:
                print(f"训练过程中发生错误: {e}")
                traceback.print_exc()
            finally:
                # 确保环境关闭
                env.close()
        
        elif choice == "2":
            print("\n比较不同探索策略...")
            try:
                compare_exploration_strategies(TreasureMazeEnv, episodes=50, trials=2)
            except Exception as e:
                print(f"比较探索策略时发生错误: {e}")
                traceback.print_exc()
        
        elif choice == "3":
            print("\n比较不同Q表初始化策略...")
            try:
                compare_q_initializations(TreasureMazeEnv, episodes=50, trials=2)
            except Exception as e:
                print(f"比较Q表初始化策略时发生错误: {e}")
                traceback.print_exc()
        
        elif choice == "4":
            print("\n比较不同奖励函数...")
            try:
                compare_reward_functions(episodes=50, trials=2)
            except Exception as e:
                print(f"比较奖励函数时发生错误: {e}")
                traceback.print_exc()
        
        elif choice == "0":
            print("退出程序")
            return
        
        else:
            print("无效选项，请重新运行程序")
    
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序运行时发生错误: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"程序运行时发生严重错误: {e}")
        traceback.print_exc()
        sys.exit(1) 