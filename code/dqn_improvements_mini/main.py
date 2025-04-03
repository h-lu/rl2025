"""
DQN改进算法 - 实验运行入口

本脚本提供一个简单的菜单界面，可以选择运行以下实验：
1. 基础DQN (dqn_mini.py)
2. Double DQN (double_dqn_mini.py)
3. Dueling DQN (dueling_dqn_mini.py)
4. 优先经验回放DQN (prioritized_replay_mini.py)
5. 所有算法对比实验 (run_experiments.py)

使用方法: python main.py
"""

import os
import sys
import time
import run_experiments
import run_single_dqn

def clear_screen():
    """清屏"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """打印标题"""
    print("\n" + "="*50)
    print("       DQN改进算法实验 - Blackjack环境")
    print("="*50)
    print("\n该程序提供了四种DQN算法的实现:")
    print("1. 基础DQN - 标准的深度Q学习")
    print("2. Double DQN - 解决Q值过高估计问题")
    print("3. Dueling DQN - 分离状态价值和动作优势函数")
    print("4. 优先经验回放DQN - 更有效地利用重要经验")
    print("\n所有算法均针对Blackjack-v1环境进行了优化")
    print("="*50)

def print_menu():
    """打印菜单"""
    print("\n请选择要运行的实验:")
    print("1. 运行基础DQN")
    print("2. 运行Double DQN")
    print("3. 运行Dueling DQN")
    print("4. 运行优先经验回放DQN")
    print("5. 运行所有算法对比实验")
    print("0. 退出程序")
    print("\n选择 [0-5]: ", end="")

def run_algorithm(algo):
    """运行单个算法"""
    sys.argv = ['run_single_dqn.py', '--algo', algo]
    try:
        run_single_dqn.main()
    except Exception as e:
        print(f"\n运行时发生错误: {e}")
        input("\n按回车键继续...")

def run_all_experiments():
    """运行所有算法对比实验"""
    try:
        run_experiments.main()
    except Exception as e:
        print(f"\n运行时发生错误: {e}")
        input("\n按回车键继续...")

def main():
    """主函数"""
    while True:
        clear_screen()
        print_header()
        print_menu()
        
        try:
            choice = input().strip()
            if choice == '0':
                print("\n谢谢使用，再见！")
                break
                
            elif choice == '1':
                run_algorithm('basic')
                
            elif choice == '2':
                run_algorithm('double')
                
            elif choice == '3':
                run_algorithm('dueling')
                
            elif choice == '4':
                run_algorithm('prioritized')
                
            elif choice == '5':
                run_all_experiments()
                
            else:
                print("\n无效的选择，请重试。")
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n\n操作被用户中断。")
            break
        except Exception as e:
            print(f"\n发生错误: {e}")
            time.sleep(2)

if __name__ == "__main__":
    main() 