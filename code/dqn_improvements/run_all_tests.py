"""
DQN改进技术测试套件

该脚本顺序运行所有DQN改进技术的测试，包括：
1. 双重DQN - 解决过估计问题
2. 双重网络DQN - 分离状态价值和优势
3. 优先经验回放DQN - 根据TD误差优先采样
4. 多步学习DQN - 加速价值传播
5. 噪声网络DQN - 自适应探索
6. Rainbow DQN - 结合以上所有技术
"""

import os
import subprocess
import time

# 设置环境变量减少TensorFlow日志（只显示警告和错误）
# 移除了禁用GPU的设置，因为我们已经在各测试文件中添加了GPU支持
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def run_test(script_name, description):
    """运行测试脚本并显示进度"""
    print(f"\n{'='*80}")
    print(f"运行 {script_name} - {description}")
    print(f"{'='*80}")
    
    # 使用subprocess运行脚本，注意添加正确的相对路径
    script_path = f'code/dqn_improvements/{script_name}'
    start_time = time.time()
    try:
        result = subprocess.run(['python', script_path], 
                               check=True, 
                               text=True, 
                               capture_output=True)
        print(result.stdout)
        if result.stderr:
            print(f"警告/错误: {result.stderr}")
        print(f"完成! 用时: {time.time() - start_time:.2f} 秒")
        return True
    except subprocess.CalledProcessError as e:
        print(f"错误运行 {script_name}: {e}")
        print(f"输出: {e.stdout}")
        print(f"错误: {e.stderr}")
        return False

def main():
    """运行所有测试"""
    print("开始运行DQN改进技术测试套件...")
    
    # 定义要运行的测试
    tests = [
        ("double_dqn_test.py", "双重DQN - 解决过估计问题"),
        ("dueling_dqn_test.py", "双重网络DQN - 分离状态价值和优势"),
        ("prioritized_replay_test.py", "优先经验回放DQN - 根据TD误差优先采样"),
        ("n_step_learning_test.py", "多步学习DQN - 加速价值传播"),
        ("noisy_nets_test.py", "噪声网络DQN - 自适应探索"),
        ("simple_rainbow_test.py", "Rainbow DQN - 结合所有改进技术")
    ]
    
    # 运行每个测试
    results = {}
    for script, desc in tests:
        success = run_test(script, desc)
        results[script] = success
    
    # 显示结果摘要
    print("\n\n")
    print("=" * 50)
    print("测试结果摘要")
    print("=" * 50)
    
    all_success = True
    for script, desc in tests:
        status = "通过 ✓" if results[script] else "失败 ✗"
        all_success = all_success and results[script]
        print(f"{script}: {status}")
    
    print("\n总体结果:", "全部通过 ✓" if all_success else "部分失败 ✗")
    print("请查看生成的图表以比较各种DQN改进技术的性能")

if __name__ == "__main__":
    main() 