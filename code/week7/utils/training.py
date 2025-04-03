import numpy as np
import time
import torch
from collections import deque

def train_agent(env, agent, num_episodes=500, max_steps=200, log_interval=10):
    """
    训练DQN智能体
    
    参数:
        env: 训练环境
        agent: DQN智能体
        num_episodes: 训练的回合数
        max_steps: 每个回合的最大步数
        log_interval: 日志打印间隔
        
    返回:
        episode_rewards: 每个回合的累积奖励
        losses: 每次更新的损失值
        epsilons: epsilon值的变化（如果智能体使用epsilon-greedy策略）
    """
    # 记录数据
    episode_rewards = []
    running_reward = deque(maxlen=log_interval)
    losses = []
    epsilons = []
    
    # 训练循环
    start_time = time.time()
    for episode in range(1, num_episodes+1):
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = []
        
        for step in range(max_steps):
            # 选择动作
            action = agent.select_action(state)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 存储经验
            agent.buffer.add(state, action, reward, next_state, done)
            
            # 更新网络
            if len(agent.buffer) > agent.batch_size:
                loss = agent.update()
                episode_loss.append(loss)
                
                # 记录epsilon（如果有）
                if hasattr(agent, 'epsilon'):
                    epsilons.append(agent.epsilon)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # 记录回合奖励
        episode_rewards.append(episode_reward)
        running_reward.append(episode_reward)
        
        # 记录平均损失
        if episode_loss:
            losses.append(np.mean(episode_loss))
        
        # 打印训练信息
        if episode % log_interval == 0:
            avg_reward = np.mean(running_reward)
            avg_loss = np.mean(losses[-log_interval:]) if losses else 0
            elapsed_time = time.time() - start_time
            print(f"Episode {episode}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Epsilon: {agent.epsilon if hasattr(agent, 'epsilon') else 'N/A':.2f} | "
                  f"Time: {elapsed_time:.2f}s")
    
    return episode_rewards, losses, epsilons


def evaluate_agent(env, agent, num_episodes=100):
    """
    评估DQN智能体
    
    参数:
        env: 评估环境
        agent: DQN智能体
        num_episodes: 评估的回合数
        
    返回:
        avg_reward: 平均回合奖励
        success_rate: 成功率（奖励为正的回合比例）
    """
    rewards = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # 使用贪婪策略（不探索）
            if hasattr(agent, 'select_action'):
                action = agent.select_action(state, epsilon=0)
            else:
                action = agent.q_network(torch.FloatTensor(state)).argmax().item()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state
        
        rewards.append(episode_reward)
    
    avg_reward = np.mean(rewards)
    success_rate = np.mean([r > 0 for r in rewards])
    
    return avg_reward, success_rate
