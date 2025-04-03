import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
from tetris_env import TetrisEnv

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        # 计算卷积层输出尺寸
        conv_out_size = self._get_conv_out(input_shape)
        
        # 全连接层输入尺寸 = 卷积输出 + 额外特征(5个)
        self.fc_input_size = conv_out_size + 5
        
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    
    def _get_conv_out(self, shape):
        # 计算卷积层输出尺寸
        with torch.no_grad():
            x = torch.zeros(1, *shape)
            x = self.conv(x)
            return int(np.prod(x.size()))
    
    def forward(self, x_board, x_features):
        # 处理游戏板输入
        conv_out = self.conv(x_board).view(x_board.size()[0], -1)
        
        # 拼接卷积输出和额外特征
        full_features = torch.cat([conv_out, x_features], dim=1)
        
        # 全连接层
        return self.fc(full_features)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=100000)
        self.gamma = 0.99    # discount rate
        self.epsilon = 1.0    # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.00025
        self.batch_size = 64
        self.update_target_every = 1000
        
        self.model = DQN((1, state_dim[0], state_dim[1]), action_dim)
        self.target_model = DQN((1, state_dim[0], state_dim[1]), action_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        
        self.steps = 0
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            board = torch.FloatTensor(state["board"]).unsqueeze(0).unsqueeze(0)
            features = torch.FloatTensor([
                state["current_piece"] / 6.0,
                state["next_piece"] / 6.0,
                state["position"][0] / 10.0,
                state["position"][1] / 20.0,
                state["rotation"] / 3.0
            ]).unsqueeze(0)
            
            act_values = self.model(board, features)
            return torch.argmax(act_values[0]).item()
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        
        boards = torch.FloatTensor(np.array([s[0]["board"] for s in minibatch])).unsqueeze(1)
        features = torch.FloatTensor(np.array([
            [
                s[0]["current_piece"] / 6.0,
                s[0]["next_piece"] / 6.0,
                s[0]["position"][0] / 10.0,
                s[0]["position"][1] / 20.0,
                s[0]["rotation"] / 3.0
            ] for s in minibatch
        ]))
        actions = torch.LongTensor(np.array([s[1] for s in minibatch]))
        rewards = torch.FloatTensor(np.array([s[2] for s in minibatch]))
        next_boards = torch.FloatTensor(np.array([s[3]["board"] for s in minibatch])).unsqueeze(1)
        next_features = torch.FloatTensor(np.array([
            [
                s[3]["current_piece"] / 6.0,
                s[3]["next_piece"] / 6.0,
                s[3]["position"][0] / 10.0,
                s[3]["position"][1] / 20.0,
                s[3]["rotation"] / 3.0
            ] for s in minibatch
        ]))
        dones = torch.FloatTensor(np.array([s[4] for s in minibatch]))
        
        # 计算当前Q值
        current_q = self.model(boards, features).gather(1, actions.unsqueeze(1))
        
        # 计算目标Q值
        with torch.no_grad():
            next_q = self.target_model(next_boards, next_features).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # 计算损失并更新网络
        loss = self.loss_fn(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # 定期更新目标网络
        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        
        return loss.item()

def train_dqn():
    """训练DQN智能体玩俄罗斯方块
    
    主要流程:
    1. 初始化环境和智能体
    2. 进行多轮训练
    3. 每轮中智能体与环境交互
    4. 存储经验并训练网络
    5. 记录并显示训练指标
    """
    env = TetrisEnv()
    state_dim = (env.board_height, env.board_width)  # 游戏板尺寸
    action_dim = env.action_space.n  # 可选动作数量
    
    # 初始化DQN智能体
    agent = DQNAgent(state_dim, action_dim)
    episodes = 2000  # 总训练轮数
    scores = []  # 记录每轮得分
    losses = []  # 记录每轮平均损失
    best_score = -float('inf')  # 记录最佳得分
    
    # 训练循环
    for e in range(episodes):
        state, _ = env.reset()
        total_reward = 0  # 本轮总奖励
        total_loss = 0    # 本轮总损失
        steps = 0         # 本轮步数
        
        # 单轮游戏循环
        while True:
            # 1. 智能体选择动作
            action = agent.act(state)
            
            # 2. 执行动作并获取新状态
            next_state, reward, done, _, _ = env.step(action)
            
            # 3. 存储经验并训练
            agent.remember(state, action, reward, next_state, done)
            loss = agent.replay()
            if loss is not None:
                total_loss += loss
            
            # 更新状态和统计信息
            state = next_state
            total_reward += reward
            steps += 1
            
            # 游戏结束条件
            if done:
                break
        
        # 计算本轮平均损失
        avg_loss = total_loss / steps if steps > 0 else 0
        scores.append(total_reward)
        losses.append(avg_loss)
        
        # 更新最佳得分
        if total_reward > best_score:
            best_score = total_reward
        
        # 实时显示训练状态
        print(f"\n=== Episode {e+1}/{episodes} ===")
        print(f"Score: {total_reward:.1f} (Best: {best_score:.1f})")
        print(f"Steps: {steps}")
        print(f"Epsilon: {agent.epsilon:.3f} (Exploration rate)")
        print(f"Avg Loss: {avg_loss:.4f}")
        print(f"Memory Size: {len(agent.memory)}")
        
        # 每100轮保存一次模型
        if (e+1) % 100 == 0:
            torch.save(agent.model.state_dict(), f"tetris_dqn_{e+1}.pth")
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(scores)
    plt.title('Score per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    
    plt.subplot(1, 2, 2)
    plt.plot(losses)
    plt.title('Average Loss per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig('training_curve.png')
    plt.show()

if __name__ == "__main__":
    train_dqn()