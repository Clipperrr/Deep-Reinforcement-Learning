import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import random
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# DQN 网络模型
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 经验回放类
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)


# DQN 代理类
class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 learning_rate=0.001, batch_size=64, buffer_size=100000):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size

        # 创建 Q 网络和目标网络
        self.q_network = DQN(state_size, action_size).to(device)
        self.target_network = DQN(state_size, action_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())  # 初始化目标网络

        # 设置优化器
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=learning_rate)

        # 经验回放
        self.memory = ReplayBuffer(buffer_size)

    def act(self, state):
        # 贪婪策略：探索或利用
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # 随机选择动作
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)  # 转换为 tensor
            q_values = self.q_network(state)  # 计算 Q 值
            return torch.argmax(q_values[0]).item()  # 返回最大 Q 值对应的动作


    def learn(self):
        if self.memory.size() < self.batch_size:
            return

        # 从经验池中随机采样一批经验
        minibatch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # 转换为 tensor
        states = torch.tensor(states, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.int64).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)

        # 计算当前 Q 值
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # 计算目标 Q 值
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # 计算损失
        loss = nn.MSELoss()(q_values.squeeze(1), target_q_values)

        # 反向传播和优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新 epsilon 值
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())


# 环境初始化
env = gym.make('CartPole-v0')

# DQN 代理初始化
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)

# 训练
num_episodes = 50
for e in range(num_episodes):
    state = env.reset()[0]
    total_reward = 0
    done = False
    while not done:
        action = agent.act(state)  # 选择动作
        next_state, reward, done, truncated, info = env.step(action)

        # 存储经验
        agent.memory.add((state, action, reward, next_state, done))
        # 学习
        agent.learn()
        state = next_state
        total_reward += reward

    agent.update_target_network()  # 每一轮后更新目标网络

    print(f"Episode {e + 1}/{num_episodes}, Total Reward: {total_reward}")

env.close()
