import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import gymnasium as gym
import random
import numpy as np
from collections import deque
from torch.distributions import Normal
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gymnasium_robotics
import pandas as pd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)
q_loss_list = []
actor_loss_list = []
# DQN 网络模型
class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        self.max_action = max_action

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))#.clone()

        return x * self.max_action

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)
        self.max_size = buffer_size

    def add(self, experience):
        if self.size() == self.max_size:
            self.buffer.popleft()
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)

class DDPG:
    def __init__(self, state_size, action_size, hidden_size, DQN_lr, Actor_lr, max_action):
        self.state_size = state_size
        self.action_size = action_size
        self.target_actor = Actor(state_size, action_size, hidden_size, max_action).to(device)
        self.target_q = DQN(state_size, action_size, hidden_size).to(device)
        self.eval_actor = Actor(state_size, action_size, hidden_size, max_action).to(device)
        self.eval_q = DQN(state_size, action_size, hidden_size).to(device)
        self.Memory = ReplayBuffer(500000)
        self.gamma = 0.98  # 折扣因子
        self.tau = 0.05  # 目标网络软更新系数
        self.noise = 0.3
        self.batch_size = 256

        self.Actor_optimizer = optim.Adam(self.eval_actor.parameters(), lr=Actor_lr)
        self.Q_optimizer = optim.Adam(self.eval_q.parameters(), lr=DQN_lr)

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0)
        action = self.eval_actor(state)
        normal = Normal(0, 1)
        noise = self.noise * normal.sample(action.shape).to(device)
        action += noise
        return action.cpu().detach().numpy().flatten()


    def update(self):
        if self.Memory.size() < self.batch_size:
            return

        # 采样经验回放
        minibatch = self.Memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32).to(device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(device).unsqueeze(1)
        dones = torch.tensor(np.array(dones), dtype=torch.float32).to(device).unsqueeze(1)

        # 计算 Q 值估计
        q_estimate = self.eval_q(states, actions)

        # 计算目标 Q 值（使用目标 Actor 计算下个动作）
        next_actions = self.target_actor(next_states).detach()
        q_target = rewards + self.gamma * (1 - dones) * self.target_q(next_states, next_actions).detach()

        # 计算 Critic（Q 网络）损失并优化
        q_loss = F.mse_loss(q_estimate, q_target)
        q_loss_list.append(q_loss.tolist())
        self.Q_optimizer.zero_grad()
        q_loss.backward()
        self.Q_optimizer.step()

        # 计算 Actor 网络的损失（最大化 Q 值）
        estimate_action = self.eval_actor(states)
        actor_loss = -self.eval_q(states, estimate_action).mean()
        actor_loss_list.append(actor_loss.tolist())
        self.Actor_optimizer.zero_grad()
        actor_loss.backward()
        self.Actor_optimizer.step()

        self.soft_update(self.target_q, self.eval_q)
        self.soft_update(self.target_actor, self.eval_actor)
    def soft_update(self, net_target, net):
        for target_param, param in zip(net_target.parameters(), net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)


    def test_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0)
        action = self.eval_actor(state)
        return np.array([action.item()])


gym.register_envs(gymnasium_robotics)
env = gym.make('FetchReachDense-v3', max_episode_steps=100)

torch.manual_seed(0)
state_size = 16
action_size = 4
max_action = 1
hidden_size = 256
Actor_lr = 1e-3
DQN_lr = 1e-3
episodes = 500
reward_list = []

agent = DDPG(state_size, action_size, hidden_size, DQN_lr, Actor_lr, max_action)

for e in tqdm.tqdm(range(episodes)):
    state = env.reset()[0]
    state = np.concatenate([
        state['observation'],
        state['achieved_goal'],
        state['desired_goal']
    ])
    truncated = False
    total_reward = 0
    while not truncated:
        action = agent.take_action(state)  # 选择动作
        next_state, reward, done, truncated, info = env.step(action)
        next_state = np.concatenate([
            next_state['observation'],
            next_state['achieved_goal'],
            next_state['desired_goal']
        ])
        agent.Memory.add([state, action, reward, next_state, done])

        agent.update()
        state = next_state
        total_reward += reward

    success = info.get('is_success', False)
    reward_list.append(1 if success else 0)
    print(f"Episode {e}: Reward = {total_reward}")

df = pd.DataFrame(list(zip(reward_list, actor_loss_list, q_loss_list)), columns=['reward', 'actor_loss', 'q_loss'])
df.to_csv('DDPG_FetchReach.csv', index=False)
torch.save(agent.eval_actor.state_dict(), 'DDPG_FetchReach.pth')

plt.subplot(1, 3, 1)
plt.plot(reward_list)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Reward")
plt.subplot(1, 3, 2)
plt.plot(actor_loss_list)
plt.xlabel("Episode")
plt.ylabel("Actor_Loss")
plt.title("Actor_Loss")
plt.subplot(1, 3, 3)
plt.plot(q_loss_list)
plt.xlabel("Episode")
plt.ylabel("Q_Loss")
plt.title("Q_Loss")
plt.savefig('DDPG_FetchReach.png')
plt.show()
env.close()
plt.show()
env.close()
