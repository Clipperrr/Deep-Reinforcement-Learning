import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import gymnasium as gym
import random
import numpy as np
from collections import deque
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

class TD3:
    def __init__(self, state_size, action_size, hidden_size, DQN_lr, Actor_lr, max_action):

        self.target_actor = Actor(state_size, action_size, hidden_size, max_action).to(device)
        self.target_q_1 = DQN(state_size, action_size, hidden_size).to(device)
        self.target_q_2 = DQN(state_size, action_size, hidden_size).to(device)
        self.eval_actor = Actor(state_size, action_size, hidden_size, max_action).to(device)
        self.eval_q_1 = DQN(state_size, action_size, hidden_size).to(device)
        self.eval_q_2 = DQN(state_size, action_size, hidden_size).to(device)
        self.target_actor.load_state_dict(self.eval_actor.state_dict())
        self.target_q_1.load_state_dict(self.eval_q_1.state_dict())
        self.target_q_2.load_state_dict(self.eval_q_2.state_dict())

        self.Actor_optimizer = optim.Adam(self.eval_actor.parameters(), lr=Actor_lr)
        self.Q_optimizer_1 = optim.Adam(self.eval_q_1.parameters(), lr=DQN_lr)
        self.Q_optimizer_2 = optim.Adam(self.eval_q_2.parameters(), lr=DQN_lr)

        self.state_size = state_size
        self.action_size = action_size
        self.Memory = ReplayBuffer(500000)
        self.gamma = 0.98  # 折扣因子
        self.tau = 0.005  # 目标网络软更新系数
        self.noise = 0.5
        self.batch_size = 256
        self.cnt = 0

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0)
        action = self.eval_actor(state)
        noise = np.clip(np.random.normal(0, self.noise, size=action.shape), -0.5, 0.5)
        action = np.clip(action.cpu().detach().numpy() + noise, -self.target_actor.max_action, self.target_actor.max_action)
        return action.flatten()


    def update(self):
        if self.Memory.size() < self.batch_size:
            return

        self.cnt += 1
        # 采样经验回放
        minibatch = self.Memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32).to(device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(device).unsqueeze(1)
        dones = torch.tensor(np.array(dones), dtype=torch.float32).to(device).unsqueeze(1)

        # 计算 Q 值估计
        q_1_estimate = self.eval_q_1(states, actions)
        q_2_estimate = self.eval_q_2(states, actions)
        # 计算目标 Q 值（使用目标 Actor 计算下个动作）
        next_actions = self.target_actor(next_states).detach()
        # 在目标 Actor 计算的动作上添加噪声（TD3 特有的目标动作平滑）
        noise = torch.clamp(torch.normal(0, 0.5, size=next_actions.shape).to(device), -0.5, 0.5)
        next_actions = torch.clamp(next_actions + noise, -self.target_actor.max_action, self.target_actor.max_action)

        min_target_q = torch.min(self.target_q_1(next_states, next_actions), self.target_q_2(next_states, next_actions))
        q_target = rewards + self.gamma * (1 - dones) * min_target_q.detach()

        # 计算 Critic（Q 网络）损失并优化
        q_1_loss = F.mse_loss(q_1_estimate, q_target)
        q_2_loss = F.mse_loss(q_2_estimate, q_target)
        q_loss_list.append(torch.mean(q_1_loss + q_2_loss).tolist())
        self.Q_optimizer_1.zero_grad()
        self.Q_optimizer_2.zero_grad()
        q_1_loss.backward()
        q_2_loss.backward()
        self.Q_optimizer_1.step()
        self.Q_optimizer_2.step()
        self.soft_update(self.target_q_1, self.eval_q_1)
        self.soft_update(self.target_q_2, self.eval_q_2)

        if self.cnt % 2 == 0:
            # 计算 Actor 网络的损失（最大化 Q 值）
            estimate_action = self.eval_actor(states)
            actor_loss = -torch.min(self.eval_q_1(states, estimate_action),
                                    self.eval_q_2(states, estimate_action)).mean()
            actor_loss_list.append(torch.mean(actor_loss).tolist())
            self.Actor_optimizer.zero_grad()
            actor_loss.backward()
            self.Actor_optimizer.step()
            self.soft_update(self.target_actor, self.eval_actor)

    def soft_update(self, net_target, net):
        for target_param, param in zip(net_target.parameters(), net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)


    def test_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0)
        action = self.eval_actor(state)
        return np.array([action.item()])


gym.register_envs(gymnasium_robotics)
env = gym.make('FetchReach-v3', max_episode_steps=100)

torch.manual_seed(0)
state_size = 16
action_size = 4
max_action = 1
hidden_size = 256
Actor_lr = 1e-3
DQN_lr = 2e-3
episodes = 500
reward_list = []

agent = TD3(state_size, action_size, hidden_size, DQN_lr, Actor_lr, max_action)

for e in tqdm.tqdm(range(episodes)):
    state = env.reset()[0]
    state = np.array([v for values in state.values() for v in values])
    truncated = False
    done = False
    cnt = 0
    total_reward = 0
    while not truncated | done:
        cnt += 1
        action = agent.take_action(state)  # 选择动作
        next_state, reward, done, truncated, info = env.step(action)
        next_state = np.array([v for values in next_state.values() for v in values])
        agent.Memory.add([state, action, reward, next_state, done])

        agent.update()
        state = next_state
        total_reward += reward

    success = info.get('is_success', False)
    reward_list.append(1 if success else 0)
    print(f"Episode {e}: Reward = {total_reward}")

df = pd.DataFrame(list(zip(reward_list, actor_loss_list, q_loss_list)), columns=['reward', 'actor_loss', 'q_loss'])
df.to_csv('TD3_FetchReach.csv', index=False)

torch.save(agent.eval_actor.state_dict(), 'TD3_FetchReach.pth')
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
plt.savefig('TD3_FetchReach.png')
plt.show()
env.close()
