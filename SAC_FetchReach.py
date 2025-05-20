import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import gymnasium as gym
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gymnasium_robotics
import pandas as pd
import tqdm

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
actor_loss_list = []
critic_loss_list = []
success = []

# SAC 需要的温度系数
LOG_ALPHA = torch.tensor(np.log(0.1), requires_grad=True, device=device)
ALPHA_OPTIMIZER = optim.Adam([LOG_ALPHA], lr=3e-4)
TARGET_ENTROPY = -4  # 目标熵值：action dim 的负值

def get_alpha():
    return LOG_ALPHA.exp()


# 经验回放池
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)


# SAC Actor（使用高斯策略）
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action

        # 初始化权重
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.mean.weight)
        nn.init.xavier_uniform_(self.log_std.weight)
        self.log_std.bias.data.fill_(0)  # 初始化为较小的标准差

    def forward(self, state):
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)  # 限制log_std的范围
        std = log_std.exp()
        return mean, std

    def sample(self, state):
        mean, std = self.forward(state)
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()  # 重参数化采样

        # 使用tanh压缩到[-1,1]范围
        action = torch.tanh(z) * self.max_action

        # 计算log概率，考虑tanh变换的雅可比行列式
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)


# Critic 网络
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        # Q1 网络
        self.fc1_q1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.ln1_q1 = nn.LayerNorm(hidden_dim)
        self.fc2_q1 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2_q1 = nn.LayerNorm(hidden_dim)
        self.fc3_q1 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3_q1 = nn.LayerNorm(hidden_dim)
        self.q1_out = nn.Linear(hidden_dim, 1)

        # Q2 网络
        self.fc1_q2 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.ln1_q2 = nn.LayerNorm(hidden_dim)
        self.fc2_q2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2_q2 = nn.LayerNorm(hidden_dim)
        self.fc3_q2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3_q2 = nn.LayerNorm(hidden_dim)
        self.q2_out = nn.Linear(hidden_dim, 1)

        # 初始化权重
        for layer in [self.fc1_q1, self.fc2_q1, self.fc3_q1, self.q1_out,
                      self.fc1_q2, self.fc2_q2, self.fc3_q2, self.q2_out]:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)

        # Q1
        x1 = F.relu(self.ln1_q1(self.fc1_q1(sa)))
        x1 = F.relu(self.ln2_q1(self.fc2_q1(x1)))
        x1 = F.relu(self.ln3_q1(self.fc3_q1(x1)))
        q1 = self.q1_out(x1)

        # Q2
        x2 = F.relu(self.ln1_q2(self.fc1_q2(sa)))
        x2 = F.relu(self.ln2_q2(self.fc2_q2(x2)))
        x2 = F.relu(self.ln3_q2(self.fc3_q2(x2)))
        q2 = self.q2_out(x2)

        return q1, q2

# SAC 训练器
class SAC:
    def __init__(self, state_dim, action_dim, hidden_dim, actor_lr, critic_lr, max_action):
        self.actor = Actor(state_dim, action_dim, hidden_dim, max_action).to(device)
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.target_critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.memory = ReplayBuffer(100000)
        self.gamma = 0.98
        self.tau = 0.005
        self.batch_size = 256

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0)
        action, _ = self.actor.sample(state)
        return action.cpu().detach().numpy().flatten()

    def update(self):
        if self.memory.size() < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32).to(device)
        rewards = np.clip(np.array(rewards), -1, 1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1).to(device)

        # 计算目标 Q 值
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_target, q2_target = self.target_critic(next_states, next_actions)
            q_target = rewards + self.gamma * (1 - dones) * (
                        torch.min(q1_target, q2_target) - get_alpha() * next_log_probs)

        # 计算当前 Q 值
        q1, q2 = self.critic(states, actions)
        critic_loss = torch.mean(F.smooth_l1_loss(q1, q_target) + F.smooth_l1_loss(q2, q_target))

        self.critic_optimizer.zero_grad()
        critic_loss_list.append(critic_loss.item())
        critic_loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # 计算 Actor 损失
        new_actions, log_probs = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        actor_loss = (get_alpha() * log_probs - torch.min(q1_new, q2_new)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss_list.append(actor_loss.item())
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # 更新 alpha（熵项）
        alpha_loss = -(get_alpha() * (log_probs.detach() + TARGET_ENTROPY)).mean()
        ALPHA_OPTIMIZER.zero_grad()
        alpha_loss.backward()
        ALPHA_OPTIMIZER.step()

        # 软更新目标网络
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


gym.register_envs(gymnasium_robotics)
env = gym.make('FetchReachDense-v3', max_episode_steps=100)

torch.manual_seed(0)
state_size = 6
action_size = 4
max_action = 1
hidden_size = 256
Actor_lr = 1e-3
DQN_lr = 1e-3
episodes = 1000
reward_list = []

agent = SAC(state_size, action_size, hidden_size, DQN_lr, Actor_lr, max_action)

for e in tqdm.tqdm(range(episodes)):
    state = env.reset()[0]
    state = np.concatenate([
        state['observation'][:3],
        state['achieved_goal'] - state['desired_goal']
    ])

    truncated = False
    total_reward = 0
    while not truncated:
        action = agent.take_action(state)  # 选择动作
        next_state, reward, done, truncated, info = env.step(action)
        next_state = np.concatenate([
            next_state['observation'][:3],
            next_state['achieved_goal'] - next_state['desired_goal']
        ])
        agent.memory.add([state, action, reward, next_state, done])
        agent.update()
        state = next_state
        total_reward += reward

    success.append(info.get('is_success', 0))
    reward_list.append(total_reward)
    print(f"Episode {e}: Reward = {total_reward}, Success = {success[-1]}")


def smooth(data, weight=0.9):
    smoothed = []
    last = data[0]
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

reward_list = smooth(reward_list, 0.8)
df = pd.DataFrame(list(zip(reward_list, actor_loss_list, critic_loss_list, success)), columns=['reward', 'actor_loss', 'q_loss', 'success'])
df.to_csv('SAC_FetchReach.csv', index=False)

torch.save(agent.actor.state_dict(), 'SAC_FetchReach.pth')
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
plt.plot(critic_loss_list)
plt.xlabel("Episode")
plt.ylabel("Q_Loss")
plt.title("Q_Loss")
plt.savefig('SAC_FetchReach.png')
plt.show()
env.close()
