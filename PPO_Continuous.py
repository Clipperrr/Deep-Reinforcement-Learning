import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import gymnasium as gym
import tqdm
import matplotlib.pyplot as plt
import numpy as np
from torch.distributions import Normal
from collections import deque
from torch.utils.data import TensorDataset, DataLoader
import gymnasium_robotics
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
actor_loss_list = []
q_loss_list = []
success = []

class Actor(nn.Module):
    def __init__(self, state_size, hidden_size, action_size, max_action):
        super(Actor, self).__init__()
        # 增加网络层数和宽度
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)  # 添加层归一化
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.ln3 = nn.LayerNorm(hidden_size // 2)
        self.fc4_mean = nn.Linear(hidden_size // 2, action_size)
        self.fc4_std = nn.Linear(hidden_size // 2, action_size)
        self.max_action = max_action
        self.apply(self._init_weights)

        # 初始化最后一层权重较小，确保初始策略更稳定
        nn.init.orthogonal_(self.fc4_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.fc4_std.weight, gain=0.01)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        mean = torch.tanh(self.fc4_mean(x)) * self.max_action
        # 限制标准差范围，避免过大或过小
        std = torch.clamp(F.softplus(self.fc4_std(x)), min=1e-3, max=1.0)
        return mean, std

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0)


class Critic(nn.Module):
    def __init__(self, state_size, hidden_size):
        super(Critic, self).__init__()
        # 增加网络层数和宽度
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)  # 添加层归一化
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.ln3 = nn.LayerNorm(hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, 1)
        self.apply(self._init_weights)

        # 初始化最后一层权重较小
        nn.init.orthogonal_(self.fc4.weight, gain=0.01)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        x = self.fc4(x)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0)

def Calculate_Advantage(td_delta, gamma, lamda):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for td in td_delta[::-1]:
        advantage = td + gamma * advantage * lamda
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(np.array(advantage_list), dtype=torch.float32)


class PPO:
    def __init__(self, state_size, hidden_size, action_size, actor_lr, critic_lr, gamma, lamda, max_action, device):
        self.epoch = 10
        self.gamma = gamma
        self.device = device
        self.lamda = lamda
        self.eps = 0.15
        self.batch_size = 2048
        self.actor = Actor(state_size, hidden_size, action_size, max_action).to(self.device)
        self.critic = Critic(state_size, hidden_size).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.replay_buffer = deque(maxlen=100)  # 存储多个episode的经验
    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
        mean, std = self.actor(state)

        normal = Normal(mean, std)
        action = normal.sample()
        action = torch.clamp(action, -self.actor.max_action, self.actor.max_action)  # 限制动作范围
        log_prob = normal.log_prob(action).sum(dim=-1, keepdim=True)  # 计算 log 概率

        return action.cpu().detach().numpy().flatten(), log_prob.cpu().detach().numpy().flatten()

    def store_episode(self, episode_dict):
        self.replay_buffer.append(episode_dict)

    def update(self):
        if len(self.replay_buffer) < 50:
            return  # 等待收集足够的episode
        transition_dict = {
            'state': [],
            'action': [],
            'next_state': [],
            'reward': [],
            'done': [],
            'log_prob': [],
            'advantage': [],
            'td_target': []
        }

        for episode in self.replay_buffer:
            state = torch.tensor(np.array(episode['state']), dtype=torch.float32).to(self.device)
            next_state = torch.tensor(np.array(episode['next_state']), dtype=torch.float32).to(self.device)
            reward = torch.tensor(np.array(episode['reward']), dtype=torch.float32).unsqueeze(1).to(self.device)
            done = torch.tensor(np.array(episode['done']), dtype=torch.float32).unsqueeze(1).to(self.device)

            td_target = reward + self.gamma * self.critic(next_state) * (1 - done)
            transition_dict['td_target'].extend(td_target.view(-1, 1).tolist())
            td_delta = td_target - self.critic(state)
            advantages = Calculate_Advantage(td_delta.cpu(), self.gamma, self.lamda).to(self.device)
            transition_dict['advantage'].extend(((advantages - advantages.mean()) / (advantages.std() + 1e-8)).view(-1, 1).tolist())
            for key in episode.keys():
                transition_dict[key].extend(episode[key])

        state = torch.tensor(np.array(transition_dict['state']), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(transition_dict['action']), dtype=torch.float32).to(self.device)
        old_log_prob = torch.tensor(np.array(transition_dict['log_prob']), dtype=torch.float32).to(self.device)
        advantages = torch.tensor(np.array(transition_dict['advantage']), dtype=torch.float32).to(self.device)
        td_target = torch.tensor(np.array(transition_dict['td_target']), dtype=torch.float32).to(self.device)

        for _ in range(self.epoch):
            dataset = TensorDataset(state, actions, old_log_prob, advantages, td_target.detach())
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            for batch_states, batch_actions, batch_old_log_probs, batch_advantages, batch_td_target in dataloader:
                mean, std = self.actor(batch_states)
                normal = Normal(mean, std)
                log_prob = normal.log_prob(batch_actions).sum(dim=-1, keepdim=True)
                ratio = torch.exp(log_prob - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(batch_td_target, self.critic(batch_states))

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                actor_loss_list.append(actor_loss.item())
                q_loss_list.append(critic_loss.item())
        self.replay_buffer.clear()


gym.register_envs(gymnasium_robotics)
env = gym.make('FetchReachDense-v3', max_episode_steps=150)
#env = gym.make('FetchReach-v3')

torch.manual_seed(0)
state_size = 9
action_size = 4
max_action = 1
hidden_size = 512
actor_lr = 1e-4
critic_lr = 3e-4
gamma = 0.98
episodes = 20000
lamda = 0.95
reward_list = []
agent = PPO(state_size, hidden_size, action_size, actor_lr, critic_lr, gamma, lamda, max_action, device)

for e in tqdm.tqdm(range(episodes)):
    episode_dict = {
        'state': [],
        'action': [],
        'next_state': [],
        'reward': [],
        'done': [],
        'log_prob': []
    }
    obs = env.reset()[0]
    goal_delta = obs['desired_goal'] - obs['achieved_goal']
    state = np.concatenate([obs['observation'][:3], obs['observation'][5:8], goal_delta])
    truncated = False
    total_reward = 0
    while not truncated:
        action, log_prob = agent.select_action(state)
        obs_next, reward, done, truncated, info = env.step(action)
        goal_delta = obs['desired_goal'] - obs['achieved_goal']
        next_state = np.concatenate([obs['observation'][:3], obs['observation'][5:8], goal_delta])
        reward = reward / (np.linalg.norm(goal_delta) + 1e-6)
        episode_dict['state'].append(state)
        episode_dict['reward'].append(reward)
        episode_dict['action'].append(action)
        episode_dict['done'].append(truncated)
        episode_dict['next_state'].append(next_state)
        episode_dict['log_prob'].append(log_prob)
        total_reward += reward
        state = next_state

    success.append(info['is_success'])
    reward_list.append(total_reward)
    agent.store_episode(episode_dict)
    agent.update()
    print(f"Episode {e}: Reward = {total_reward}, Success = {success[-1]}")
df = pd.DataFrame(list(zip(reward_list, success)), columns=['reward', 'success'])
df.to_csv('PPO_FetchReach.csv', index=False)
df = pd.DataFrame(list(zip(actor_loss_list, q_loss_list)), columns=['actor_loss', 'q_loss'])
df.to_csv('PPO_FetchReach_loss.csv', index=False)

torch.save(agent.actor.state_dict(), 'PPO_FetchReach.pth')
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
plt.savefig('PPO_FetchReach.png')
plt.show()
env.close()
plt.show()
env.close()
