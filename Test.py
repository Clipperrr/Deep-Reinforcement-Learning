import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import gymnasium as gym
import gymnasium_robotics
from collections import deque
import tqdm
import matplotlib.pyplot as plt
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

actor_loss_list = []
q_loss_list = []
# ======================== Replay Buffer with HER ========================
class HERReplayBuffer:
    def __init__(self, buffer_size, k=4):
        self.buffer = deque(maxlen=buffer_size)
        self.k = k  # HER samples per transition

    def add_episode(self, episode):
        self.buffer.append(episode)

    def sample(self, batch_size, env):
        transitions = []
        for _ in range(batch_size):
            episode = random.choice(self.buffer)
            t = random.randint(0, len(episode) - 1)
            s, a, r, s_, d, ag, dg = episode[t]

            # 添加原始 transition
            transitions.append((s, a, r, s_, d))

            # 添加 HER transition
            for _ in range(self.k):
                if t < len(episode) - 1:
                    future = random.randint(t + 1, len(episode) - 1)
                else:
                    future = t
                new_goal = episode[future][5]  # achieved_goal at future time
                s_her = np.concatenate([episode[t][0][:3], episode[t][5] - new_goal])
                s_next_her = np.concatenate([episode[t][3][:3], episode[t][5] - new_goal])
                reward_her = env.unwrapped.compute_reward(episode[t][5], new_goal, None)
                done_her = float(reward_her >= -0.05)   #不应该是0
                transitions.append((s_her, a, reward_her, s_next_her, done_her))

        states, actions, rewards, next_states, dones = zip(*transitions)
        return (
            torch.tensor(np.array(states), dtype=torch.float32).to(device),
            torch.tensor(np.array(actions), dtype=torch.float32).to(device),
            torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1).to(device),
            torch.tensor(np.array(next_states), dtype=torch.float32).to(device),
            torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1).to(device),
        )

# ======================== Actor ========================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, max_action):
        super().__init__()
        self.max_action = max_action

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = self.net(x)
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        std = torch.exp(log_std)
        return mean, std

    def sample(self, state):
        mean, std = self(state)
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z) * self.max_action
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(1, keepdim=True)

# ======================== Critic ========================
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        input_dim = state_dim + action_dim

        def build_critic():
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),

                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),

                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),

                nn.Linear(hidden_dim, 1),
            )

        self.q1 = build_critic()
        self.q2 = build_critic()

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.q1(x), self.q2(x)

# ======================== SAC Agent ========================
class SACAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, 256, max_action).to(device)
        self.critic = Critic(state_dim, action_dim, 256).to(device)
        self.critic_target = Critic(state_dim, action_dim, 256).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=5e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.log_alpha = torch.tensor(np.log(0.1), requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)
        self.target_entropy = -action_dim

        self.gamma = 0.95
        self.tau = 0.01
        self.replay_buffer = HERReplayBuffer(100000)

    def get_alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        action, _ = self.actor.sample(state)
        return action.cpu().detach().numpy().flatten()

    def update(self, env, batch_size=512):

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size, env)

        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_target, q2_target = self.critic_target(next_states, next_actions)
            q_target = rewards + self.gamma * (1 - dones) * (
                torch.min(q1_target, q2_target) - self.get_alpha() * next_log_probs
            )

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_optimizer.zero_grad()
        q_loss_list.append(critic_loss.item())
        critic_loss.backward()
        self.critic_optimizer.step()

        new_actions, log_probs = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        actor_loss = (self.get_alpha() * log_probs - torch.min(q1_new, q2_new)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss_list.append(actor_loss.item())
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.get_alpha() * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # soft update
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# ======================== Training ========================
env = gym.make("FetchReachDense-v3",max_episode_steps=100)
agent = SACAgent(state_dim=6, action_dim=4, max_action=1.0)

rewards, success_rate = [], []
episodes = 3000

for episode in tqdm.tqdm(range(episodes)):
    obs = env.reset()[0]
    goal_delta = obs['desired_goal'] - obs['achieved_goal']
    state = np.concatenate([obs['observation'][:3], goal_delta])
    episode_data = []
    total_reward = 0
    done, truncated = False, False

    while not done and not truncated:
        if episode < 100:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)
        obs_next, reward, done, truncated, info = env.step(action)
        goal_delta = obs['desired_goal'] - obs['achieved_goal']
        next_state = np.concatenate([obs['observation'][:3], goal_delta])
        episode_data.append((state, action, reward, next_state, done, obs['achieved_goal'], obs['desired_goal']))
        state = next_state
        obs = obs_next
        total_reward += reward

    agent.replay_buffer.add_episode(episode_data)
    if len(agent.replay_buffer.buffer) > 50:
        for _ in range(50):
            agent.update(env)

    rewards.append(total_reward)
    success_rate.append(info['is_success'])
    print(f"Episode {episode}, Reward: {total_reward:.2f}, Success: {info['is_success']}")

# ======================== Save & Plot ========================
pd.DataFrame({'reward': rewards, 'success': success_rate}).to_csv("SAC_HER_FetchReach1.csv", index=False)
pd.DataFrame({'actor_loss': actor_loss_list, 'critic_loss': q_loss_list}).to_csv("SAC_HER_FetchReach_loss1.csv", index=False)
torch.save(agent.actor.state_dict(), "sac_her_actor.pth")

plt.plot(rewards, label="Reward")
plt.plot(success_rate, label="Success")
plt.legend()
plt.title("SAC + HER on FetchReachDense-v3")
plt.xlabel("Episode")
plt.show()

env.close()
