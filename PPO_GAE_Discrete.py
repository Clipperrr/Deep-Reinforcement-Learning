import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import gym
import tqdm
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_size, hidden_size, action_size):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1)
        return x



class Critic(nn.Module):
    def __init__(self, state_size, hidden_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


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
    def __init__(self, state_size, hidden_size, action_size, actor_lr, critic_lr, gamma, epoch, lamda, device):
        self.epoch = epoch
        self.gamma = gamma
        self.device = device
        self.lamda = lamda
        self.eps = 0.2
        self.actor = Actor(state_size, hidden_size, action_size).to(self.device)
        self.critic = Critic(state_size, hidden_size).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def select_action(self, state):
        state = torch.tensor(np.array(state), dtype=torch.float32).to(self.device).unsqueeze(0)
        probs = self.actor(state)
        prob_list = torch.distributions.Categorical(probs)  # 生成概率分布
        action = prob_list.sample()

        return action.item()

    def update(self, transition_dict):
        state = torch.tensor(np.array(transition_dict['state']), dtype=torch.float32).to(self.device)
        next_state = torch.tensor(np.array(transition_dict['next_state']), dtype=torch.float32).to(self.device)
        reward = torch.tensor(np.array(transition_dict['reward']), dtype=torch.float32).unsqueeze(1).to(self.device)
        actions = torch.tensor(np.array(transition_dict['action']), dtype=torch.int64).unsqueeze(1).to(self.device)
        done = torch.tensor(np.array(transition_dict['done']), dtype=torch.float32).unsqueeze(1).to(self.device)

        old_log_prob = torch.log(self.actor(state).gather(1, actions)).detach()
        td_target = reward + self.gamma * self.critic(next_state) * (1 - done)

        td_delta = td_target - self.critic(state)
        advantage = Calculate_Advantage(td_delta.cpu(), self.gamma, self.lamda).to(self.device)

        for _ in range(self.epoch):
            log_prob = torch.log(self.actor(state).gather(1, actions))
            ratio = torch.exp(log_prob - old_log_prob)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(td_target.detach(), self.critic(state))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

env = gym.make('CartPole-v1')
torch.manual_seed(0)

state_size = env.observation_space.shape[0]
hidden_size = 128
action_size = env.action_space.n
actor_lr = 1e-3
critic_lr = 1e-2
gamma = 0.98
episodes = 5000
epoch = 10
lamda = 0.95
reward_list = []
agent = PPO(state_size, hidden_size, action_size, actor_lr, critic_lr, gamma, epoch, lamda, device)

for e in tqdm.tqdm(range(episodes)):
    episode_dict = {
        'state': [],
        'action': [],
        'next_state': [],
        'reward': [],
        'done': []
    }
    state = env.reset()[0]
    done = False
    total_reward = 0
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, truncated, info = env.step(action)
        if done:
            reward = -20
        episode_dict['state'].append(state)
        episode_dict['reward'].append(reward)
        episode_dict['action'].append(action)
        episode_dict['done'].append(done | truncated)
        episode_dict['next_state'].append(next_state)
        total_reward += reward
        state = next_state

    reward_list.append(total_reward)
    agent.update(episode_dict)
    print(f"Episode {e}: Reward = {total_reward}")

plt.plot(reward_list)
plt.show()
env.close()

