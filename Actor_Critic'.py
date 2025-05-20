import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import gym
import tqdm
import matplotlib.pyplot as plt

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


class ActorCritic:
    def __init__(self, state_size, hidden_size, action_size, actor_lr, critic_lr , gamma, epoch, device):
        self.epoch = epoch
        self.gamma = gamma
        self.device = device
        self.actor = Actor(state_size, hidden_size, action_size).to(self.device)
        self.critic = Critic(state_size, hidden_size).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def select_action(self, state):
        state = torch.tensor([state], dtype=torch.float32).to(self.device).unsqueeze(0)
        probs = self.actor(state)
        prob_list = torch.distributions.Categorical(probs)  # 生成概率分布
        action = prob_list.sample()

        return action.item()


    def update(self, transition_dict):
        state = torch.tensor(transition_dict['state'], dtype=torch.float32).to(self.device)
        next_state = torch.tensor(transition_dict['next_state'], dtype=torch.float32).to(self.device)
        reward = torch.tensor(transition_dict['reward'], dtype=torch.float32).view(-1, 1).to(self.device)
        actions = torch.tensor(transition_dict['action'], dtype=torch.int64).view(-1, 1).to(self.device)
        done = torch.tensor(transition_dict['done'], dtype=torch.float32).view(-1, 1).to(self.device)

        log_prob = torch.log(self.actor(state).gather(1, actions))
        td_target = reward + self.gamma * self.critic(next_state).view(-1, 1) * (1 - done)

        td_delta = td_target - self.critic(state).view(-1, 1)
        actor_loss = torch.mean(-log_prob * td_delta.detach())
        critic_loss = F.mse_loss(td_target.detach(), self.critic(state).view(-1, 1))

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
reward_list = []
agent = ActorCritic(state_size, hidden_size, action_size, actor_lr, critic_lr, gamma, epoch, device)

for e in tqdm.tqdm(range(episodes // epoch)):
    episode_dict = {
        'state': [],
        'action': [],
        'next_state': [],
        'reward': [],
        'done': []
    }
    for _ in range(epoch):
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
            episode_dict['done'].append(done)
            episode_dict['next_state'].append(next_state)
            total_reward += reward
            state = next_state

        total_reward = min(total_reward, 200)
        reward_list.append(total_reward)
    agent.update(episode_dict)
    print(f"Episode {e * epoch}: Reward = {total_reward}")

plt.plot(reward_list)
plt.show()
env.close()

