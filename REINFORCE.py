import torch.nn.functional as F
import torch.optim as optim
from torch import nn
import torch
import gym
import tqdm
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class critic(nn.Module):
    def __init__(self, state_size, hidden_size, action_size):
        super(critic, self).__init__()

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x


class REINFORCE:
    def __init__(self, state_size, hidden_size, action_size, lr, gamma, device):
        self.device = device
        self.policy_net = critic(state_size, hidden_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma

    def select_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.policy_net(state)
        prob_dist = torch.distributions.Categorical(probs)
        action = prob_dist.sample()
        return action.item()

    def update(self, transitions):
        action_list = transitions['action']
        reward_list = transitions['rewards']
        state_list = transitions['state']
        G = 0
        loss_list = []
        for i in range(len(state_list)- 1, -1, -1):

            state = torch.tensor(state_list[i], dtype=torch.float).to(self.device).unsqueeze(0)
            action = torch.tensor(action_list[i], dtype=torch.int64).view(1, -1).to(self.device)
            print(state)
            print(self.policy_net(state))
            G = self.gamma * G + reward_list[i]
            #Monte Carlo Methods Still, if TD, It's hard to predict V(s),Thus the Advantage Function could be  baseline = torch.mean(torch.tensor(rewards))  # 这里不能用 TD 估计的 V(s)
            log_prob = torch.log(self.policy_net(state).gather(1, action))

            loss = -log_prob * G
            loss_list.append(loss)

        self.optimizer.zero_grad()
        loss = torch.cat(loss_list).sum()
        loss.backward()
        self.optimizer.step()


# 环境初始化
env = gym.make('CartPole-v1')
torch.manual_seed(0)

state_size = env.observation_space.shape[0]
hidden_size = 128
action_size = env.action_space.n
lr = 5e-4
gamma = 0.99
agent = REINFORCE(state_size, hidden_size, action_size, lr, gamma, device)


# 训练
num_episodes = 1000
max_reward = 0
reward_list = []

for e in tqdm.tqdm(range(num_episodes)):
    state = env.reset()[0]
    total_reward = 0
    done = False
    transitions = {
        'state': [],
        'action': [],
        'next_state': [],
        'rewards': []
    }
    while not done:
        action = agent.select_action(state)  # 选择动作
        next_state, reward, done, truncated, info = env.step(action)
        transitions['state'].append(state)
        transitions['action'].append(action)
        transitions['next_state'].append(next_state)
        transitions['rewards'].append(reward)
        state = next_state
        total_reward += reward
    reward_list.append(total_reward)
    max_reward = max(total_reward, max_reward)

    # 学习
    agent.update(transitions)
    if e % 10 == 0:
        print(f"Episode {e}: Max Reward = {max_reward}")
print(f"Episode {e}: Max Reward = {max_reward}")
plt.plot(reward_list)
plt.show()
env.close()