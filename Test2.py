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
import random


# 设置随机种子以确保可重复性
def set_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


set_seeds()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class Actor(nn.Module):
    def __init__(self, state_size, hidden_size, action_size, max_action):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
        )

        self.mean_layer = nn.Linear(hidden_size // 2, action_size)
        self.log_std_layer = nn.Linear(hidden_size // 2, action_size)
        self.max_action = max_action

        # 初始化
        self.apply(self._init_weights)
        nn.init.orthogonal_(self.mean_layer.weight, gain=0.01)
        nn.init.orthogonal_(self.log_std_layer.weight, gain=0.01)

    def forward(self, x):
        features = self.net(x)
        mean = torch.tanh(self.mean_layer(features)) * self.max_action
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, -20, 2)  # 避免过大或过小的标准差
        std = log_std.exp()
        return mean, std

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0)

    def get_action(self, state, deterministic=False):
        mean, std = self(state)
        if deterministic:
            return mean
        else:
            dist = Normal(mean, std)
            action = dist.sample()
            action = torch.clamp(action, -self.max_action, self.max_action)
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            return action, log_prob


class Critic(nn.Module):
    def __init__(self, state_size, hidden_size):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

        self.apply(self._init_weights)

    def forward(self, x):
        return self.net(x)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear) and m.out_features == 1:
                nn.init.orthogonal_(m.weight, gain=0.01)


def compute_gae(rewards, values, next_values, dones, gamma, lam):
    """计算广义优势估计(GAE)"""
    gae = 0
    returns = []

    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * next_values[step] * (1 - dones[step]) - values[step]
        gae = delta + gamma * lam * (1 - dones[step]) * gae
        returns.insert(0, gae + values[step])

    return returns


class PPO:
    def __init__(self, state_size, hidden_size, action_size, actor_lr, critic_lr, gamma, lam, max_action, device):
        self.actor = Actor(state_size, hidden_size, action_size, max_action).to(device)
        self.critic = Critic(state_size, hidden_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = 0.2  # PPO clip参数
        self.epochs = 10  # 每批数据的训练轮数
        self.ent_coef = 0.05  # 熵正则化系数
        self.max_grad_norm = 0.5  # 梯度裁剪
        self.batch_size = 512  # 批量大小
        self.device = device

        # 收集轨迹的缓冲区
        self.buffer = []

    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            if deterministic:
                action = self.actor.get_action(state, deterministic=True)
                return action.cpu().numpy().flatten()
            else:
                action, log_prob = self.actor.get_action(state)
                return action.cpu().numpy().flatten(), log_prob.cpu().numpy().flatten()

    def store_transition(self, state, action, reward, next_state, done, log_prob):
        self.buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'log_prob': log_prob
        })

    def update(self):
        if len(self.buffer) == 0:
            return 0, 0  # 如果缓冲区为空，则返回零损失

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        old_log_probs = []

        # 收集所有数据
        for transition in self.buffer:
            states.append(transition['state'])
            actions.append(transition['action'])
            rewards.append(transition['reward'])
            next_states.append(transition['next_state'])
            dones.append(transition['done'])
            old_log_probs.append(transition['log_prob'])

        # 转换为张量
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(old_log_probs)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)

        # 计算价值和GAE
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)

        # 计算折扣回报和优势
        returns = torch.FloatTensor(compute_gae(
            rewards.cpu().numpy(),
            values.cpu().numpy(),
            next_values.cpu().numpy(),
            dones.cpu().numpy(),
            self.gamma,
            self.lam
        )).to(self.device)

        advantages = returns - values

        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 准备数据集
        dataset = TensorDataset(states, actions, old_log_probs, advantages, returns)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        actor_losses = []
        critic_losses = []

        # 多轮训练
        for _ in range(self.epochs):
            for batch_states, batch_actions, batch_old_log_probs, batch_advantages, batch_returns in dataloader:
                # Actor更新
                mean, std = self.actor(batch_states)
                dist = Normal(mean, std)
                log_probs = dist.log_prob(batch_actions).sum(dim=-1, keepdim=True)
                entropy = dist.entropy().mean()

                # 计算比率和裁剪
                ratio = torch.exp(log_probs - batch_old_log_probs)
                clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                actor_loss = -(torch.min(ratio * batch_advantages, clip_adv)).mean() - self.ent_coef * entropy

                # Critic更新
                critic_values = self.critic(batch_states)
                critic_loss = F.mse_loss(critic_values, batch_returns)

                # 梯度更新
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())

        # 清空缓冲区
        self.buffer = []

        return np.mean(actor_losses) if actor_losses else 0, np.mean(critic_losses) if critic_losses else 0


def evaluate_agent(env, agent, num_episodes=10):
    """评估智能体在环境中的表现"""
    success_rate = 0
    total_rewards = 0

    for _ in range(num_episodes):
        obs = env.reset()[0]
        state = get_state_from_obs(obs)
        done = False
        truncated = False
        episode_reward = 0

        while not (done or truncated):
            action = agent.select_action(state, deterministic=True)
            next_obs, reward, done, truncated, info = env.step(action)
            next_state = get_state_from_obs(next_obs)

            episode_reward += reward
            state = next_state

        success_rate += info['is_success']
        total_rewards += episode_reward

    return total_rewards / num_episodes, success_rate / num_episodes


def get_state_from_obs(obs):
    """从观测中提取状态 - 正确处理目标差异"""
    # 提取末端执行器位置和目标位置
    achieved_goal = obs['achieved_goal']  # 当前末端位置
    desired_goal = obs['desired_goal']  # 目标位置

    # 目标相对位置（方向向量）
    goal_direction = desired_goal - achieved_goal

    # 组合成完整状态
    state = np.concatenate([
        #obs['observation'],  # 抓手位置 + 速度 + 抓手开合状态等
        obs['desired_goal'] - obs['achieved_goal']  # 关键特征：距离向量
    ])

    return state


if __name__ == "__main__":
    # 创建环境
    env = gym.make('FetchReachDense-v3', max_episode_steps=150)  # 减少单次episode的步数

    # 环境参数
    obs = env.reset()[0]
    test_state = get_state_from_obs(obs)
    state_size = len(test_state)  # 动态获取状态维度
    action_size = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    print(f"State size: {state_size}, Action size: {action_size}, Max action: {max_action}")

    # PPO超参数
    hidden_size = 256
    actor_lr = 3e-4
    critic_lr = 3e-4
    gamma = 0.99
    lam = 0.95
    episodes = 10000  # 减少episode数量，专注于稳定学习
    eval_interval = 50

    # 创建PPO代理
    agent = PPO(state_size, hidden_size, action_size, actor_lr, critic_lr, gamma, lam, max_action, device)

    # 训练数据记录
    reward_history = []
    success_history = []
    actor_loss_history = []
    critic_loss_history = []
    eval_reward_history = []
    eval_success_history = []
    eval_episodes = []  # 记录评估发生的episode编号

    # 训练循环
    for episode in tqdm.tqdm(range(episodes)):
        obs = env.reset()[0]
        state = get_state_from_obs(obs)
        done = False
        truncated = False
        total_reward = 0

        # 单个episode的轨迹收集
        while not (done or truncated):
            action, log_prob = agent.select_action(state)
            next_obs, reward, done, truncated, info = env.step(action)
            next_state = get_state_from_obs(next_obs)

            agent.store_transition(state, action, reward, next_state, float(done or truncated), log_prob)

            state = next_state
            total_reward += reward

        # 记录数据
        reward_history.append(total_reward)
        success_history.append(float(info['is_success']))

        # 每轮更新
        actor_loss, critic_loss = agent.update()
        actor_loss_history.append(actor_loss)
        critic_loss_history.append(critic_loss)

        # 打印信息
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(reward_history[-10:])
            avg_success = np.mean(success_history[-10:]) * 100
            print(
                f"Episode {episode + 1}: Reward = {avg_reward:.2f}, Success Rate = {avg_success:.2f}%, Actor Loss = {actor_loss:.4f}, Critic Loss = {critic_loss:.4f}")

        # 定期评估
        if (episode + 1) % eval_interval == 0:
            eval_reward, eval_success = evaluate_agent(env, agent)
            eval_reward_history.append(eval_reward)
            eval_success_history.append(eval_success)
            eval_episodes.append(episode)
            print(
                f"Evaluation at episode {episode + 1}: Reward = {eval_reward:.2f}, Success Rate = {eval_success * 100:.2f}%")

    # 保存最终模型
    torch.save({
        'actor': agent.actor.state_dict(),
        'critic': agent.critic.state_dict(),
        'episode': episodes - 1,
        'reward': eval_reward_history[-1] if eval_reward_history else 0,
        'success_rate': eval_success_history[-1] if eval_success_history else 0
    }, 'ppo_fetchreach_final.pt')

    # 分别保存训练数据和评估数据
    training_data = {
        'episode': list(range(episodes)),
        'reward': reward_history,
        'success_rate': success_history,
        'actor_loss': actor_loss_history,
        'critic_loss': critic_loss_history,
    }

    evaluation_data = {
        'episode': eval_episodes,
        'eval_reward': eval_reward_history,
        'eval_success': eval_success_history,
    }

    pd.DataFrame(training_data).to_csv('ppo_fetchreach_training_data.csv', index=False)
    pd.DataFrame(evaluation_data).to_csv('ppo_fetchreach_evaluation_data.csv', index=False)

    # 绘制训练曲线
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(reward_history)
    plt.plot(eval_episodes, eval_reward_history, 'r-')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training & Evaluation Reward")
    plt.legend(["Training", "Evaluation"])

    plt.subplot(2, 2, 2)
    plt.plot([s * 100 for s in success_history])
    plt.plot(eval_episodes, [s * 100 for s in eval_success_history], 'r-')
    plt.xlabel("Episode")
    plt.ylabel("Success Rate (%)")
    plt.title("Training & Evaluation Success Rate")
    plt.legend(["Training", "Evaluation"])

    plt.subplot(2, 2, 3)
    plt.plot(actor_loss_history)
    plt.xlabel("Episode")
    plt.ylabel("Actor Loss")
    plt.title("Actor Loss")

    plt.subplot(2, 2, 4)
    plt.plot(critic_loss_history)
    plt.xlabel("Episode")
    plt.ylabel("Critic Loss")
    plt.title("Critic Loss")

    plt.tight_layout()
    plt.savefig('ppo_fetchreach_training_curves.png')
    plt.show()

    env.close()