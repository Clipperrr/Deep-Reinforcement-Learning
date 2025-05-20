import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import gymnasium as gym
import random
import numpy as np
from collections import deque, namedtuple
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gymnasium_robotics
import pandas as pd
import math
from typing import Dict, List, Tuple

# 设置设备和随机种子
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 跟踪指标
q_loss_list = []
actor_loss_list = []
reward_list = []
success_rate_list = []

# 经验元组
Experience = namedtuple('Experience',
                        ['state', 'action', 'reward', 'next_state', 'done', 'info'])


class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.ln3 = nn.LayerNorm(hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, 1)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = self.ln1(F.relu(self.fc1(x)))
        x = self.ln2(F.relu(self.fc2(x)))
        x = self.ln3(F.relu(self.fc3(x)))
        return self.fc4(x)


class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.ln3 = nn.LayerNorm(hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, action_size)
        self.max_action = max_action

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.ln1(F.relu(self.fc1(x)))
        x = self.ln2(F.relu(self.fc2(x)))
        x = self.ln3(F.relu(self.fc3(x)))
        x = torch.tanh(self.fc4(x))
        return x * self.max_action


class OUNoise:
    """Ornstein-Uhlenbeck过程，用于时间相关的探索"""

    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


class HindsightReplayBuffer:
    """带有Hindsight Experience Replay的优先经验回放"""

    def __init__(self, capacity, state_dim, action_dim, goal_dim, alpha=0.6, beta=0.4, beta_increment=0.001,
                 k_future=4):
        self.capacity = capacity
        self.alpha = alpha  # 优先级指数
        self.beta = beta  # 重要性采样指数
        self.beta_increment = beta_increment
        self.k_future = k_future  # 未来时间步数，用于HER

        # 存储原始状态和目标
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim

        # 初始化缓冲区
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        self.episode_buffer = []

    def add_episode(self, episode):
        """添加整个情节到回放缓冲区，并应用HER"""
        # 存储原始经验
        for transition in episode:
            self._add(*transition)

        # 应用HER - 使用k_future个不同的未来目标
        for t in range(len(episode)):
            state, action, reward, next_state, done, info = episode[t]

            # 从未来的achieved_goals中选择k_future个作为新目标
            future_idxs = []
            for _ in range(self.k_future):
                # 随机选择t之后的时间步
                future_idx = np.random.randint(t, len(episode))
                future_idxs.append(future_idx)

            # 为每个选定的未来目标创建新的HER转换
            for future_idx in future_idxs:
                _, _, _, future_next_state, _, _ = episode[future_idx]

                # 使用未来状态的achieved_goal作为新目标
                future_ag = future_next_state['achieved_goal'].copy()

                # 创建新的目标状态
                her_state = {
                    'observation': state['observation'].copy(),
                    'achieved_goal': state['achieved_goal'].copy(),
                    'desired_goal': future_ag
                }

                her_next_state = {
                    'observation': next_state['observation'].copy(),
                    'achieved_goal': next_state['achieved_goal'].copy(),
                    'desired_goal': future_ag
                }

                # 计算新的奖励和完成状态
                her_reward = float(self._compute_reward(next_state['achieved_goal'], future_ag, info))
                her_done = False if her_reward == 0 else done  # 如果达到目标，则完成

                # 添加HER转换
                self._add(her_state, action, her_reward, her_next_state, her_done, info)

    def _add(self, state, action, reward, next_state, done, info):
        """添加单个转换到缓冲区"""
        # 新样本获得最高优先级
        max_priority = self.priorities.max() if self.size > 0 else 1.0

        if self.size < self.capacity:
            self.buffer.append((state, action, reward, next_state, done, info))
            self.size += 1
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done, info)

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def _compute_reward(self, achieved_goal, desired_goal, info):
        """计算稀疏奖励"""
        # 计算欧氏距离
        dist = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return -(dist > 0.05).astype(np.float32)  # 如果距离小于阈值，则奖励为0，否则为-1

    def sample(self, batch_size):
        if self.size < batch_size:
            indices = np.random.choice(self.size, batch_size, replace=True)
        else:
            # 计算采样概率
            priorities = self.priorities[:self.size]
            probabilities = priorities ** self.alpha
            probabilities /= probabilities.sum()

            # 采样
            indices = np.random.choice(self.size, batch_size, p=probabilities)

        # 计算重要性权重
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        # 提取批次数据
        samples = [self.buffer[idx] for idx in indices]

        # 初始化批次数组
        states = np.zeros((batch_size, self.state_dim))
        actions = np.zeros((batch_size, self.action_dim))
        rewards = np.zeros((batch_size, 1))
        next_states = np.zeros((batch_size, self.state_dim))
        dones = np.zeros((batch_size, 1))

        # 填充批次数组
        for i, (state, action, reward, next_state, done, _) in enumerate(samples):
            # 处理字典状态
            state_vector = self._process_state(state)
            next_state_vector = self._process_state(next_state)

            states[i] = state_vector
            actions[i] = action
            rewards[i] = reward
            next_states[i] = next_state_vector
            dones[i] = done

        # 转换为张量
        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.float32).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)
        weights = torch.tensor(weights, dtype=torch.float32).to(device).unsqueeze(1)

        # 增加beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        return states, actions, rewards, next_states, dones, weights, indices

    def _process_state(self, state):
        """处理字典状态，将其转换为向量"""
        # 连接观察、已达到目标和期望目标
        obs = state['observation']
        achieved_goal = state['achieved_goal']
        desired_goal = state['desired_goal']

        return np.concatenate([obs, achieved_goal, desired_goal])

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-5  # 添加小值避免零优先级


class TD3:
    def __init__(self, state_size, action_size, goal_size, hidden_size, critic_lr, actor_lr, max_action, device):
        self.device = device
        self.state_size = state_size
        self.action_size = action_size
        self.goal_size = goal_size
        self.full_state_dim = state_size + 2 * goal_size  # 观察 + 已达到目标 + 期望目标
        self.max_action = max_action

        # 初始化网络
        self.actor = Actor(self.full_state_dim, action_size, hidden_size, max_action).to(device)
        self.actor_target = Actor(self.full_state_dim, action_size, hidden_size, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic_1 = DQN(self.full_state_dim, action_size, hidden_size).to(device)
        self.critic_2 = DQN(self.full_state_dim, action_size, hidden_size).to(device)
        self.critic_target_1 = DQN(self.full_state_dim, action_size, hidden_size).to(device)
        self.critic_target_2 = DQN(self.full_state_dim, action_size, hidden_size).to(device)
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())

        # 初始化优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer_1 = optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_optimizer_2 = optim.Adam(self.critic_2.parameters(), lr=critic_lr)

        # 学习率调度器
        self.actor_scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=200, gamma=0.5)
        self.critic_scheduler_1 = optim.lr_scheduler.StepLR(self.critic_optimizer_1, step_size=200, gamma=0.5)
        self.critic_scheduler_2 = optim.lr_scheduler.StepLR(self.critic_optimizer_2, step_size=200, gamma=0.5)

        # 初始化经验回放
        self.replay_buffer = HindsightReplayBuffer(
            capacity=1000000,
            state_dim=self.full_state_dim,
            action_dim=action_size,
            goal_dim=goal_size,
            k_future=4
        )

        # 初始化噪声过程
        self.noise = OUNoise(action_size)

        # 超参数
        self.gamma = 0.98  # 折扣因子
        self.tau = 0.005  # 目标网络软更新系数
        self.policy_noise = 0.2  # 目标策略噪声
        self.noise_clip = 0.5  # 噪声裁剪
        self.policy_freq = 2  # 策略更新频率
        self.batch_size = 128  # 减小批次大小
        self.noise_scale = 0.4  # 增加初始噪声尺度
        self.noise_decay = 0.998  # 更慢的噪声衰减率
        self.min_noise = 0.05  # 最小噪声水平
        self.warm_up = 1000  # 预热步骤数

        self.total_it = 0
        self.n_updates = 0

    def preprocess_state(self, state):
        """将字典状态预处理为向量"""
        obs = state['observation']
        achieved_goal = state['achieved_goal']
        desired_goal = state['desired_goal']

        return np.concatenate([obs, achieved_goal, desired_goal])

    def select_action(self, state, training=True):
        with torch.no_grad():
            state_vector = self.preprocess_state(state)
            state_tensor = torch.FloatTensor(state_vector).to(self.device)
            action = self.actor(state_tensor.unsqueeze(0)).cpu().numpy().flatten()

        if training:
            # 添加时间相关噪声
            noise = self.noise_scale * self.noise.sample()
            action = np.clip(action + noise, -self.max_action, self.max_action)

        return action

    def train(self, n_updates=1):
        if self.replay_buffer.size < self.warm_up:
            return

        for _ in range(n_updates):
            self.total_it += 1

            # 从回放缓冲区中采样
            states, actions, rewards, next_states, dones, weights, indices = self.replay_buffer.sample(self.batch_size)

            with torch.no_grad():
                # 选择带噪声的动作
                noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                next_actions = (self.actor_target(next_states) + noise).clamp(-self.max_action, self.max_action)

                # 计算目标Q值
                target_Q1 = self.critic_target_1(next_states, next_actions)
                target_Q2 = self.critic_target_2(next_states, next_actions)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = rewards + (1 - dones) * self.gamma * target_Q

            # 获取当前Q估计
            current_Q1 = self.critic_1(states, actions)
            current_Q2 = self.critic_2(states, actions)

            # 计算TD误差
            td_error1 = torch.abs(current_Q1 - target_Q).detach().cpu().numpy()
            td_error2 = torch.abs(current_Q2 - target_Q).detach().cpu().numpy()
            td_errors = np.mean([td_error1, td_error2], axis=0)

            # 更新优先级
            self.replay_buffer.update_priorities(indices, td_errors.flatten())

            # 计算critic损失
            critic_loss_1 = F.mse_loss(current_Q1, target_Q)
            critic_loss_2 = F.mse_loss(current_Q2, target_Q)
            critic_loss = critic_loss_1 + critic_loss_2

            if len(q_loss_list) < 10000:  # 限制列表大小
                q_loss_list.append(critic_loss.item())

            # 更新critics
            self.critic_optimizer_1.zero_grad()
            self.critic_optimizer_2.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), 1.0)
            self.critic_optimizer_1.step()
            self.critic_optimizer_2.step()

            # 延迟策略更新
            if self.total_it % self.policy_freq == 0:
                # 计算actor损失
                actor_loss = -self.critic_1(states, self.actor(states)).mean()

                if len(actor_loss_list) < 10000:  # 限制列表大小
                    actor_loss_list.append(actor_loss.item())

                # 更新actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                self.actor_optimizer.step()

                # 更新目标网络
                self.soft_update(self.critic_target_1, self.critic_1)
                self.soft_update(self.critic_target_2, self.critic_2)
                self.soft_update(self.actor_target, self.actor)

            self.n_updates += 1

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def decay_noise(self):
        self.noise_scale = max(self.min_noise, self.noise_scale * self.noise_decay)

    def save(self, filename):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic_1': self.critic_1.state_dict(),
            'critic_2': self.critic_2.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target_1': self.critic_target_1.state_dict(),
            'critic_target_2': self.critic_target_2.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer_1': self.critic_optimizer_1.state_dict(),
            'critic_optimizer_2': self.critic_optimizer_2.state_dict()
        }, filename)

    def load(self, filename):
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic_1.load_state_dict(checkpoint['critic_1'])
        self.critic_2.load_state_dict(checkpoint['critic_2'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target_1.load_state_dict(checkpoint['critic_target_1'])
        self.critic_target_2.load_state_dict(checkpoint['critic_target_2'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer_1.load_state_dict(checkpoint['critic_optimizer_1'])
        self.critic_optimizer_2.load_state_dict(checkpoint['critic_optimizer_2'])


def train(env_name, episodes=2000, eval_interval=50):
    # 创建环境
    env = gym.make(env_name, max_episode_steps=100, render_mode=None)

    # 获取环境信息
    obs_space = env.observation_space.spaces['observation'].shape[0]
    goal_space = env.observation_space.spaces['desired_goal'].shape[0]
    action_space = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # 初始化代理
    hidden_size = 256
    actor_lr = 1e-4
    critic_lr = 3e-4

    agent = TD3(obs_space, action_space, goal_space, hidden_size, critic_lr, actor_lr, max_action, device)

    success_window = deque(maxlen=100)  # 用于跟踪成功率
    best_success_rate = 0

    for e in tqdm.tqdm(range(episodes)):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False

        # 存储当前episode的经验
        episode_exp = []

        # 在每集开始时重置噪声过程
        agent.noise.reset()

        while not (done or truncated):
            # 选择动作
            action = agent.select_action(state)

            # 执行步骤
            next_state, reward, done, truncated, info = env.step(action)

            # 添加到当前episode的经验
            episode_exp.append((state, action, reward, next_state, done, info))

            state = next_state
            episode_reward += reward

        # 将完整的episode添加到回放缓冲区（会应用HER）
        agent.replay_buffer.add_episode(episode_exp)

        # 多次更新网络
        n_updates = min(len(episode_exp), 40)  # 每个episode最多更新40次
        agent.train(n_updates=n_updates)

        # 衰减探索噪声
        agent.decay_noise()

        # 步进学习率调度器
        agent.actor_scheduler.step()
        agent.critic_scheduler_1.step()
        agent.critic_scheduler_2.step()

        # 记录指标
        success = info.get('is_success', 0)
        success_window.append(float(success))
        reward_list.append(episode_reward)
        success_rate = sum(success_window) / len(success_window)
        success_rate_list.append(success_rate)

        if (e + 1) % 10 == 0:
            print(
                f"Episode {e + 1}/{episodes} | 奖励: {episode_reward:.2f} | 成功: {success} | "
                f"成功率: {success_rate:.2f} | 噪声: {agent.noise_scale:.3f} | 更新次数: {agent.n_updates}")

        # 保存最佳模型
        if success_rate > best_success_rate and len(success_window) >= 20:
            best_success_rate = success_rate
            agent.save(f"best_td3_{env_name}.pth")
            print(f"保存最佳模型，成功率: {best_success_rate:.2f}")

        # 定期评估
        if (e + 1) % eval_interval == 0:
            eval_success_rate = evaluate(agent, env_name, episodes=20)
            print(f"评估成功率: {eval_success_rate:.2f}")

    # 保存最终模型和指标
    agent.save(f"final_td3_{env_name}.pth")

    # 保存指标到CSV
    df = pd.DataFrame({
        'reward': reward_list,
        'success_rate': success_rate_list
    })

    # 截取actor_loss和critic_loss的最新部分以匹配数据框长度
    loss_length = len(df)
    df['actor_loss'] = actor_loss_list[:loss_length] if len(actor_loss_list) >= loss_length else actor_loss_list + [
        None] * (loss_length - len(actor_loss_list))
    df['critic_loss'] = q_loss_list[:loss_length] if len(q_loss_list) >= loss_length else q_loss_list + [None] * (
                loss_length - len(q_loss_list))

    df.to_csv(f'improved_td3_{env_name}_metrics.csv', index=False)

    # 绘制结果
    plot_results(reward_list, actor_loss_list, q_loss_list, success_rate_list, env_name)

    return agent


def evaluate(agent, env_name, episodes=20):
    """在没有探索噪声的情况下评估代理"""
    env = gym.make(env_name, max_episode_steps=100, render_mode=None)

    successes = []

    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        truncated = False

        while not (done or truncated):
            action = agent.select_action(state, training=False)
            next_state, _, done, truncated, info = env.step(action)
            state = next_state

        successes.append(float(info.get('is_success', 0)))

    return sum(successes) / len(successes)


def plot_results(rewards, actor_losses, critic_losses, success_rates, env_name):
    """绘制训练结果"""
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # 平滑化数据
    def smooth(data, weight=0.9):
        smoothed = []
        last = data[0]
        for point in data:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    # 绘制奖励
    axs[0, 0].plot(rewards, alpha=0.3, color='blue')
    axs[0, 0].plot(smooth(rewards), color='blue')
    axs[0, 0].set_title('Episode Rewards')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Reward')

    # 绘制actor损失
    if actor_losses:
        axs[0, 1].plot(actor_losses[:5000], alpha=0.3, color='red')  # 限制绘制的点
        axs[0, 1].plot(smooth(actor_losses[:5000]), color='red')
        axs[0, 1].set_title('Actor Loss')
        axs[0, 1].set_xlabel('Update Step')
        axs[0, 1].set_ylabel('Loss')

    # 绘制critic损失
    if critic_losses:
        axs[1, 0].plot(critic_losses[:5000], alpha=0.3, color='green')  # 限制绘制的点
        axs[1, 0].plot(smooth(critic_losses[:5000]), color='green')
        axs[1, 0].set_title('Critic Loss')
        axs[1, 0].set_xlabel('Update Step')
        axs[1, 0].set_ylabel('Loss')

    # 绘制成功率
    axs[1, 1].plot(success_rates, color='purple')
    axs[1, 1].set_title('Success Rate')
    axs[1, 1].set_xlabel('Episode')
    axs[1, 1].set_ylabel('Rate')
    axs[1, 1].set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(f'improved_td3_{env_name}_results.png')
    plt.show()


def demo(env_name, model_path):
    """展示训练好的代理"""
    env = gym.make(env_name, render_mode="human")

    # 获取环境信息
    obs_space = env.observation_space.spaces['observation'].shape[0]
    goal_space = env.observation_space.spaces['desired_goal'].shape[0]
    action_space = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # 初始化代理
    hidden_size = 256
    agent = TD3(obs_space, action_space, goal_space, hidden_size, 3e-4, 1e-4, max_action, device)
    agent.load(model_path)

    for _ in range(10):  # 展示10个情节
        state, _ = env.reset()
        done = False
        truncated = False

        while not (done or truncated):
            action = agent.select_action(state, training=False)
            next_state, _, done, truncated, info = env.step(action)
            state = next_state

        print(f"Episode finished: Success = {info.get('is_success', 0)}")

    env.close()


if __name__ == "__main__":
    # 设置环境名称
    env_name = "FetchReach-v3"  # 你也可以使用其他环境，如 "FetchPush-v2", "FetchSlide-v2", "FetchPickAndPlace-v2"

    # 训练代理
    # print(f"开始训练代理在 {env_name} 环境中...")
    # agent = train(env_name, episodes=2000, eval_interval=50)

    # 展示训练好的代理
    print("展示训练好的代理...")
    demo(env_name, f"best_td3_{env_name}.pth")

    print("完成!")