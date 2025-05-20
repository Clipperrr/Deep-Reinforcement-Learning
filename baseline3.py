import gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt

# 创建环境
env = gym.make("Pendulum-v1")

# 创建 PPO 模型
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_pendulum/")

# 训练模型
timesteps = 100_000  # 训练 10 万步
model.learn(total_timesteps=timesteps)

# 保存模型
model.save("ppo_pendulum")

# 评估模型
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=False)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# 加载模型
model = PPO.load("ppo_pendulum")

# 运行测试
obs = env.reset()
rewards = []
for _ in range(200):
    action, _states = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    rewards.append(reward)
    env.render()

env.close()

# 绘制奖励曲线
plt.plot(rewards)
plt.xlabel("Steps")
plt.ylabel("Reward")
plt.title("PPO Pendulum-v1 Reward Curve")
plt.show()
