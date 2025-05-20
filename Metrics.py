import pandas as pd
import matplotlib.pyplot as plt
window_size = 20  # 取 20 个 episode 的平均值
rolling_success_rate = pd.Series(success_list).rolling(window=window_size).mean()
file_path = "TD3_FetchReach.csv"  # 替换为你的 CSV 文件路径
df = pd.read_csv(file_path)
df = df['r']
plt.plot(rolling_success_rate, label=f"Rolling Success Rate (Window={window_size})")
plt.xlabel("Episode")
plt.ylabel("Success Rate")
plt.title("Rolling Success Rate Over Episodes")
plt.legend()
plt.show()