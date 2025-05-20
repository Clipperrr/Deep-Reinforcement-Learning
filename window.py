import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from collections import deque
# 读取 CSV 文件
def load_csv(file_path):
    df = pd.read_csv(file_path)
    return df

# 平滑数据（使用 Savitzky-Golay 滤波器）
def smooth_data(data, window_size=11, poly_order=1):
    return savgol_filter(data, window_size, poly_order)

# 绘制原始数据和平滑后的数据
def plot_data(df):
    plt.figure(figsize=(12, 6))
    for column in df.columns:
        original = df[column]
        smoothed = smooth_data(original)
        plt.plot(original, alpha=0.4, label=f"Original {column}")
        plt.plot(smoothed, label=f"Smoothed {column}")
        plt.legend()
        plt.xlabel("Episode")
        plt.ylabel("Value")
        plt.title("Reward")
        plt.show()

# 示例：加载 CSV 并绘图
file_path = "PPO_FetchReach1.csv"  # 替换为你的 CSV 文件路径
df = load_csv(file_path)
original = df['success']
window = original[:100]
window = deque(window, maxlen=100)
result = []
for i in range(9, len(original)):
    result.append(sum(window) / 80)
    window.append(original[i])
    window.popleft()
plt.plot(result)
plt.xlabel("Episode")
plt.ylabel("Value")
plt.title("success_rate")
plt.show()

