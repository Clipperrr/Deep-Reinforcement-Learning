import numpy as np

width = 4
gamma = 0.9
actions = np.array([[0, -1], [0, 1], [-1, 0], [1, 0]])
actionReward = np.zeros(width)
V = np.zeros((width, width))

Rewards = -0.01  # Constant Reward for Movement
Prob = 0.25   # Same Prob on 4 directions
Pi = np.random.randint(0, 4, size=(width, width))
startR, startC = 0, 0
EndR, EndC = 0, 2
V[EndR][EndC] = 2   # Set the Terminal State

def GetV(row, col, action, gamma):
    if (row + actions[action][0]) in range(0, width) and (col + actions[action][1]) in range(0, width):
        row += actions[action][0]
        col += actions[action][1]
    return Rewards + gamma * V[row][col]

def policyInteration(gamma, iter = 100, theta=1e-7):
    steps = 0
    stable = False
    while not stable and steps < iter:
        stable = True
        steps += 1
        while True:
            delta = 0  # 最大变化量
            for row in range(width):
                for col in range(width):
                    if (row == startR and col == startC) or (row == EndR and col == EndC):
                        continue
                    lastV = V[row][col]
                    V[row][col] = GetV(row, col, Pi[row][col], gamma)  # 使用当前策略更新状态值
                    delta = max(delta, abs(lastV - V[row][col]))
            if delta < theta:  # 判断是否收敛
                break

        for row in range(width):
            for col in range(width):
                if (row == startR and col == startC) or (row == EndR and col == EndC):
                    continue
                lastPiS = Pi[row][col]
                v = np.zeros(width)
                for action in range(width):
                    v[action] = GetV(row, col, action, gamma)

                Pi[row][col] = np.argmax(v)
                if lastPiS != Pi[row][col]:
                    stable = False


policyInteration(gamma, iter = 100, theta=1e-6)
print(Pi)
print(V)