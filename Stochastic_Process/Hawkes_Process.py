import numpy as np
import matplotlib.pyplot as plt

# 参数设置
mu = 0.1  # 基础发放率
alpha = 0.8  # 激发强度
beta = 1.5  # 衰减速率
T = 100  # 模拟时长

# 模拟事件发生
t_events = []
t = 0
while t < T:
    lambda_t = mu
    for t_i in t_events:
        lambda_t += alpha * np.exp(-beta * (t - t_i))
    
    # 生成下一个事件时间
    dt = -np.log(np.random.rand()) / lambda_t
    t += dt
    if t < T:
        t_events.append(t)

# 画出事件时间
plt.eventplot(t_events, lineoffsets=1, colors='black')
plt.xlabel('Time (s)')
plt.title('Hawkes Process Simulation')
plt.show()