import numpy as np
import matplotlib.pyplot as plt

# 参数设置
T = 0.01  # 周期 (100 Hz)
mu = 0.5  # 泊松背景率
A = 0.8  # 周期性强度
f0 = 100  # 周期频率 (Hz)
phi = 0  # 相位

# 时间参数
T_max = 1  # 模拟时间长度
dt = 0.001
time = np.arange(0, T_max, dt)

# 周期性过程
periodic_spikes = (np.sin(2 * np.pi * f0 * time + phi) > 0.99).astype(int)

# 泊松过程
poisson_spikes = np.random.rand(len(time)) < mu * dt

# 组合发放
spikes = np.logical_or(periodic_spikes, poisson_spikes)

# 事件时间
event_times = time[spikes]
isi = np.diff(event_times)

# 绘制 ISI 分布
plt.hist(isi * 1000, bins=100, alpha=0.7)
plt.xlabel('Inter-spike Interval (ms)')
plt.ylabel('Counts')
plt.title('Periodic + Poisson Process ISI Distribution')
plt.show()
