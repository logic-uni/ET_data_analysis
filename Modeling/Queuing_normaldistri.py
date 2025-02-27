"""
# coding: utf-8
@author: Yuhao Zhang
last updated: 09/04/2024
data from: Xinchao Chen
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# 设置参数
arrival_rate = 12  # 每秒钟到达4个spike
service_time = 0.05  # 每个顾客的服务时间为50ms   正常突触囊泡传递仅需0.01s  Syt2敲除后的突触囊泡传递需要
simulation_time = 30  # 模拟1分钟

num_customers = int(arrival_rate * simulation_time)  # 总顾客数

# 生成到达时间（均匀分布）
np.random.seed(42)
arrival_times = np.sort(np.random.uniform(0, simulation_time, num_customers))

# 模拟服务台
service_completion_times = []
next_free_time = 0

for arrival_time in arrival_times:
    if arrival_time >= next_free_time:
        next_free_time = arrival_time + service_time
        service_completion_times.append(next_free_time)

# 计算服务完成时间间隔
intervals = np.diff(service_completion_times)

# 仅统计0-0.25s之间的间隔
filtered_intervals = intervals[(intervals > 0) & (intervals <= 0.25)]

# 画出服务完成时刻的竖线
plt.figure(figsize=(10, 6))
plt.vlines(service_completion_times, 0, 1, colors='b', label='Service Completion')
plt.yticks([])
plt.xlabel('Time (s)')
plt.title('Service Completion Times')
plt.legend()
plt.show()

# 画出服务完成时间间隔的直方图
plt.figure(figsize=(10, 6))
plt.hist(filtered_intervals, bins=100, density=True, alpha=0.75, color='b', edgecolor='black')
plt.xlabel('Interval between service completions (s)')
plt.ylabel('Density')
plt.title('Histogram of Service Completion Intervals (0-0.25s)')

# 拟合指数分布并画出拟合曲线
loc, scale = stats.expon.fit(filtered_intervals)
x = np.linspace(0, 0.25, 100)
pdf = stats.expon.pdf(x, loc, scale)
plt.plot(x, pdf, 'r--', label=f'Exponential fit\nλ={1/scale:.2f}')
plt.legend()
plt.show()
