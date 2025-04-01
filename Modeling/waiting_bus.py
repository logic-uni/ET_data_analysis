import numpy as np
import matplotlib.pyplot as plt

# 参数
arrival_rate = 1000  # 乘客总到达率 (人/秒)
min_passengers = 10  # 发车最小乘客数
max_passengers = 50  # 发车最大乘客数
bus_delay_min = 0.002    # 新车到达时间下限 (秒)
bus_delay_max = 0.004    # 新车到达时间上限 (秒)
passenger_wait = 0.02  # 乘客等待时间 (秒)

# 模拟发车时间间隔
num_simulations = 10000
intervals = []

for _ in range(num_simulations):
    # 累积乘客
    passengers = 0
    time_accum = 0
    while passengers < min_passengers:
        # 生成下一个乘客到达时间
        delta_t = np.random.exponential(1 / arrival_rate)
        time_accum += delta_t
        passengers += 1
        # 忽略乘客离开（因为 time_accum < passenger_wait）
    
    # 发车时间 = 累积时间 + 新车到达时间
    bus_delay = np.random.uniform(bus_delay_min, bus_delay_max)
    interval = time_accum + bus_delay
    intervals.append(interval)

# 绘制统计图
plt.hist(intervals, bins=50, density=True, alpha=0.7)
plt.xlabel('Departure Interval (s)')
plt.ylabel('Probability Density')
plt.title('Distribution of Bus Departure Intervals')
plt.show()