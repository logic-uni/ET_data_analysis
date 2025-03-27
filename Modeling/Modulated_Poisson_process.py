import numpy as np
import matplotlib.pyplot as plt

def generate_arrival_times(rate_per_second, max_passengers):
    """生成泊松过程的乘客到达时间（毫秒）"""
    rate_per_ms = rate_per_second / 1000
    intervals = np.random.exponential(1/rate_per_ms, max_passengers)
    return np.cumsum(intervals)

def adjust_arrival_time(t):
    """调整到达时间模拟新挡板效应（关闭10ms，开启3ms）"""
    cycle = 13  # 总周期13ms
    close_duration = 10  # 关闭持续时间
    cycle_start = (t // cycle) * cycle
    offset = t - cycle_start
    # 关闭期间到达的乘客对齐到周期开始+10ms（开启时刻）
    return cycle_start + close_duration if offset < close_duration else t

def process_departures(arrival_times):
    """处理发车逻辑并计算时间间隔"""
    unique_times, counts = np.unique(arrival_times, return_counts=True)
    departures = []
    current_passengers = 0
    for t, cnt in zip(unique_times, counts):
        current_passengers += cnt
        while current_passengers >= 30:
            departures.append(t)
            current_passengers -= 30
    return np.diff(departures) if len(departures) > 1 else []

# 参数配置
rate = 1500  # 乘客到达率（人/秒）
max_passengers = 50000  # 模拟乘客总数

# 生成数据
np.random.seed(42)
base_arrival = generate_arrival_times(rate, max_passengers)

# 无挡板情况
departure_intervals_raw = process_departures(base_arrival)

# 有挡板情况
adjusted_arrival = np.array([adjust_arrival_time(t) for t in base_arrival])
departure_intervals_adj = process_departures(adjusted_arrival)

# 可视化
plt.figure(figsize=(14, 6))

# 原始分布
plt.subplot(1, 2, 1)
plt.hist(departure_intervals_raw, bins=np.arange(0, 200, 5), 
         alpha=0.7, color='steelblue', edgecolor='black')
plt.title('Original Poisson Process')
plt.xlabel('Departure Interval (ms)')
plt.ylabel('Frequency')
plt.grid(alpha=0.3)

# 调整后分布
plt.subplot(1, 2, 2)
plt.hist(departure_intervals_adj, bins=np.arange(0, 250, 0.5),
         alpha=0.7, color='salmon', edgecolor='black')
plt.title('With 10ms-close/3ms-open Barrier')
plt.xlabel('Departure Interval (ms)')
plt.ylabel('Frequency')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()