import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon

# 系统参数
base_lambda = 10  # 基础发车率(次/秒)，对应平均间隔100ms
gate_delay = 0.05  # 安检关卡固定延迟(秒)
gate_prob = 0.7    # 需要安检的概率

# 模拟设置
num_simulations = 10000

def simulate_intervals():
    intervals = []
    for _ in range(num_simulations):
        # 基础发车间隔(指数分布)
        base_interval = expon.rvs(scale=1/base_lambda)
        
        # 随机决定是否需要安检
        if np.random.rand() < gate_prob:
            final_interval = base_interval + gate_delay
        else:
            final_interval = base_interval
            
        intervals.append(final_interval)
    return np.array(intervals)

# 运行模拟
intervals = simulate_intervals()

# 绘制混合分布统计图
plt.figure(figsize=(12, 6))
counts, bins, patches = plt.hist(intervals, 
                                bins=200, 
                                density=True, 
                                range=(0, 0.5),
                                alpha=0.7)

# 标记50ms峰值位置
plt.axvline(x=gate_delay, color='r', linestyle='--', 
           label=f'Gate delay peak at {gate_delay*1000:.0f}ms')
plt.xlabel('Departure Interval (seconds)')
plt.ylabel('Probability Density')
plt.title('Bus Departure Intervals: Exponential + Gate Delay Peak')
plt.legend()
plt.show()

# 输出统计信息
print(f"Base exponential mean: {1/base_lambda:.3f}s (100ms)")
print(f"Observed mean interval: {np.mean(intervals):.3f}s")
print(f"Peak location: {gate_delay*1000:.0f}ms")
print(f"Minimum interval: {np.min(intervals)*1000:.2f}ms")
print(f"Maximum interval: {np.max(intervals)*1000:.2f}ms")