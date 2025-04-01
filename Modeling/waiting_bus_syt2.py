import numpy as np
import matplotlib.pyplot as plt

# 系统参数
arrival_rate = 1000    # 实际乘客到达率(人/秒)
min_passengers = 10    # 发车最小乘客数
max_passengers = 50    # 发车最大乘客数
bus_delay_min = 0.002  # 新车到达时间下限(秒)
bus_delay_max = 0.004  # 新车到达时间上限(秒)
passenger_wait = 0.02  # 乘客等待时间(秒)

# 关卡参数
gate_min = 10          # 关卡最小累积人数
gate_max = 20          # 关卡最大累积人数
leave_min = 1          # 最少离开人数
leave_max = 3         # 最多离开人数

# 模拟设置
num_simulations = 100000
intervals = []

for _ in range(num_simulations):
    # 初始化状态
    waiting_passengers = 0  # 等待筛选的乘客池
    passed_passengers = 0   # 已通过关卡的乘客
    time_accum = 0          # 累积时间
    
    # 直到累积足够的通过乘客
    while passed_passengers < min_passengers:
        # 生成下一批实际到达乘客
        delta_t = np.random.exponential(1/arrival_rate)
        time_accum += delta_t
        new_passengers = 1  # 泊松过程每次到达1人
        waiting_passengers += new_passengers
        
        # 检查是否触发关卡筛选
        if waiting_passengers >= gate_min:
            # 随机决定本次筛选的累积人数(10-20人)
            gate_threshold = np.random.randint(gate_min, gate_max+1)
            if waiting_passengers >= gate_threshold:
                # 放行1人
                passed_passengers += 1
                waiting_passengers -= gate_threshold
                
                # 随机决定离开人数(5-10人)
                leave_count = np.random.randint(leave_min, leave_max+1)
                leave_count = min(leave_count, waiting_passengers)
                waiting_passengers -= leave_count
    
    # 随机选择本次发车的目标乘客数(10-50人)
    n = np.random.randint(min_passengers, max_passengers+1)
    n = min(n, passed_passengers)  # 不能超过实际通过人数
    
    # 计算实际需要的累积时间
    # 通过率约为 arrival_rate/(gate_threshold均值*(1+leave_count均值))
    # gate_threshold均值=15, leave_count均值=7.5
    # 所以有效通过率 ≈ 1000/(15 * 8.5) ≈ 7.84人/秒
    # 但更准确的做法是保留之前的time_accum
    
    # 新车到达延迟
    bus_delay = np.random.uniform(bus_delay_min, bus_delay_max)
    
    # 总发车间隔
    interval = time_accum + bus_delay
    intervals.append(interval)

# 绘制统计图
plt.figure(figsize=(10, 6))
plt.hist(intervals, bins=100, density=True, alpha=0.7, edgecolor='black')
plt.xlabel('Departure Interval (seconds)')
plt.ylabel('Probability Density')
plt.title('Bus Departure Interval Distribution With Complex Gate')
plt.show()

# 输出统计信息
print(f"Minimum interval: {min(intervals):.6f} s")
print(f"Maximum interval: {max(intervals):.6f} s")
print(f"Average interval: {np.mean(intervals):.6f} s")
print(f"Median interval: {np.median(intervals):.6f} s")