import numpy as np
from scipy.stats import zscore

def detect_phase_transition(matrix):
    """检测总活动的突变点"""
    total_activity = matrix.sum(axis=0)  # 每个时间点的总激活数
    z = zscore(total_activity)
    
    # CUSUM参数
    k = 0.5    # 容差
    h = 5.0    # 报警阈值
    S_pos = S_neg = 0
    alarms = []
    
    for t in range(len(z)):
        S_pos = max(0, S_pos + z[t] - k)
        S_neg = min(0, S_neg + z[t] + k)
        
        if S_pos > h or S_neg < -h:
            alarms.append(t)
            S_pos = S_neg = 0  # 重置
    
    return alarms

# 生成测试数据（注入一个突变）
matrix = np.random.randint(0, 2, (50, 500))
matrix[:, 300:] = np.random.choice([0,1], (50,200), p=[0.95, 0.05])  # 300时刻后活动骤降

alarms = detect_phase_transition(matrix)
print(f"检测到突变时刻: {alarms}")
