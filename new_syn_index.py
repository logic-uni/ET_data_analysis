import numpy as np
import matplotlib.pyplot as plt

def calculate_sync(matrix, window_size=50):
    """计算滑动窗口内的同步指数"""
    n_neurons, n_time = matrix.shape
    sync_series = []
    
    for t in range(0, n_time - window_size + 1, window_size//2):  # 50%重叠
        window = matrix[:, t:t+window_size]
        corr = np.corrcoef(window)  # 相关系数矩阵
        np.fill_diagonal(corr, 0)   # 排除自相关
        sync = np.abs(corr).sum() / (n_neurons * (n_neurons - 1))  # 同步指数
        sync_series.append(sync)
    
    # 可视化
    plt.plot(sync_series, label='Sync Index')
    plt.axhline(y=0.3, color='r', linestyle='--', label='Threshold')
    plt.xlabel('Window Index')
    plt.ylabel('Synchrony')
    plt.legend()
    plt.show()
    
    return sync_series

# 生成测试数据（100个神经元，1000时间点）
matrix = np.random.randint(0, 2, (100, 1000))  # 随机0-1矩阵
sync = calculate_sync(matrix)

# 判断失稳
if np.max(sync) > 0.3:  # 根据实验数据调整阈值
    print("警告：系统失稳（同步性异常）")