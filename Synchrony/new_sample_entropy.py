import numpy as np
import nolds

def binary_to_continuous(matrix, window=10):
    """将二值信号转为连续发放率"""
    kernel = np.ones(window) / window  # 滑动平均
    continuous = np.apply_along_axis(
        lambda x: np.convolve(x, kernel, mode='same'), 
        axis=1, 
        arr=matrix.astype(float)
    )
    return continuous.mean(axis=0)  # 全局平均

# 生成测试数据
matrix = np.random.randint(0, 2, (50, 1000))
rate = binary_to_continuous(matrix)

# 计算样本熵
sampen = nolds.sampen(rate)

# 判断失稳（示例阈值）
if sampen < 0.5 or sampen > 2.0:
    print(f"样本熵异常: {sampen:.2f}（可能失稳）")