import numpy as np
import nolds

def calculate_lyap(matrix, neuron_idx=0, window=10):
    """计算指定神经元的李雅普诺夫指数"""
    # 转为连续信号
    continuous = np.convolve(matrix[neuron_idx], np.ones(window)/window, mode='same')
    # 计算指数（要求数据长度足够）
    lyap = nolds.lyap_r(continuous, min_tsep=10, lag=5)
    return lyap

# 生成测试数据（混沌模拟）
matrix = np.zeros((50, 1000))
for t in range(1, 1000):
    matrix[:, t] = (matrix[:, t-1] + np.random.rand(50)) % 1 > 0.5  # 伪混沌模型

lyap = calculate_lyap(matrix)
print(f"最大李雅普诺夫指数: {lyap:.3f}")

if lyap > 0:
    print("系统处于混沌状态（失稳）")