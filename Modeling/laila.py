import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon, gamma

# 混合分布参数
p = 0.2  # 指数成分权重
lambda_1 = 1 / 10  # 指数成分速率 (峰值在10ms左右)
k = 4  # Gamma 分布的形状参数
beta = (k - 1) / 60  # 调整 beta 以使主峰位于60ms

# 生成样本数量
n_samples = 10000

# 生成混合样本
exp_samples = expon.rvs(scale=1/lambda_1, size=n_samples)
gamma_samples = gamma.rvs(a=k, scale=1/beta, size=n_samples)

# 生成混合分布
mixture_samples = np.where(np.random.rand(n_samples) < p, exp_samples, gamma_samples)

# 绘制分布直方图
plt.figure(figsize=(8, 5))
plt.hist(mixture_samples, bins=200, density=False, alpha=0.7)
plt.xlabel("Time interval (ms)")
plt.ylabel("Count")
plt.title("Simulated Mixed Distribution of Event Intervals")
plt.show()