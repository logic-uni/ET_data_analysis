import numpy as np
import matplotlib.pyplot as plt
from tick.hawkes import SimuHawkes, HawkesExpKern
from statsmodels.tsa.stattools import acf
from tick.hawkes import HawkesExpKern
from statsmodels.tsa.seasonal import STL

# 生成泊松过程（恒定速率）
def generate_poisson(lamb, T):
    n = np.random.poisson(lamb * T)
    return np.sort(np.random.uniform(0, T, n))

# 生成周期性泊松过程（速率含周期性）
def generate_periodic_poisson(base_lamb, amplitude, period, T):
    t = np.linspace(0, T, 1000)
    lamb_t = base_lamb + amplitude * np.sin(2 * np.pi * t / period)
    events = []
    current_time = 0
    while current_time < T:
        current_lamb = base_lamb + amplitude * np.sin(2 * np.pi * current_time / period)
        next_time = current_time + np.random.exponential(1 / current_lamb)
        if next_time < T:
            events.append(next_time)
        current_time = next_time
    return np.array(events)

# 生成Hawkes过程（自激励）
def generate_hawkes(mu, alpha, beta, T):
    hawkes = SimuHawkes(
        baseline=[mu], 
        kernel=[[alpha * beta * np.exp(-beta * t)]], 
        end_time=T, verbose=False
    )
    hawkes.simulate()
    return hawkes.timestamps[0]

# 参数设置
T = 100  # 时间范围
np.random.seed(42)

# 生成三种过程的数据
events_poisson = generate_poisson(lamb=0.5, T=T)
events_periodic = generate_periodic_poisson(base_lamb=0.3, amplitude=0.2, period=10, T=T)
events_hawkes = generate_hawkes(mu=0.1, alpha=0.5, beta=1.0, T=T)

def plot_events(events, title):
    plt.figure(figsize=(10, 2))
    plt.eventplot(events, lineoffsets=0.5, linelengths=0.5)
    plt.title(title)
    plt.xlabel("Time")
    plt.yticks([])
    plt.show()

# 可视化事件序列
plot_events(events_poisson, "Poisson Process")
plot_events(events_periodic, "Periodic Poisson Process")
plot_events(events_hawkes, "Hawkes Process")

# 计算自相关函数（ACF）
def plot_acf(events, max_lag=20):
    intervals = np.diff(events)
    acf_values = acf(intervals, nlags=max_lag, fft=True)
    plt.stem(range(max_lag+1), acf_values)
    plt.title("ACF of Inter-Event Times")
    plt.xlabel("Lag")
    plt.ylabel("ACF")
    plt.show()

plot_acf(events_poisson)     # ACF应接近零
plot_acf(events_periodic)    # ACF可能显示微弱周期
plot_acf(events_hawkes)      # ACF在短滞后上显著非零



# 方法1：拟合Hawkes模型，检查参数显著性
def fit_hawkes(events):
    model = HawkesExpKern(decay=1.0)
    model.fit(events)
    print(f"Hawkes模型参数 - 基线: {model.baseline[0]:.3f}, 自激强度: {model.adjacency[0,0]:.3f}")
    return model.score()  # 返回对数似然

# 方法2：检测周期性（STL分解）
def detect_periodicity(events, period=10):
    counts, bins = np.histogram(events, bins=np.arange(0, T, 1))
    stl = STL(counts, period=period, seasonal=13)
    result = stl.fit()
    result.plot()
    plt.show()

# 方法3：泊松过程检验（Kolmogorov-Smirnov）
def test_poisson(events):
    intervals = np.diff(events)
    from scipy.stats import kstest, expon
    ks_stat, p_value = kstest(intervals, expon.cdf, args=(1/np.mean(intervals),))
    print(f"KS检验p值: {p_value:.3f} (p>0.05接受泊松假设)")

# 对不同数据应用方法
print("=== Poisson Process ===")
test_poisson(events_poisson)
detect_periodicity(events_poisson)

print("\n=== Periodic Poisson Process ===")
test_poisson(events_periodic)  # 预期拒绝泊松假设
detect_periodicity(events_periodic)

print("\n=== Hawkes Process ===")
log_likelihood = fit_hawkes([events_hawkes])
test_poisson(events_hawkes)     # 预期拒绝泊松假设