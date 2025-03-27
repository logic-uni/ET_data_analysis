import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from statsmodels.tsa.stattools import acf

# 加载数据
series1_path = r"C:\Users\zyh20\Desktop\Research\01_ET_data_analysis\Research\ISI_distribution\xinchao_np2_benchmark\20250310_benchmark_tremor_VN_spike_time_neuron_5.npy"
series2_path = r"C:\Users\zyh20\Desktop\Research\01_ET_data_analysis\Research\ISI_distribution\jiejue_data_benchmark\controlVN_spike_time_20221118_NP1_session3_neuron_30.npy"

t1 = np.load(series1_path)  # 替换为实际文件名
t2 = np.load(series2_path)


# 设定分箱宽度（根据数据调整）
bin_width = 0.1
max_time = max(np.max(t1), np.max(t2))
bins = np.arange(0, max_time + bin_width, bin_width)

# 生成计数序列
counts1, _ = np.histogram(t1, bins=bins)
counts2, _ = np.histogram(t2, bins=bins)
'''
# 计算ACF，设置合适的滞后阶数
nlags = 40
acf1 = acf(counts1, nlags=nlags, fft=True)
acf2 = acf(counts2, nlags=nlags, fft=True)

# 绘制结果
plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 1)
plt.stem(range(nlags+1), acf1, use_line_collection=True)
plt.title('ACF of Series 1')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')

plt.subplot(2, 1, 2)
plt.stem(range(nlags+1), acf2, use_line_collection=True)
plt.title('ACF of Series 2')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')

plt.tight_layout()
plt.show()
'''

def plot_spectrum(counts, bin_width, title):
    """
    计算并绘制功率谱密度
    :param counts: 分箱后的计数序列
    :param bin_width: 分箱宽度（秒）
    :param title: 图像标题
    """
    # 计算采样频率（Hz）
    fs = 1 / bin_width  # 例如 bin_width=0.1秒 → fs=10Hz
    
    # 使用Welch方法估计PSD
    f, Pxx = signal.welch(
        counts,
        fs=fs,
        window='hann',    # 汉宁窗减少频谱泄漏
        nperseg=256,      # 分段长度
        scaling='density' # 功率谱密度
    )
    
    # 可视化
    plt.figure(figsize=(10, 4))
    plt.semilogy(f, Pxx, color='blue')
    plt.title(f'Power Spectrum: {title}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.grid(True)
    
    # 标注最高峰
    peak_idx = np.argmax(Pxx[1:]) + 1  # 排除0频率
    plt.annotate(f'Peak: {f[peak_idx]:.2f} Hz\n(Period: {1/f[peak_idx]:.2f} sec)',
                 xy=(f[peak_idx], Pxx[peak_idx]),
                 xytext=(f[peak_idx]+0.2, Pxx[peak_idx]*0.8),
                 arrowprops=dict(arrowstyle="->"))
    plt.show()

# 分析Series 1（假设bin_width=0.1秒）
plot_spectrum(counts1, bin_width=0.1, title="Series 1")

# 分析Series 2
plot_spectrum(counts2, bin_width=0.1, title="Series 2")