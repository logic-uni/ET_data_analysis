"""
# coding: utf-8
@author: Yuhao Zhang
last updated: 04/16/2025
data from: Xinchao Chen
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.signal import find_peaks

mice_name = '20250310_VN_control'  # 20250310_VN_control 20250310_VN_harmaline  20250310_VN_tremor

# ------- NO NEED CHANGE -------
fs = 30000  # 30 kHz for NP2
lfp_data = np.load(f"/data1/zhangyuhao/xinchao_data/NP2/{mice_name}/LFP_npy/{mice_name}.npy")

sorting_path = f"/data1/zhangyuhao/xinchao_data/NP2/{mice_name}/Sorted/"
identities = np.load(sorting_path + '/spike_clusters.npy') # time series: unit id of each spike
times = np.load(sorting_path + '/spike_times.npy')  # time series: spike time of each spike
neurons = pd.read_csv(f"/data1/zhangyuhao/xinchao_data/NP2/{mice_name}/filtered_quality_metrics.csv")  # QC neurons
#neurons = pd.read_csv(sorting_path + '/cluster_group.tsv', sep='\t')   # all neurons
print(neurons)
print(f"LFP duration: {lfp_data.shape[1]/fs}")
print(f"AP duration: {times[-1] / fs}")
save_path = f"/home/zhangyuhao/Desktop/Result/ET/Phase_Lock/NP2/{mice_name}/" 


freqs = np.arange(0.8, 30.1, 0.1)

def compute_vector_strength(spike_times):
    # 生成频率数组 (0.8Hz 到 30Hz，间隔0.1Hz)
    vector_strengths = []
    for freq in freqs:
        # 计算所有spike的相位角 (保留原始计算值)
        phases = 2 * np.pi * freq * spike_times
        # 计算复数指数项的和
        sum_complex = np.sum(np.exp(1j * phases))
        # 计算向量强度并标准化
        vs = np.abs(sum_complex)* 100 / len(spike_times)  # 100是scaler，可去
        vector_strengths.append(vs)
    return np.array(vector_strengths)

def singleneuron_spiketimes(id):
    x = np.where(identities == id)
    y=x[0]
    spike_times=np.empty(len(y))
    for i in range(0,len(y)):
        z=y[i]
        spike_times[i]=times[z]/fs
    return spike_times

def each_neuron_vs_fq():
    plt.figure(figsize=(10, 4))
    for index, row in neurons.iterrows():
        unit_id = row['cluster_id']
        spike_times = singleneuron_spiketimes(unit_id)
        fr = len(spike_times)/(times[-1] / fs)
        if fr > 2:  # filter fr
            vs_fq = compute_vector_strength(spike_times)
            plt.plot(freqs, vs_fq)
            plt.title("Vector Strength vs. Frequency")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Vector Strength")
            plt.grid(True)

    plt.savefig(os.path.join(save_path, f'phase_vector.png'))

def prominence_scale(data):
    """
    Scale data specifically for prominence display:
    1. Subtract baseline (minimum value)
    2. Normalize by maximum prominence
    """
    baseline = np.min(data)
    scaled = data - baseline
    max_prominence = np.max(scaled)
    if max_prominence > 0:
        return scaled / max_prominence
    return scaled  # avoid division by zero if all values are equal

def prominan(spectrum):
    # 1) Define exponential background model: A·exp(−B·x) + C
    def exp_background(x, A, B, C):
        return A * np.exp(-B * x) + C

    # 2) Choose initial guesses
    A0 = spectrum.max() - spectrum.min()
    B0 = 1e-3  # assume slow decay
    C0 = np.percentile(spectrum, 5)  # a little above the absolute min to avoid log issues

    p0 = [A0, B0, C0]

    # 3) Set bounds: A≥0, B≥0, C≥0
    lower = [0, 0, 0]
    upper = [np.inf, np.inf, np.inf]

    # 4) Fit with more function evaluations
    try:
        popt, pcov = curve_fit(
            exp_background, freqs, spectrum,
            p0=p0, bounds=(lower, upper),
            maxfev=5000
        )
        background = exp_background(freqs, *popt)
    except RuntimeError:
        # Fallback: use flat background at the 5th percentile
        popt = [np.nan, np.nan, C0]
        background = np.full_like(spectrum, fill_value=C0)
        print("Warning: exponential fit failed – using constant background.")

    # 5) Subtract background
    spec_corrected = spectrum - background

    # 6) Smooth with Gaussian‐weighted moving average
    window_size = 31
    x_gauss = np.linspace(-3, 3, window_size)
    gauss_kernel = norm.pdf(x_gauss)
    gauss_kernel /= gauss_kernel.sum()
    spec_smooth = np.convolve(spec_corrected, gauss_kernel, mode='same')

    # 7) Peak detection with 1%‐of‐mean prominence
    prom_thresh = 0.01 * np.mean(spec_smooth)
    peaks, props = find_peaks(spec_smooth, prominence=prom_thresh)

    # 8) Collect peak info
    peak_freqs       = freqs[peaks]
    peak_heights     = spec_smooth[peaks]
    peak_prominences = props['prominences']

    prominence_array = np.zeros_like(freqs)
    for freq, prom in zip(peak_freqs, peak_prominences):
        idx = np.argmin(np.abs(freqs - freq))  # Find the closest index in freqs
        prominence_array[idx] = prom
    print(prominence_array)
    return prominence_array

def accumulate():
    current_sum = np.zeros_like(freqs, dtype=np.float32)
    current_sum_history = []  # 用于保存每个步骤的累积向量强度
    # 遍历神经元
    for index, row in neurons.iterrows():
        unit_id = row['cluster_id']
        spike_times = singleneuron_spiketimes(unit_id)
        fr = len(spike_times)/(times[-1]/fs)
        
        if fr > 2:  # most important parameter!! if not 2, you will not get correct answer
            vs = compute_vector_strength(spike_times)
            '''
            # Option 1 traditional scaler
            vs[vs < 0.8] = 0
            vs[vs > 1.2] *= 1.2
            vs = vs * vs * vs #  非常有效，大的越大，小的越小
            '''
            # Option 2 peak detection and prominance calculate
            promi = prominan(vs)
            #promi = promi * promi * promi
            current_sum += promi  # Accumulate smoothed vector strength
            #current_sum_subt = current_sum - np.min(current_sum)  # normalize for ploting heatmap
            current_sum_history.append(current_sum.copy())  # 保存当前状态
    
    # 转换为二维数组（神经元数 x 频率）
    data = np.array(current_sum_history)

    # 创建热图
    plt.figure(figsize=(14, 8))
    plt.imshow(
        data.T,  # 转置，使频率作为纵轴
        aspect='auto',
        cmap='viridis',
        origin='lower',
        extent=[0.5, len(data)+0.5, freqs[0], freqs[-1]],  # 横轴范围
        interpolation='nearest'  # 避免插值
    )

    # 设置坐标轴和标签
    plt.colorbar(label='Cumulative Vector Strength')
    plt.xlabel('Number of Accumulated Neurons')
    plt.ylabel('Frequency (Hz)')

    # 调整横轴刻度为整数
    plt.xticks(np.arange(1, len(data)+1, step=max(1, len(data)//10)))

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'heatmap_accumulation.png'), dpi=300)
    plt.close()

    # 获取最后一个累积状态的数据
    last_slice = current_sum_history[-1]  # 最后一行数据

    # 创建二维曲线图
    plt.figure(figsize=(12, 5))
    plt.plot(freqs, last_slice, 
            color='#2E86C1',  # 使用与热图协调的颜色
            lw=1.5, 
            alpha=0.9)

    # 增强可视化效果
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Cumulative Vector Strength', fontsize=12)
    plt.title('Final Accumulated Vector Strength', fontsize=14, pad=15)

    # 如果频率范围较大，可以设置为对数坐标
    if np.max(freqs) / np.min(freqs) > 100:
        plt.xscale('log')
        plt.xticks([1, 10, 100, 1000], ['1', '10', '100', '1000'])

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'final_accumulation_curve.png'), dpi=300)
    plt.close()

#each_neuron_vs_fq()
accumulate()