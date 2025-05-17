"""
# coding: utf-8
@author: Yuhao Zhang
last updated: 05/15/2025
data from: Xinchao Chen

Befor Running, Please check two things
1. FFT of LFP signal must add Window, without that the FFT resukt will be fault
2. Must exclude bad channel, without that the heatmap and spctra will be fault
"""
from scipy.fft import fft, fftfreq  
from scipy import signal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from matplotlib import cm
from scipy.signal import iirnotch, filtfilt
import cupy as cp
from cupyx.scipy.fft import rfftfreq
import cupyx.scipy.signal as signal
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.signal import find_peaks
import scipy.ndimage as ndi 
np.set_printoptions(threshold=np.inf)

# ------- NEED CHANGE -------
data_path = '/data2/zhangyuhao/xinchao_data/Givenme/1423_15_control-Day1-1CVC-FM_g0/'
save_path = '/home/zhangyuhao/Desktop/Result/ET/LFP_FFT/NP2/givenme/1423_15_control-Day1-1CVC-FM_g0'
region_name = 'VN'

# ------- NO NEED CHANGE -------
fs = 30000  # 30 kHz for NP2
freq_low, freq_high = 8, 30
marker = pd.read_csv(data_path + "/Marker/static_motion_segement.csv")
print(marker)
LFP = np.load(data_path + "/LFP/LFP_npy/export.npy")
print("Test if LFP duration same as marker duration...")
print(f"LFP duration: {LFP.shape[1]/fs}")
print(f"marker duration: {marker['time_interval_right_end'].iloc[-1]}")
print(f"LFP shape: {LFP.shape}")

#注意以下使用的函数 np.fft.fftfreq 中，1/fs表示相邻样本之间的时间间隔,因此fs必须是实际数据真实的采样率
def enhanced_notch_filter(data, notch_freqs=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500], Q=30, n_stages=3):
    """
    增强型多级陷波滤波器
    参数:
        n_stages: 级联的滤波器级数(增加滤波深度)
    """
    filtered_data = data.copy()
    
    for _ in range(n_stages):  # 多级滤波
        for freq in notch_freqs:
            b, a = signal.iirnotch(freq, Q, fs)
            for i in range(data.shape[0]):
                filtered_data[i] = signal.filtfilt(b, a, filtered_data[i])
    
    return filtered_data

def plot_multi_channel_spectrum(data,state,trial, freq_range, title_suffix=""):
    """
    绘制多通道重叠频谱图（所有通道在同一图中）
    参数:
        freq_range: 要显示的频率范围元组 (low, high)
    """
    freq_low, freq_high = freq_range
    n_samples = data.shape[1]
    freqs = fftfreq(n_samples, 1/fs)
    # 创建频率掩码 (排除0Hz附近)
    if freq_low == 2:
        freq_mask = (freqs >= freq_low) & (freqs <= freq_high)
    else:
        freq_mask = (freqs > freq_low) & (freqs <= freq_high)
    
    freqs = freqs[freq_mask]
    n_channels = data.shape[0]
    # 创建颜色映射
    colors = cm.viridis(np.linspace(0, 1, n_channels))
    # 计算并绘制每个通道的频谱
    for i in range(n_channels):
        fft_result = fft(data[i])
        spectrum = np.abs(fft_result[freq_mask])
        plt.plot(freqs, spectrum, color=colors[i], alpha=0.7, linewidth=1, 
                label=f'Ch {i+1}' if i % 5 == 0 else "")  # 每隔5个通道显示一个图例
    # 设置图形属性
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.xlim(freq_low, freq_high)
    plt.grid(True, linestyle='--', alpha=0.6)
    # 添加颜色条表示通道编号
    sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(vmin=1, vmax=n_channels))
    sm.set_array([])
    cbar = plt.colorbar(sm, pad=0.02)
    cbar.set_label('Channel Number', rotation=270, labelpad=15)
    # 设置标题和图例
    title = f'Multi-channel Spectrum ({freq_low}-{freq_high}Hz){title_suffix}'
    plt.title(title, fontsize=14, pad=20)
    plt.legend(loc='upper right', fontsize=8, ncol=3)
    plt.tight_layout()
    plt.savefig(save_path+f"/{region_name}_{state}_trial{trial}_{freq_low}_{freq_high}.png")
    plt.clf()

def analyze_all_bands(data, state, trial):
    # 应用增强型陷波滤波器
    filtered_data = enhanced_notch_filter(data, fs)
    # 定义要分析的频段
    frequency_bands = [
        (2, 100),    # 低频段
        (100, 200),  # 工频谐波频段
        (200, 300),  # 中频段
        (300, 400),  # 中高频段
        (400, 500)   # 高频段
    ]
    # 对各频段进行分析
    for band in frequency_bands:
        low, high = band
        print(f"\nAnalyzing frequency band {low}-{high}Hz...")
        plot_multi_channel_spectrum(
            filtered_data, 
            state,
            trial,
            fs, 
            freq_range=(low, high),
            title_suffix="\n(After Enhanced Notch Filtering)"
        )

def fq_heatmap_cupy_norm(data, state, trial, title_suffix=""):
    n_channels, n_samples = data.shape
    # 数据转移到GPU
    data_gpu = cp.asarray(data)
    # 使用cupy计算频率
    freqs = rfftfreq(n_samples, 1/fs)
    freq_mask = (freqs >= freq_low) & (freqs <= freq_high)
    freqs = freqs[freq_mask].get()  # 转回CPU
    # cupy生成窗
    window = cp.hanning(n_samples)
    window_power = cp.sum(window**2)  # 新增：计算窗函数能量
    all_psd = []  # 变量名改为all_psd
    for i in range(n_channels):
        # GPU处理
        signal_gpu = data_gpu[i] * window
        fft_result = cp.fft.rfft(signal_gpu)
        
        # 修改为PSD计算（|FFT|^2 / (fs * window_power)）
        psd_gpu = cp.abs(fft_result[freq_mask])**2 / (fs * window_power)
        all_psd.append(psd_gpu.get())  # 修改14：转回CPU

    all_psd = np.array(all_psd)
    X, Y = np.meshgrid(freqs, np.arange(n_channels))

    pc = plt.pcolormesh(X, Y, all_psd,
                       shading='auto',
                       cmap='plasma')
    plt.gca().invert_yaxis()  # 保证通道0在底部
    cbar = plt.colorbar(pc, pad=0.02)
    cbar.set_label('PSD', rotation=270, labelpad=20)
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Channel Number', fontsize=12)
    plt.xlim(freq_low, freq_high)
    plt.yticks(np.arange(0, n_channels, 10), fontsize=8)
    title = f'Multi-channel Spectrum Heatmap ({freq_low}-{freq_high}Hz){title_suffix}'
    plt.title(title, fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path+f"/{region_name}_{state}_trial{trial}_heatmap_norm.png")
    plt.clf()

def main():
    plt.figure(figsize=(14, 8))
    for i in range(len(marker['run_or_stop'])):
        print(f"Processing trial {i}...")
        start = int(marker['time_interval_left_end'].iloc[i] * fs)
        end = int(marker['time_interval_right_end'].iloc[i] * fs)
        state = marker['run_or_stop'].iloc[i]
        state_name = 'run' if state == 1 else 'stop'
        trail_LFP = LFP[:, start:end]
        if trail_LFP.shape[1] < 20 * fs:
            print("too short, ignore")
            continue
        #fq_spec_cupy(trail_LFP, state_name,i)
        fq_heatmap_cupy_norm(trail_LFP, state_name,i) 

main()