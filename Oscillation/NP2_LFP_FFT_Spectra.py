"""
# coding: utf-8
@author: Yuhao Zhang
last updated: 05/22/2025
data from: Xinchao Chen
Befor Running, Please check two things
1. FFT of LFP signal must add Window, without that the FFT result will be fault
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
from scipy.signal import find_peaks
import scipy.io
np.set_printoptions(threshold=np.inf)

# ------- NEED CHANGE -------
data_path = '/data2/zhangyuhao/xinchao_data/Givenme/1670-2-tremor-Day5-bank_1CVC-FM_g0'
save_path_heatmap = '/home/zhangyuhao/Desktop/Result/ET/LFP_FFT_chs_heatmap/NP2/givenme/1670-2-tremor-Day5-bank_1CVC-FM_g0'
save_path_single_ch = '/home/zhangyuhao/Desktop/Result/ET/LFP_FFT_singlech_spectra/NP2/givenme/1670-2-tremor-Day5-bank_4CVC-FM_g0'
mannual_exclude_ch = None # None
# ------- NO NEED CHANGE -------
chanmap_mat = scipy.io.loadmat(data_path + '/Rawdata/ChanMap.mat')  # 加载ChanMap.mat文件
# 从ChanMap.mat中提取ycoords
ycoords = np.array(chanmap_mat['ycoords']).flatten()
fs = 30000  # 30 kHz for NP2
freq_low, freq_high = 0.5, 30
marker = pd.read_csv(data_path + "/Marker/static_motion_segement.csv")
print(marker)
LFP = np.load(data_path + "/LFP/LFP_npy/export.npy")
if mannual_exclude_ch is not None and len(mannual_exclude_ch) > 0:
    LFP = np.delete(LFP, mannual_exclude_ch, axis=0)
print("Test if LFP duration same as marker duration...")
print(f"LFP duration: {LFP.shape[1]/fs}")
print(f"Marker duration: {marker['time_interval_right_end'].iloc[-1]}")
print(f"LFP shape: {LFP.shape}")

# If there are excluded channels from spikeinterface export, use this
exclu_chs = np.load(data_path + "/LFP/excluded_channels.npy")
print(f"Exclude channel: {exclu_chs}")
ycoords = np.delete(ycoords, exclu_chs)  # 删除ycoords中exclu_chs序号的元素

# 根据ycoords从小到大对LFP进行排序（从下到上）
print("Rearrange LFP channels...")
rearrange_ch_id = np.argsort(ycoords) # 从小到大排序，即probe从下到上  其实就是重排后的LFP每个通道的原始编号，因为原始LFP是按照0-384顺序编号的
arranged_LFP = LFP[rearrange_ch_id]

def Grouped_to_get_real_LFP():
    """
    将LFP每32个通道加在一起，返回合并后的LFP
    """
    print("Grouping to get real LFP...")
    group_size = 32
    n_channels, n_samples = LFP.shape
    n_groups = n_channels // group_size
    grouped_LFP = []
    for i in range(n_groups):
        group = LFP[i*group_size:(i+1)*group_size, :]
        grouped_LFP.append(np.sum(group, axis=0))
    
    return np.array(grouped_LFP)

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
    plt.savefig(save_path+f"{state}_trial{trial}_{freq_low}_{freq_high}.png")
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

def plot_raw_heatmap(freqs,n_channels,all_psd,state,time_interval):
    print("psd_db max:", np.max(all_psd).item())
    print("psd_db min:", np.min(all_psd).item())
    X, Y = np.meshgrid(freqs, np.arange(n_channels))
    pc = plt.pcolormesh(X, Y, all_psd,
        shading="gouraud", 
        cmap="plasma",
        vmax=10,
        vmin=0,
        rasterized=True
    )
        #shading='auto',
        #cmap='plasma')
    cbar = plt.colorbar(pc, pad=0.02)
    cbar.set_label('PSD', rotation=270, labelpad=20)
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Channel Number not channel id', fontsize=12)
    plt.xlim(freq_low, freq_high)
    plt.yticks(np.arange(0, n_channels, 10), fontsize=8)
    title = f'Multi-channel Spectrum Heatmap ({freq_low}-{freq_high}Hz) ({time_interval}s)'
    plt.title(title, fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(save_path_heatmap+f"/{state}_{time_interval}_raw.png")
    plt.clf()

def plot_normalized_heatmap(freqs,n_channels,all_psd,state,time_interval):
    # 缩放
    all_psd = np.log10(all_psd + 1e-12)
    # 沿着每一列进行normalize，即不同频率的幅值都可以得到体现
    all_psd = all_psd.T
    all_psd = np.array([(x_i - np.nanmean(x_i)) / np.std(x_i) for x_i in all_psd])
    all_psd = all_psd.T
    # 高斯滤波
    all_psd = gaussian_filter(all_psd, sigma=6)
    print("psd_db max:", np.max(all_psd).item())
    print("psd_db min:", np.min(all_psd).item())
    X, Y = np.meshgrid(freqs, np.arange(n_channels))
    pc = plt.pcolormesh(X, Y, all_psd,
        shading="gouraud", 
        cmap="Spectral_r",
        vmax=1.2,
        vmin=-1.2,                                      
        rasterized=True
    )
    cbar = plt.colorbar(pc, pad=0.02)
    cbar.set_label('Z-score', rotation=270, labelpad=20)
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Channel Number not channel id', fontsize=12)
    plt.xlim(freq_low, freq_high)
    plt.yticks(np.arange(0, n_channels, 10), fontsize=8)
    title = f'Multi-channel Spectrum Heatmap ({freq_low}-{freq_high}Hz) ({time_interval}s)'
    plt.title(title, fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(save_path_heatmap+f"/{state}_{time_interval}_normalize.png")
    plt.clf()

def plot_each_ch_spectra(freqs,rearrange_ch_ids,all_psd,state,trial):
    plt.figure(figsize=(12, 6))
    for i in range(len(rearrange_ch_ids)):
        ch_id = rearrange_ch_ids[i]
        plt.plot(freqs, all_psd[i], label=f'Ch{ch_id}', alpha=0.7)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('FFT Amplitude')
        plt.title(f'Channel FFT Amplitude vs Frequency ')
        plt.xlim(freq_low, freq_high)
        plt.legend(loc='upper right', fontsize=8, ncol=3)
        plt.tight_layout()
        plt.savefig(save_path_single_ch + f"/{state}_trial{trial}_ch{ch_id}_fft_amplitude.png")
        plt.clf()

def fq_heatmap_cupy(data, state, time_interval):
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
        # PSD calculate |FFT|^2 / (fs * window_power)
        psd_gpu = cp.abs(fft_result[freq_mask])**2 / (fs * window_power)
        all_psd.append(psd_gpu.get())  # 修改14：转回CPU

    all_psd = np.array(all_psd)
    plot_raw_heatmap(freqs,n_channels,all_psd,state,time_interval)
    plot_normalized_heatmap(freqs,n_channels,all_psd,state,time_interval)
    #plot_each_ch_spectra(freqs,n_channels,all_psd,state,trial)

def main():
    processed_LFP = arranged_LFP
    #processed_LFP = Grouped_to_get_real_LFP()
    print(f"Processed LFP size:{processed_LFP.shape}")
    plt.figure(figsize=(14, 8))
    ## compute static time interval
    start_time, end_time = 750,800
    start, end = int(start_time * fs), int(end_time * fs)
    trail_LFP = processed_LFP[:, start:end]
    fq_heatmap_cupy(trail_LFP, 'stop', f'{start_time}-{end_time}') 
    '''
    ## compute by trials divided by motion
    for i in range(len(marker['run_or_stop'])):
        print(f"Processing trial {i}...")
        start = int(marker['time_interval_left_end'].iloc[i] * fs)
        end = int(marker['time_interval_right_end'].iloc[i] * fs)
        state = marker['run_or_stop'].iloc[i]
        state_name = 'run' if state == 1 else 'stop'
        trail_LFP = processed_LFP[:, start:end]
        if trail_LFP.shape[1] < 20 * fs:
            print("too short, ignore")
            continue
        fq_heatmap_cupy(trail_LFP, state_name,i) 
    '''
main()