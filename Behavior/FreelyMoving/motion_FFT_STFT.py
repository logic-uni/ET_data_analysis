"""
# coding: utf-8
@author: Yuhao Zhang
last updated: 04/26/2025
data from: Xinchao Chen
"""
## Most Important, FFT of non-stationary signal must add Window, without that will get fault result
from scipy.fft import fft, fftfreq  
from scipy import signal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from scipy.signal import iirnotch, filtfilt
import cupy as cp
from cupyx.scipy.fft import rfftfreq
from scipy.ndimage import gaussian_filter
import cupyx.scipy.signal as signal
np.set_printoptions(threshold=np.inf)

# ------- NEED CHANGE -------
data_path = "/data1/zhangyuhao/xinchao_data/NP2/test/control/Mice_1411_3/20250109_control_Mice_1411_3_VN_head_fixation"
save_path = "/home/zhangyuhao/Desktop/Result/ET/Motion_FFT/NP2/test/control/Mice_1411_3/20250109_control_Mice_1411_3_VN_head_fixation"
freq_low, freq_high = 0.5, 30
start_time = 438
end_time = 439.6 #marker["time_interval_right_end"].iloc[-1]

# ------- NO NEED CHANGE -------
fs = 10593.2
marker = pd.read_csv(data_path + "/Behavior/marker.csv")
motion_data = np.load(data_path + "/Behavior/motion.npy")
motion_data = motion_data[0]
print(marker)

#注意以下使用的函数 np.fft.fftfreq 中，1/fs表示相邻样本之间的时间间隔,因此fs必须是实际数据真实的采样率
def stft_cupy(data, state, start, end):
    # --- 输入校验 ---
    data = cp.atleast_2d(data)
    if data.shape[0] != 1:
        raise ValueError("输入必须为单通道数据")

    # --- 参数优化 ---
    freq_range = (freq_low, freq_high)              # 聚焦频带范围
    window_sec = 25                     # 时间窗口延长以提高频率分辨率
    nperseg = int(fs * window_sec)     # 21186 samples @ fs=10593.2Hz
    noverlap = int(nperseg * 0.5)     # 75%重叠平衡分辨率与计算量
    nfft = nperseg * 2                 # 补零提高频率插值精度

    # --- 数据校验 ---
    n_samples = data.shape[1]
    if n_samples < nperseg:
        raise ValueError(f"需要至少{nperseg}样本，当前{n_samples}")

    # --- GPU计算 ---
    data_gpu = cp.asarray(data)
    window = cp.hanning(nperseg)
    f, t, Zxx = signal.stft(
        data_gpu[0], fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        boundary="even"
    )
    # --- PSD计算 ---
    '''
    S1 = cp.sum(window**2)
    psd = cp.abs(Zxx)**2 / (S1 * fs)
    psd_db = 10 * cp.log10(psd + 1e-12)
    '''
    # --- PSD计算 (normalize) ---
    S1 = cp.sum(window**2)
    psd = cp.abs(Zxx)**2 / (S1 * fs)
    psd_cpu = cp.asnumpy(psd)
    psd_db_cpu = 10 * np.log10(psd_cpu + 1e-12)
    psd_db_cpu = np.array([(x_i - np.nanmean(x_i)) / np.std(x_i) for x_i in psd_db_cpu])
    #psd_db_cpu = gaussian_filter(psd_db_cpu, sigma=1.5)
    psd_db = cp.asarray(psd_db_cpu)  # 转回GPU
    
    # --- 频率筛选 ---
    freq_mask = (f >= freq_range[0]) & (f <= freq_range[1])
    f_filtered = f[freq_mask]
    psd_db = psd_db[freq_mask, :]
    print("psd_db max:", cp.max(psd_db).item())
    print("psd_db min:", cp.min(psd_db).item())

    # --- 可视化增强 ---
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(14, 8))
    
    # 功率谱密度图
    t_absolute = start + cp.asnumpy(t)
    im = ax1.pcolormesh(
        t_absolute, 
        cp.asnumpy(f_filtered),
        cp.asnumpy(psd_db),
        shading="gouraud", 
        cmap="Spectral_r",
        vmax=cp.max(psd_db).item(),  #根据psd max调整
        vmin=cp.min(psd_db).item(),
        rasterized=True
    )
    ax1.set_ylabel("Frequency (Hz)")
    ax1.set_title(f"PSD - {state} (Δf={1/window_sec:.2f}Hz)")

    # 原始信号时域图
    time_axis = np.linspace(start, end, n_samples)
    ax2.plot(time_axis, cp.asnumpy(data_gpu[0]), lw=0.5)
    ax2.set(xlim=(start, end), xlabel="Time (s)", ylabel="Amplitude")
    ax2.set_title(f"Time Domain Signal ({state})")

    plt.tight_layout()
    plt.savefig(f"{save_path}/PSD_{state}.png", dpi=150)
    plt.close()

def plot_motion_raw_data(data, state, start, end):
    n_samples = data.shape[0]
    time_axis = np.linspace(start, end, n_samples)
    plt.figure(figsize=(12, 4))
    plt.plot(time_axis, data, lw=0.7)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"Raw Motion Data ({state})")
    plt.tight_layout()
    plt.savefig(f"{save_path}/Raw_{state}.png", dpi=150)
    plt.close()

def FFT_cupy(data, time_interval):
    n_samples = data.shape[0]
    # 数据转移到GPU
    data_gpu = cp.asarray(data)
    # 使用cupy计算频率
    freqs = rfftfreq(n_samples, 1/fs)
    freq_mask = (freqs >= freq_low) & (freqs <= freq_high)
    freqs = freqs[freq_mask].get()  # 转回CPU
    # cupy生成窗
    window = cp.hanning(n_samples)
    window_power = cp.sum(window**2)  # 计算窗函数能量
    # GPU处理
    signal_gpu = data_gpu * window
    fft_result = cp.fft.rfft(signal_gpu)
    # PSD calculate |FFT|^2 / (fs * window_power)
    psd_gpu = cp.abs(fft_result[freq_mask])**2 / (fs * window_power)
    psd = psd_gpu.get()  # 转回CPU
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, psd)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD')
    plt.title(f'Motion_{time_interval}s')
    plt.tight_layout()
    plt.savefig(f"{save_path}/PSD_{time_interval}.png", dpi=150)
    plt.close()

def main():
    start_sample = int(start_time * fs)
    end_sample = int(end_time * fs)
    trunc_motion_data = motion_data[start_sample:end_sample]
    plot_motion_raw_data(trunc_motion_data,f'{start_time}-{end_time}', start_time, end_time)
    FFT_cupy(trunc_motion_data, f'{start_time}-{end_time}')
    '''
    stft_cupy(
        trunc_motion_data,
        state='whole',
        start=start_time,
        end=end_time
    )
    '''
main()