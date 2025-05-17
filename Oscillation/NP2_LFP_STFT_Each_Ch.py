"""
# coding: utf-8
@author: Yuhao Zhang
last updated: 05/16/2025
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
import cupyx.scipy.signal as signal
from scipy.ndimage import gaussian_filter
np.set_printoptions(threshold=np.inf)

# ------- NEED CHANGE -------
data_path = '/data2/zhangyuhao/xinchao_data/Givenme/1423_15_control-Day1-4CVC-FM_g0'
save_path = "/home/zhangyuhao/Desktop/Result/ET/LFP_FFT/NP2/givenme/1423_15_control-Day1-4CVC-FM_g0"
region_name = 'T2D'
freq_low, freq_high = 0.5, 100
# ------- NO NEED CHANGE -------
fs = 30000  # 30 kHz for NP2
marker_fs = 10593.2 # marker采样率
LFP = np.load(data_path + "/LFP/LFP_npy/export.npy")
motion_data = np.load(data_path + "/Marker/motion_marker.npy")
motion_data = motion_data[0]
print("Test if LFP duration same as marker duration...")
print(f"LFP duration: {LFP.shape[1]/fs} s")
print(f"marker duration: {len(motion_data)/marker_fs} s")
print(f"LFP shape: {LFP.shape}")

#注意以下使用的函数 np.fft.fftfreq 中，1/fs表示相邻样本之间的时间间隔,因此fs必须是实际数据真实的采样率
def stft_spectrum_cupy(data, ch, start_time, end_time, title_suffix=""):
    # --- 输入校验 ---
    data = cp.atleast_2d(data)
    if data.shape[0] != 1:
        raise ValueError("输入必须为单通道数据")

    # --- 参数优化 ---
    freq_range = (freq_low, freq_high)              # 聚焦频带范围
    window_sec = 4                   # 时间窗口延长以提高频率分辨率
    nperseg = int(fs * window_sec)     
    noverlap = int(nperseg * 0.5)     # 50%重叠平衡分辨率与计算量
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
    S1 = cp.sum(window**2)
    psd = cp.abs(Zxx)**2 / (S1 * fs)
    psd_cpu = cp.asnumpy(psd)
    psd_db_cpu = np.log10(psd_cpu + 1e-12)
    psd_db_cpu = np.array([(x_i - np.nanmean(x_i)) / np.std(x_i) for x_i in psd_db_cpu])
    psd_db_cpu = gaussian_filter(psd_db_cpu, sigma=6)
    psd_db = cp.asarray(psd_db_cpu)  # 转回GPU（如需后续处理）
    
    # --- 频率筛选 ---
    freq_mask = (f >= freq_range[0]) & (f <= freq_range[1])
    f_filtered = f[freq_mask]
    psd_db = psd_db[freq_mask, :]

    print("psd_db max:", cp.max(psd_db).item())
    print("psd_db min:", cp.min(psd_db).item())
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(14, 8))
    t_absolute = start_time + cp.asnumpy(t)
    im = ax1.pcolormesh(
        t_absolute, 
        cp.asnumpy(f_filtered),
        cp.asnumpy(psd_db),
        shading="gouraud", 
        cmap="Spectral_r",
        vmax=0.7,
        vmin=-0.7,                                      
        rasterized=True
    )
    ax1.set_ylabel("Frequency (Hz)")
    ax1.set_title(f"PSD - channel {ch} (Δf={1/window_sec:.2f}Hz)")

    # motion时域图
    time_axis = np.linspace(start_time, end_time, int(end_time * marker_fs) - int(start_time * marker_fs))
    ax2.plot(time_axis, motion_data[int(start_time*marker_fs):int(end_time*marker_fs)])  # 没动就是平的，动了才会变化，向上向下分别代表压力传感器的方向
    ax2.set(xlim=(start_time, end_time), xlabel="Time (s)", ylabel="Amplitude")
    ax2.set_title("Motion")
    plt.tight_layout()
    plt.savefig(f"{save_path}/STFT_PSD_{start_time}s-{end_time}s_ch{ch}.png", dpi=150)
    plt.close()

def main():
    for i in range(int(len(motion_data)/marker_fs) // 300):
        start_time = i * 300
        end_time = (i + 1) * 300
        print(f"Processing {start_time}s ~ {end_time}s...")
        start_LFP_sample = int(start_time * fs)
        end_LFP_sample = int(end_time * fs)
        LFP_trunc = LFP[:, start_LFP_sample:end_LFP_sample]
        for ch in range(LFP_trunc.shape[0]):
            ch_LFP = LFP_trunc[ch]
            stft_spectrum_cupy(
                ch_LFP,
                ch=ch,
                start_time=start_time,
                end_time=end_time
            )

main()

'''
# --- 去除背景噪音 ---
#noise_floor = cp.mean(psd_db, axis=1, keepdims=True)
#psd_db = psd_db - noise_floor

# LFP时域图
time_axis = np.linspace(start_LFP_sample, end_LFP_sample, n_samples)
ax2.plot(time_axis, cp.asnumpy(data_gpu[0]), lw=0.5)
ax2.set(xlim=(start_LFP_sample, end_LFP_sample), xlabel="Time (s)", ylabel="Amplitude")
ax2.set_title(f"Time Domain Signal (channel {ch})")
'''