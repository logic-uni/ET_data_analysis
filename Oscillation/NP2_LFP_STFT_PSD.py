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
import cupyx.scipy.signal as signal
from matplotlib.colors import Normalize

np.set_printoptions(threshold=np.inf)


# For test data
region_name = 'VN'
fs = 30000  # 30 kHz for NP2
mice_path = '/headtremor/Mice_1410_1/20250309_tremor_Mice_1410_1_VN_freely_moving'
data_path = '/data2/zhangyuhao/xinchao_data/NP2/test' + mice_path
marker = pd.read_csv(data_path + "/Marker/static_motion_segement.csv")
print(marker)
LFP = np.load(data_path + "/LFP_npy/export.npy")
save_path = "/home/zhangyuhao/Desktop/Result/ET/LFP_FFT/NP2/test" + mice_path

'''
# For 3 demo data
# ------- NEED CHANGE -------
mice_name = '20250310_VN_harmaline'  # 20250310_VN_control 20250310_VN_harmaline  20250310_VN_tremor
region_name = 'VN'
fs = 10593.2
marker = pd.read_csv(f"/data1/zhangyuhao/xinchao_data/NP2/{mice_name}/Marker/static_motion_segement.csv")
print(marker)
motion_data = np.load(f"/data1/zhangyuhao/xinchao_data/NP2/{mice_name}/Marker/motion_marker.npy")
motion_data = motion_data[0]
save_path = f"/home/zhangyuhao/Desktop/Result/ET/Motion_FFT/{mice_name}/"  
'''
# ------- FUNCTIONS -------
#注意以下使用的函数 np.fft.fftfreq 中，1/fs表示相邻样本之间的时间间隔,因此fs必须是实际数据真实的采样率
def stft_spectrum_cupy(data, ch, start, end, title_suffix="", fs=30000):
    """聚焦10Hz附近的时频功率谱分析 (GPU加速)"""
    # --- 输入校验 ---
    data = cp.atleast_2d(data)
    if data.shape[0] != 1:
        raise ValueError("输入必须为单通道数据")

    # --- 参数优化 ---
    freq_range = (1, 35)              # 聚焦频带范围
    window_sec = 4.0                   # 时间窗口延长以提高频率分辨率
    nperseg = int(fs * window_sec)     # 21186 samples @ fs=10593.2Hz
    noverlap = int(nperseg * 0.75)     # 75%重叠平衡分辨率与计算量
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
    psd_db = 10 * cp.log10(psd + 1e-12)

    # --- 频率聚焦 ---
    freq_mask = (f >= freq_range[0]) & (f <= freq_range[1])
    f_filtered = f[freq_mask]
    psd_db = psd_db[freq_mask, :]

    # --- 可视化增强 ---
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(14, 8))
    print(np.min(cp.asnumpy(psd_db)))
    print(np.max(cp.asnumpy(psd_db)))
    # 功率谱密度图
    t_absolute = start + cp.asnumpy(t)
    im = ax1.pcolormesh(
        t_absolute, 
        cp.asnumpy(f_filtered),
        cp.asnumpy(psd_db),
        shading="gouraud", 
        cmap="inferno",
        vmin=-105,
        vmax=-80,
        rasterized=True
    )
    #vmin=-72,   # 亮了就调大，暗了就调小  freely moving: vmin = -72, vmax = -52, head-fixed: vmin = -60, vmax = -45
    #vmax=-52,
    ax1.set_ylabel("Frequency (Hz)")
    ax1.set_title(f"PSD - channel {ch} (Δf={1/window_sec:.2f}Hz)")
    #fig.colorbar(im, ax=ax1, label='Power (dB/Hz)')

    # 原始信号时域图
    time_axis = np.linspace(start, end, n_samples)
    ax2.plot(time_axis, cp.asnumpy(data_gpu[0]), lw=0.5)
    ax2.set(xlim=(start, end), xlabel="Time (s)", ylabel="Amplitude")
    ax2.set_title(f"Time Domain Signal (channel {ch})")

    plt.tight_layout()
    plt.savefig(f"{save_path}/PSD_{region_name}_ch_{ch}.png", dpi=150)
    plt.close()

def main():
    start_time = 0 #1330
    end_time = marker["time_interval_right_end"].iloc[-1] #1368
    start_sample = int(start_time * fs)
    end_sample = int(end_time * fs)
    for ch in range(LFP.shape[0]):
        ch_LFP = LFP[0]
        trail_motion_data = ch_LFP[start_sample:end_sample]
        # 调用函数
        stft_spectrum_cupy(
            trail_motion_data,
            ch=ch,
            start=start_time,
            end=end_time
        )

main()
