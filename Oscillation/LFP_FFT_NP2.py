"""
# coding: utf-8
@author: Yuhao Zhang
last updated: 04/25/2025
data from: Xinchao Chen
"""
## Most Important, FFT of LFP signal must add Window, without that will get fault result
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
from scipy.interpolate import interp1d
np.set_printoptions(threshold=np.inf)

# For test data
# ------- NEED CHANGE -------
mice_path = '/headtremor/Mice_1410_1/20250313_tremor_Mice_1410_1_VN_freely_moving'
region_name = 'VN'
freq_low, freq_high = 1, 30
# ------- NO NEED CHANGE -------
fs = 30000  # 30 kHz for NP2
data_path = '/data2/zhangyuhao/xinchao_data/NP2/test' + mice_path
marker = pd.read_csv(data_path + "/Marker/static_motion_segement.csv")
print(marker)
LFP = np.load(data_path + "/LFP_npy/export.npy")
save_path = "/home/zhangyuhao/Desktop/Result/ET/LFP_FFT/NP2/test" + mice_path

'''
# For 3 demo data
# ------- NEED CHANGE -------
mice_name = '20250310_VN_control'  # 20250310_VN_control 20250310_VN_harmaline  20250310_VN_tremor
region_name = 'VN'
freq_low, freq_high = 1, 30
# ------- NO NEED CHANGE -------
fs = 30000  # 30 kHz for NP2
marker = pd.read_csv(f"/data1/zhangyuhao/xinchao_data/NP2/{mice_name}/Marker/static_motion_segement.csv")
print(marker)
LFP = np.load(f"/data1/zhangyuhao/xinchao_data/NP2/{mice_name}/LFP_npy/{mice_name}.npy")
save_path = f"/home/zhangyuhao/Desktop/Result/ET/LFP_FFT/NP2/{mice_name}/"  
#save_path = f"/home/zhangyuhao/Desktop/Result/ET/LFP_STFT/{mice_name}/whole/"
'''

print("Test if LFP duration same as marker duration...")
print(f"LFP duration: {LFP.shape[1]/fs}")
print(f"marker duration: {marker['time_interval_right_end'].iloc[-1]}")
print(f"LFP shape: {LFP.shape}")

# ------- FUNCTIONS -------
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
    """分析所有频段的多通道频谱"""
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

def fq_spec_cupy(data, state, trial, title_suffix=""):
    n_channels, n_samples = data.shape
    
    # 修改2：将数据转移到GPU（假设原始数据在CPU）
    data_gpu = cp.asarray(data)  # 如果输入数据已经是GPU数组可省略
    
    # 修改3：使用cupy生成汉宁窗
    window = cp.hanning(n_samples)  # 或者 cp.blackman等
    window_correction = cp.sum(window)  # 修改4：GPU计算窗能量
    
    # 修改5：使用cupy计算频率
    freqs = rfftfreq(n_samples, 1/fs)
    freq_mask = (freqs >= freq_low) & (freqs <= freq_high)
    freqs = freqs[freq_mask].get()  # 修改6：转回CPU用于绘图
    
    all_spectra = []
    for i in range(n_channels):
        # 修改7：在GPU上执行加窗和FFT
        signal_gpu = data_gpu[i] * window
        fft_result = cp.fft.rfft(signal_gpu)
        
        # 修改8：GPU处理频谱
        spectrum_gpu = cp.abs(fft_result[freq_mask]) * 2 / window_correction
        spectrum = spectrum_gpu.get()  # 转回CPU
        all_spectra.append(spectrum)
        
        # 绘图部分保持原样（使用CPU数据）
        linewidth = 1.5 if i in [0, n_channels//2, n_channels-1] else 0.8
        plt.plot(freqs, spectrum, 
                 color=cm.viridis(i/n_channels),
                 alpha=0.7,
                 linewidth=linewidth,
                 label=f'Ch{i+1}' if i % 10 == 0 else "")
               
    # 设置图形属性
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.xlim(freq_low, freq_high)
    plt.ylim(0, np.max(all_spectra)*1.1)
    plt.grid(True, linestyle='--', alpha=0.4)
    
    # 添加颜色条表示通道编号
    ax = plt.gca()  # 获取当前坐标轴
    sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(vmin=1, vmax=n_channels))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)  # 明确指定 ax
    cbar.set_label('Channel Number', rotation=270, labelpad=15)
    
    # 设置标题和图例
    title = f'Multi-channel Spectrum ({freq_low}-{freq_high}Hz){title_suffix}'
    plt.title(title, fontsize=14, pad=20)
    plt.legend(loc='upper right', fontsize=8, ncol=3)
    
    plt.tight_layout()
    plt.savefig(save_path+f"/spectrum/{region_name}_{state}_trial{trial}.png")
    plt.clf()
        
def fq_heatmap_cupy(data, state, trial, title_suffix=""):
    n_channels, n_samples = data.shape
    
    # 修改9：数据转移到GPU
    data_gpu = cp.asarray(data)
    
    # 修改10：使用cupy计算频率
    freqs = rfftfreq(n_samples, 1/fs)
    freq_mask = (freqs >= freq_low) & (freqs <= freq_high)
    freqs = freqs[freq_mask].get()  # 转回CPU
    
    # 修改11：cupy生成窗
    window = cp.hanning(n_samples)
    
    all_spectra = []
    for i in range(n_channels):
        # 修改12：GPU处理
        signal_gpu = data_gpu[i] * window
        fft_result = cp.fft.rfft(signal_gpu)
        spectrum_gpu = cp.abs(fft_result[freq_mask])
        all_spectra.append(spectrum_gpu.get())  # 修改13：转回CPU
    
    # 绘图部分保持不变（使用CPU数据）
    all_spectra = np.array(all_spectra)
    X, Y = np.meshgrid(freqs, np.arange(n_channels))

    pc = plt.pcolormesh(X, Y, all_spectra,
                       shading='auto',
                       cmap='plasma')
    
    plt.gca().invert_yaxis()  # 保证通道0在底部
    cbar = plt.colorbar(pc, pad=0.02)
    cbar.set_label('Amplitude', rotation=270, labelpad=20)
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Channel Number', fontsize=12)
    plt.xlim(freq_low, freq_high)
    plt.yticks(np.arange(0, n_channels, 10), fontsize=8)
    title = f'Multi-channel Spectrum Heatmap ({freq_low}-{freq_high}Hz){title_suffix}'
    plt.title(title, fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path+f"/heatmap/{region_name}_{state}_trial{trial}_heatmap.png")
    plt.clf()

def stft_spectrum_cupy(data, state, title_suffix="", fs=30000):
    n_channels, n_samples = data.shape
    freq_low, freq_high = 0.8, 40
    
    # 关键参数调整 ---------------------------------------------------
    window_sec = 2.0                  # 增大时间窗口到2秒
    nperseg = int(fs * window_sec)    # 60000 samples (2秒窗口)
    noverlap = int(nperseg * 0.9)     # 保持90%重叠 (54000 samples)
    nfft = nperseg                    # 频率分辨率 0.5Hz (1/window_sec)
    
    # 数据长度校验
    if n_samples < nperseg:
        raise ValueError(f"数据长度不足！需要至少{nperseg}样本，当前{n_samples}")

    # GPU计算部分 ----------------------------------------------------
    data_gpu = cp.asarray(data)
    
    for ch_idx in range(n_channels):
        # 执行STFT (GPU加速)
        f, t, Zxx = signal.stft(
            data_gpu[ch_idx],
            fs=fs,
            window=cp.hanning(nperseg),
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            boundary='zeros'  # 处理边界效应
        )
        
        # 频率筛选与标准化 ---------------
        magnitude = cp.abs(Zxx)
        freq_mask = (f >= freq_low) & (f <= freq_high)
        f_filtered = f[freq_mask]
        magnitude = magnitude[freq_mask, :]  # 保持二维结构
        
        # 行方向Z-score标准化
        mean = cp.mean(magnitude, axis=1, keepdims=True)
        std = cp.std(magnitude, axis=1, keepdims=True)
        cp.maximum(std, 1e-6, out=std)
        z_scores = (magnitude - mean) / std
        
        # 数据转换回CPU --------------------
        t_cpu = cp.asnumpy(t)
        f_cpu = cp.asnumpy(f_filtered)
        z_cpu = cp.asnumpy(z_scores)
        
        # 可视化设置 ----------------------
        plt.pcolormesh(t_cpu, f_cpu, z_cpu,
                      shading='gouraud',
                      cmap='viridis',
                      vmin=-3, vmax=3,
                      rasterized=True)  # 加速大图保存
        plt.colorbar(label='Normalized Power')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.ylim(f_filtered.min(), freq_high)  # 精确显示筛选范围
        plt.title(f'Ch{ch_idx+1} {state} (Δf={1/window_sec}Hz)')
        plt.tight_layout()

        # 保存并清理
        plt.savefig(f"{save_path}/{region_name}_{state}_ch{ch_idx+1}.png")
        plt.close()

def plot_avg_trial_ch_spectrum(avg_spectra, freqs, state, title_suffix=""):
    plt.plot(freqs, avg_spectra)
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.xlim(freq_low, freq_high)
    plt.ylim(0, np.max(avg_spectra)*1.1)
    plt.title(f'Average Spectrum ({freq_low}-{freq_high}Hz){title_suffix} - {state}', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(f"{save_path}/{region_name}_avg_{state}.png")
    plt.clf()

def plot_avg_trial_spectrum(avg_spectra, freqs, state, title_suffix=""):
    """绘制平均后的频谱图"""
    n_channels, _ = avg_spectra.shape
    
    plt.figure(figsize=(14, 8))
    for i in range(n_channels):
        spectrum = avg_spectra[i]
        linewidth = 1.5 if i in [0, n_channels//2, n_channels-1] else 0.8
        plt.plot(freqs, spectrum,
                 color=cm.viridis(i/n_channels),
                 alpha=0.7,
                 linewidth=linewidth,
                 label=f'Ch{i+1}' if i % 10 == 0 else "")
    
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.xlim(freq_low, freq_high)
    plt.ylim(0, np.max(avg_spectra)*1.1)
    plt.grid(True, linestyle='--', alpha=0.4)
    
    ax = plt.gca()
    sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(vmin=1, vmax=n_channels))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Channel Number', rotation=270, labelpad=15)
    
    title = f'Average Spectrum ({freq_low}-{freq_high}Hz){title_suffix} - {state}'
    plt.title(title, fontsize=14, pad=20)
    plt.legend(loc='upper right', fontsize=8, ncol=3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/{region_name}_avg_chs_{state}.png")
    plt.clf()

def compute_spectrum(data):
    """返回原始频谱和完整频率轴"""
    n_channels, n_samples = data.shape
    data_gpu = cp.asarray(data)
    window = cp.hanning(n_samples)
    window_correction = cp.sum(window)
    
    # 计算完整频率轴
    freqs = cp.fft.rfftfreq(n_samples, 1/fs).get()  # 转换为CPU数组
    spectra = []
    for i in range(n_channels):
        signal_gpu = data_gpu[i] * window
        fft_result = cp.fft.rfft(signal_gpu)
        spectrum = (cp.abs(fft_result) * 2 / window_correction).get()
        spectra.append(spectrum)
    return np.array(spectra), freqs


def fq_spectrum(data, state, trial, title_suffix=""):
    n_channels, n_samples = data.shape
    
    # 汉宁窗
    window = np.hanning(n_samples)
    window_correction = np.sum(window)  # 窗能量补偿系数
    
    # 计算正频率
    freqs = np.fft.rfftfreq(n_samples, 1/fs)
    freq_mask = (freqs >= freq_low) & (freqs <= freq_high)
    freqs = freqs[freq_mask]
    
    # 逐通道处理
    all_spectra = []
    for i in range(n_channels):
        # 加窗FFT
        fft_result = np.fft.rfft(data[i] * window)
        # 幅值补偿
        spectrum = np.abs(fft_result[freq_mask]) * 2 / window_correction
        all_spectra.append(spectrum)
        
        # 绘图设置（保持原可视化参数）
        linewidth = 1.5 if i in [0, n_channels//2, n_channels-1] else 0.8
        plt.plot(freqs, spectrum, 
                 color=cm.viridis(i/n_channels),
                 alpha=0.7,
                 linewidth=linewidth,
                 label=f'Ch{i+1}' if i % 10 == 0 else "")
        
    # 设置图形属性
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.xlim(freq_low, freq_high)
    plt.ylim(0, np.max(all_spectra)*1.1)
    plt.grid(True, linestyle='--', alpha=0.4)

    ax = plt.gca()  # 获取当前坐标轴
    sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(vmin=1, vmax=n_channels))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)  # 明确指定 ax
    cbar.set_label('Channel Number', rotation=270, labelpad=15)
    
    # 设置标题和图例
    title = f'Multi-channel Spectrum ({freq_low}-{freq_high}Hz){title_suffix}'
    plt.title(title, fontsize=14, pad=20)
    plt.legend(loc='upper right', fontsize=8, ncol=3)
    
    plt.tight_layout()
    plt.savefig(save_path+f"/spectrum/{region_name}_{state}_trial{trial}.png")
    plt.clf()

def main():
    plt.figure(figsize=(14, 8))
    # 配置公共频率轴（可根据需要调整分辨率）
    common_freqs = np.linspace(freq_low, freq_high, num=500)  # 500个均匀分布频率点
    
    # 按状态存储插值后的频谱
    run_spectra = []
    stop_spectra = []

    for i in range(len(marker['run_or_stop'])):
        print(f"Processing trial {i}...")
        # 数据获取
        start = int(marker['time_interval_left_end'].iloc[i] * fs)
        end = int(marker['time_interval_right_end'].iloc[i] * fs)
        state = marker['run_or_stop'].iloc[i]
        state_name = 'run' if state == 1 else 'stop'
        trail_LFP = LFP[:, start:end]
        #if trail_LFP.shape[1] < 20 * fs:
            #print("too short, ignore")
            #continue
        #fq_spectrum(trail_LFP, state_name,i)
        
        spectra, freqs = compute_spectrum(trail_LFP)
        
        # 频率筛选
        freq_mask = (freqs >= freq_low) & (freqs <= freq_high)
        spectra = spectra[:, freq_mask]
        freqs = freqs[freq_mask]

        # 频谱插值
        n_channels = spectra.shape[0]
        interp_spectra = np.zeros((n_channels, len(common_freqs)))
        for ch in range(n_channels):
            # 线性插值（超出范围填充0）
            interpolator = interp1d(
                freqs, spectra[ch], 
                kind='linear', 
                bounds_error=False, 
                fill_value=0
            )
            interp_spectra[ch] = interpolator(common_freqs)
        ## 这两句根据需要加
        #print("Average all channels...")
        #avg_interp_spectra = np.mean(interp_spectra, axis=0)
        # 存储结果
        if state_name == 'run':
            run_spectra.append(interp_spectra)
        else:
            stop_spectra.append(interp_spectra)
    print(f"Average all trials...")
    # 计算并绘制平均频谱
    def process_avg(data, state):
        if not data: return
        avg = np.mean(data, axis=0)
        plot_avg_trial_spectrum(avg, common_freqs, state)
    
    process_avg(run_spectra, 'run')
    process_avg(stop_spectra, 'stop')

main()
