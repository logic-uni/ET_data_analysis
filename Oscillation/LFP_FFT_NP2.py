"""
# coding: utf-8
@author: Yuhao Zhang
last updated: 04/15/2025
data from: Xinchao Chen
"""
## Most Important, FFT of LFP signal must add Window, without that will get fault result
from scipy.fft import fft, fftfreq  
from scipy import signal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy.signal import spectrogram
from matplotlib import cm
from scipy.signal import iirnotch, filtfilt
import cupy as cp
from cupyx.scipy.fft import rfftfreq
import cupyx.scipy.signal as signal
np.set_printoptions(threshold=np.inf)

'''
# ------- For s data -------
fs = 30000  # 30 kHz for NP2
region_name = 'VN'
freq_low, freq_high = 16, 40
LFP = np.load("/data1/zhangyuhao/xinchao_data/NP2/stolen/headtremor/02/LFP_npy/export.npy")
save_path = "/home/zhangyuhao/Desktop/Result/ET/LFP_FFT/NP2/stolen/02/" 
'''
# ------- For g data -------
# ------- NEED CHANGE -------
mice_name = '20250310_VN_tremor'  # 20250310_VN_control 20250310_VN_harmaline  20250310_VN_tremor
region_name = 'VN'
freq_low, freq_high = 4, 16

# ------- NO NEED CHANGE -------
fs = 30000  # 30 kHz for NP2
marker = pd.read_csv(f"/data1/zhangyuhao/xinchao_data/NP2/{mice_name}/Marker/static_motion_segement.csv")
print(marker)
LFP = np.load(f"/data1/zhangyuhao/xinchao_data/NP2/{mice_name}/LFP_npy/{mice_name}.npy")
#save_path = f"/home/zhangyuhao/Desktop/Result/ET/LFP_FFT/NP2/{mice_name}/"  
save_path = f"/home/zhangyuhao/Desktop/Result/ET/LFP_STFT/{mice_name}/whole/"

print("Test if LFP duration same as marker duration...")
print(f"LFP duration: {LFP.shape[1]/fs}")
print(f"marker duration: {marker['time_interval_right_end'].iloc[-1]}")
print(f"LFP shape: {LFP.shape}")
'''
def multi_notch_filter(channel_data, fs, notch_freqs=[1, 3, 5, 7], Q=2.0):
    """
    对单个通道应用多频陷波滤波器，滤除指定频率成分。
    
    参数：
        channel_data (ndarray): 输入信号数据，一维数组。
        fs (float): 采样率。
        notch_freqs (list): 需要滤除的频率列表，默认为[1,3,5,7]Hz。
        Q (float): 滤波器的品质因数，控制带宽，Q = f0/带宽。
    
    返回：
        filtered_data (ndarray): 滤波后的信号。
    """
    filtered_data = channel_data.copy().astype(np.float64)  # 避免原地修改原始数据
    nyquist = 0.5 * fs
    for f0 in notch_freqs:
        w0 = f0 / nyquist
        b, a = iirnotch(w0, Q)
        filtered_data = filtfilt(b, a, filtered_data)  # 级联滤波
    return filtered_data

# 假设原始数据为 original_data，形状 (37, 5458909)
# 对每个通道应用滤波器，返回相同形状的数组
filtered_data = np.apply_along_axis(
    func1d=multi_notch_filter,
    axis=1,  # 按行处理每个通道
    arr=LFP,
    fs=2500,
    notch_freqs=[1, 3, 5, 7],
    Q=2.0
)
print(filtered_data.shape)
'''
# ------- FUNCTIONS -------
#注意以下使用的函数 np.fft.fftfreq 中，1/fs表示相邻样本之间的时间间隔,因此fs必须是实际数据真实的采样率

def freq_heat_map(times,frequencies,Sxx,state,m):
    # 频谱热图
    #plt.figure(figsize=(10, 6))
    plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')
    plt.colorbar(label='Intensity [dB]')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.title(f'{region_name}_{state}_Channel{m}_Spectrogram')
    plt.savefig(fig_path+f"/{region_name}_{state}_Channel{m}_Spectrogram.png")
    plt.clf()

def raw_trace_signal(t,signal,state,m):
    # 原始信号
    #plt.figure(figsize=(22, 22))
    plt.plot(t, signal)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title(f"{region_name}_{state}_Channel{m}_Signal")
    plt.savefig(fig_path+f"/{region_name}_{state}_Channel{m}_Signal.png")
    plt.clf()

def freq_expand(freq_vector,fft_result,state,m):
    # 频谱峰值图
    positive_freqs = freq_vector[:len(freq_vector)//2]     # 只考虑正频率部分
    positive_fft_values = np.abs(fft_result[:len(fft_result)//2])
    #plt.figure(figsize=(10, 6))
    plt.plot(positive_freqs, positive_fft_values)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.title(f'{region_name}_{state}_Channel{m}_Frequency Spectrum')
    plt.grid()
    plt.savefig(fig_path+f"/{region_name}_{state}_Channel{m}_Frequency Spectrum.png")
    plt.clf()

def phase_diff_main_freq(data,state,main_phases,main_freqs):
    # 计算相位差矩阵
    phase_diff_matrix = np.zeros((len(data), len(data)))
    for i in range(len(data)):
        for j in range(len(data)):
            phase_diff_matrix[i, j] = main_phases[j] - main_phases[i]
    # 优化相位差矩阵：取绝对值并归一化
    abs_phase_diff_matrix = np.abs(phase_diff_matrix)
    normalized_phase_diff_matrix = (abs_phase_diff_matrix - np.min(abs_phase_diff_matrix)) / (np.max(abs_phase_diff_matrix) - np.min(abs_phase_diff_matrix))
    # 绘制相位差矩阵
    #plt.figure(figsize=(10, 8))
    sns.heatmap(normalized_phase_diff_matrix, annot=False, cmap='coolwarm')
    plt.title(f"{region_name}_{state}_Normalized Nerual Phase Difference Matrix")
    plt.xlabel("LFP Channel")
    plt.ylabel("LFP Channel")
    plt.savefig(fig_path+f"/{region_name}_{state}_phase_diff.png")
    plt.clf()
    #绘制不同通道主频率图
    indices = np.arange(len(main_freqs))
    #fig, ax = plt.subplots(figsize=(10, 6))
    plt.scatter(indices, main_freqs, marker='o', color='b')
    plt.xlabel('LFP channel')
    plt.ylabel('Main frequence')
    plt.title(f'{region_name}_{state}_main_freq')
    plt.grid(True)
    plt.savefig(fig_path+f"/{region_name}_{state}_main_freq.png")
    plt.clf()

def overlay_trial_freq_spetrum(fq_thre_low,fq_thre_high):
    ch = 1
    plt.figure(figsize=(12, 8))
    for signal in LFP:
        for i in range(0,len(treadmill['run_or_stop'])):
            marker_start = int(treadmill['time_interval_left_end'].iloc[i]*fs)
            marker_end = int(treadmill['time_interval_right_end'].iloc[i]*fs)
            ch_trail_LFP = signal[marker_start:marker_end]
            if treadmill['run_or_stop'].iloc[i] == 1:
                color = 'red'
            else:
                color = 'blue'
            # 分析每个信号的频谱
            # 计算信号的傅里叶变换
            fft_result = np.fft.fft(ch_trail_LFP)
            # 计算频率向量
            freq_vector = np.fft.fftfreq(len(ch_trail_LFP), 1/fs)  
            # 只选择正频率部分
            positive_freqs = freq_vector[:len(freq_vector)//2]
            positive_fft_result = np.abs(fft_result[:len(fft_result)//2])
            # 按范围过滤频率
            filtered_indices = (positive_freqs >= fq_thre_low) & (positive_freqs < fq_thre_high)
            filtered_freqs = positive_freqs[filtered_indices]
            filtered_fft_values = positive_fft_result[filtered_indices]
            # 绘制频谱
            plt.plot(filtered_freqs, filtered_fft_values, color=color)

        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude')
        plt.title(f'Frequency Spectrum of {region_name}_ch{ch}')
        plt.grid()
        plt.savefig(savepath_trial+f"/Frequency Spectrum of {region_name}_ch{ch}.png")
        plt.clf()
        ch = ch + 1

def FFT_freq_phase(data,state,savepath_raw,savepath_diff):
    # 储存主要频率和相位信息
    main_freqs = []
    main_phases = []
    m=0
    t=np.arange(0,data.shape[1]/fs,1/fs)
    plt.figure(figsize=(10, 6))
    # 分析每个信号的频谱
    for signal in data:
        # 计算信号的傅里叶变换
        fft_result = np.fft.fft(signal)
        # 计算频率向量
        freq_vector = np.fft.fftfreq(len(signal), 1/fs)
        # 只选择正频率部分
        positive_freqs = freq_vector[:len(freq_vector)//2]
        positive_fft_result = fft_result[:len(fft_result)//2]
        mask = positive_freqs > 0.1  # 过滤掉频率接近0的成分
        positive_freqs = positive_freqs[mask]
        positive_fft_result = positive_fft_result[mask]
        # 找到幅度最大的频率
        idx_max = np.argmax(np.abs(positive_fft_result))
        main_freq = positive_freqs[idx_max]
        main_phase = np.angle(positive_fft_result[idx_max])*100
        main_freqs.append(main_freq)
        main_phases.append(main_phase)
        # 计算频谱
        frequencies, times, Sxx = spectrogram(signal, fs)
        # 频谱热图
        freq_heat_map(times,frequencies,Sxx,state,m,savepath_raw)
        m=m+1
        # 原始信号图
        raw_trace_signal(t,signal,state,m,savepath_raw)
        # 频谱峰值图
        freq_expand(freq_vector,fft_result,state,m,savepath_raw)

    phase_diff_main_freq(data,state,main_phases,main_freqs,savepath_diff)

def fft_channels_previous(signals):
    t = np.arange(0,signals.shape[1]/fs,1/fs)
    # 计算所有信号的傅里叶变换
    fft_values = [np.fft.fft(signal) for signal in signals]
    fft_freq = np.fft.fftfreq(len(t), 1/fs)
    # 只考虑正频率部分
    positive_freqs = fft_freq[:len(fft_freq)//2]
    positive_fft_values = [np.abs(fft[:len(fft)//2]) for fft in fft_values]
    # 按范围过滤频率
    filtered_indices = (positive_freqs >= freq_low) & (positive_freqs < freq_high)
    filtered_freqs = positive_freqs[filtered_indices]
    filtered_fft_values = [fft[filtered_indices] for fft in positive_fft_values]
    return filtered_freqs,filtered_fft_values

def fft_channels(signals):
    # 确保频率范围有效
    fq_high = min(freq_high, fs // 2)
    
    t = np.arange(0, signals.shape[1] / fs, 1 / fs)
    fft_values = [np.fft.fft(signal) / len(t) for signal in signals]  # 归一化
    fft_freq = np.fft.fftfreq(len(t), 1 / fs)
    
    n_half = len(fft_freq) // 2
    positive_freqs = fft_freq[:n_half]
    positive_fft_values = [np.abs(fft[:n_half]) for fft in fft_values]
    
    filtered_indices = (positive_freqs >= freq_low) & (positive_freqs <= fq_high)
    return positive_freqs[filtered_indices], [fft[filtered_indices] for fft in positive_fft_values]

# 计算功率谱密度
#power_spectrum_values = [fft**2 / (len(t) * fs) for fft in positive_fft_values]
#filtered_power_spectrum = [psd[filtered_indices] for psd in power_spectrum_values]
#print(fft_values)

def trials_channels_PSD(state,freq_low,freq_high):
    plt.figure(figsize=(10, 6))
    trails = treadmill[treadmill['run_or_stop'] == state]  # 筛选run trails
    for i in range(0,len(trails['run_or_stop'])):
        #读取each trial的LFP
        marker_start = int(trails['time_interval_left_end'].iloc[i]*fs)
        marker_end = int(trails['time_interval_right_end'].iloc[i]*fs)
        trail_LFP = LFP[:, marker_start:marker_end]
        #对每个trial LFP进行PSD
        filtered_freqs,filtered_fft_values,filtered_power_spectrum = fft_channels(trail_LFP,freq_low,freq_high)
        #画每个channel的PSD，不同颜色不同channel
        colors = plt.cm.viridis(np.linspace(0, 1, len(filtered_power_spectrum)))  # 颜色循环
        for j, (psd, color) in enumerate(zip(filtered_power_spectrum, colors)):
            plt.plot(filtered_freqs, psd, label=f'Signal {j+1}', color=color)  

    if state == 0: state_name = 'stop'
    else: state_name = 'run'
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.title(f"/Power Spectrum of {region_name}_{state_name}_Signals.png")
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.05), ncol=2, fontsize='small')
    plt.grid()
    plt.savefig(savepath_ch+f"/Power Spectrum of {region_name}_{state_name}_Signals.png")

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

def fq_heatmap(data, state, trial, title_suffix=""):
    n_channels, n_samples = data.shape

    # 计算正频率
    freqs = np.fft.rfftfreq(n_samples, 1/fs)
    freq_mask = (freqs >= freq_low) & (freqs <= freq_high)
    freqs = freqs[freq_mask]
    
    # 添加汉宁窗
    window = np.hanning(n_samples)

    # 逐通道处理
    all_spectra = []
    for i in range(n_channels):
        fft_result = np.fft.rfft(data[i] * window)  # 加窗处理
        spectrum = np.abs(fft_result[freq_mask])   # 或计算功率谱: np.abs(fft_result[freq_mask])​**​2
        all_spectra.append(spectrum)
    all_spectra = np.array(all_spectra)

    # 绘图部分保持不变，调整标题
    plt.title(f'Multi-channel Spectrum Heatmap ({freq_low}-{freq_high}Hz){title_suffix}', fontsize=14, pad=20)
    # 生成坐标网格
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

def fq_spectrum_cupy(data, state, trial, title_suffix=""):
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
        
        # 频率筛选与标准化 --------------------------------------------
        magnitude = cp.abs(Zxx)
        freq_mask = (f >= freq_low) & (f <= freq_high)
        f_filtered = f[freq_mask]
        magnitude = magnitude[freq_mask, :]  # 保持二维结构
        
        # 行方向Z-score标准化
        mean = cp.mean(magnitude, axis=1, keepdims=True)
        std = cp.std(magnitude, axis=1, keepdims=True)
        cp.maximum(std, 1e-6, out=std)
        z_scores = (magnitude - mean) / std
        
        # 数据转换回CPU ----------------------------------------------
        t_cpu = cp.asnumpy(t)
        f_cpu = cp.asnumpy(f_filtered)
        z_cpu = cp.asnumpy(z_scores)
        
        # 可视化设置 -------------------------------------------------
        plt.figure(figsize=(15, 4))
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

def main():
    plt.figure(figsize=(14, 8))

    trail_LFP = LFP[:, 800*fs:980*fs]
    #analyze_all_bands()
    stft_spectrum_cupy(trail_LFP, '800-980')
    #fq_spectrum_cupy(LFP, 'whole', 0)
    #fq_heatmap_cupy(LFP, 'whole', 0)
    '''
    for i in range(0,len(marker['run_or_stop'])):
        start = int(marker['time_interval_left_end'].iloc[i]*fs)   #乘以采样率，转换为采样点
        end = int(marker['time_interval_right_end'].iloc[i]*fs)
        state = marker['run_or_stop'].iloc[i]
        if state == 0: state_name = 'stop'
        else: state_name = 'run'
        #读取each trial的LFP
        trail_LFP = LFP[:, start:end]
        #analyze_all_bands()
        fq_spectrum_cupy(trail_LFP, state_name, i)
        fq_heatmap_cupy(trail_LFP, state_name, i)
    '''
main()

#filtered_data = enhanced_notch_filter(trail_LFP)  # 应用增强型陷波滤波器
#freqs,values = fft_channels(trail_LFP)
## trails
# 当前脑区，所有channel的信号、频谱图（峰值图、热图）、主频率、channel之间的相位差矩阵，区分运动静止
#FFT_freq_phase(run_LFP[a],f'run_{a}',savepath_1,savepath_2)
# 当前脑区，各个channel，频谱峰值trail叠加图，区分运动静止
#overlay_trial_freq_spetrum(2,500) 
# 当前脑区，run trials和stop trials两张图，不同颜色不同channel
#trials_channels_FFT(0,freq_low,freq_high)
#trials_channels_FFT(0,freq_low,freq_high)
#trials_channels_PSD(1,freq_low,freq_high)
#trials_channels_PSD(1,freq_low,freq_high)
## 全时间尺度
# 当前脑区，所有channel的信号、频谱图（峰值图、热图）、主频率、channel之间的相位差矩阵，区分运动静止
#FFT_freq_phase(LFP,f'all_time',savepath_9,savepath_10)
# 当前脑区，各个trail，频谱峰值channel叠加图
#alltime_fft = fft(LFP)
#overlay_ch_freq_spetrum(LFP,f'all_time',savepath_11) 