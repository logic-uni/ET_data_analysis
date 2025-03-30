"""
# coding: utf-8
@author: Yuhao Zhang
last updated: 03/01/2025
data from: Xinchao Chen
"""
from scipy.fft import fft, fftfreq  
from scipy import signal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy.signal import spectrogram
from matplotlib import cm
np.set_printoptions(threshold=np.inf)

# ------- NEED CHANGE -------
####这个千万别忘了开或者关LFP截断
mice_name = '20230113_littermate'
LFP_file = '/Spinal vestibular nucleus_20230113_338_3_liteermate_female_stable_set_g0_t0.exported.imec0.lf.csv'
freq_low = 2
freq_high = 500

# ------- NO NEED CHANGE -------
sample_rate = 2500 # 2,499.985345140265   #spikeGLX neuropixel LFP sample rate
treadmill = pd.read_csv(rf'E:\xinchao\Data\useful_data\NP1\{mice_name}\Marker\treadmill_move_stop_velocity.csv',index_col=0)
LFP = pd.read_csv(rf'E:\xinchao\Data\useful_data\NP1\{mice_name}\LFP' + LFP_file)
region_name = LFP_file[1:]
region_name = region_name.split('_')[0]
save_path = rf'C:\Users\zyh20\Desktop\Research\01_ET_data_analysis\Research\LFP_FFT\NP1\Xinchao_sort\{mice_name}'  

LFP = LFP.T
LFP = LFP.to_numpy()
LFP = LFP[:,800*sample_rate:2200*sample_rate] 
# 对于littermate，请截取800-2200s
# 对于20230604_Syt2_conditional_tremor_mice2_medial请截取 0-1922s
# 其余的全部即可，无需截取
print("检查treadmill总时长和LFP总时长是否一致")
print("LFP总时长")
print(LFP.shape[1]/sample_rate)
print("跑步机总时长")
print(treadmill['time_interval_right_end'].iloc[-1])
print(LFP.shape)

# ------- FUNCTIONS -------
#注意以下使用的函数 np.fft.fftfreq 中，1/sample_rate表示相邻样本之间的时间间隔,因此sample_rate必须是实际数据真实的采样率

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
            marker_start = int(treadmill['time_interval_left_end'].iloc[i]*sample_rate)
            marker_end = int(treadmill['time_interval_right_end'].iloc[i]*sample_rate)
            ch_trail_LFP = signal[marker_start:marker_end]
            if treadmill['run_or_stop'].iloc[i] == 1:
                color = 'red'
            else:
                color = 'blue'
            # 分析每个信号的频谱
            # 计算信号的傅里叶变换
            fft_result = np.fft.fft(ch_trail_LFP)
            # 计算频率向量
            freq_vector = np.fft.fftfreq(len(ch_trail_LFP), 1/sample_rate)  
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
    t=np.arange(0,data.shape[1]/sample_rate,1/sample_rate)
    plt.figure(figsize=(10, 6))
    # 分析每个信号的频谱
    for signal in data:
        # 计算信号的傅里叶变换
        fft_result = np.fft.fft(signal)
        # 计算频率向量
        freq_vector = np.fft.fftfreq(len(signal), 1/sample_rate)
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
        frequencies, times, Sxx = spectrogram(signal, sample_rate)
        # 频谱热图
        freq_heat_map(times,frequencies,Sxx,state,m,savepath_raw)
        m=m+1
        # 原始信号图
        raw_trace_signal(t,signal,state,m,savepath_raw)
        # 频谱峰值图
        freq_expand(freq_vector,fft_result,state,m,savepath_raw)

    phase_diff_main_freq(data,state,main_phases,main_freqs,savepath_diff)

def fft_channels_previous(signals):
    t = np.arange(0,signals.shape[1]/sample_rate,1/sample_rate)
    # 计算所有信号的傅里叶变换
    fft_values = [np.fft.fft(signal) for signal in signals]
    fft_freq = np.fft.fftfreq(len(t), 1/sample_rate)
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
    fq_high = min(freq_high, sample_rate // 2)
    
    t = np.arange(0, signals.shape[1] / sample_rate, 1 / sample_rate)
    fft_values = [np.fft.fft(signal) / len(t) for signal in signals]  # 归一化
    fft_freq = np.fft.fftfreq(len(t), 1 / sample_rate)
    
    n_half = len(fft_freq) // 2
    positive_freqs = fft_freq[:n_half]
    positive_fft_values = [np.abs(fft[:n_half]) for fft in fft_values]
    
    filtered_indices = (positive_freqs >= freq_low) & (positive_freqs <= fq_high)
    return positive_freqs[filtered_indices], [fft[filtered_indices] for fft in positive_fft_values]

# 计算功率谱密度
#power_spectrum_values = [fft**2 / (len(t) * sample_rate) for fft in positive_fft_values]
#filtered_power_spectrum = [psd[filtered_indices] for psd in power_spectrum_values]
#print(fft_values)

def trials_channels_PSD(state,freq_low,freq_high):
    plt.figure(figsize=(10, 6))
    trails = treadmill[treadmill['run_or_stop'] == state]  # 筛选run trails
    for i in range(0,len(trails['run_or_stop'])):
        #读取each trial的LFP
        marker_start = int(trails['time_interval_left_end'].iloc[i]*sample_rate)
        marker_end = int(trails['time_interval_right_end'].iloc[i]*sample_rate)
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

def enhanced_notch_filter(data, sample_rate=2500, notch_freqs=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500], Q=30, n_stages=3):
    """
    增强型多级陷波滤波器
    参数:
        n_stages: 级联的滤波器级数(增加滤波深度)
    """
    filtered_data = data.copy()
    
    for _ in range(n_stages):  # 多级滤波
        for freq in notch_freqs:
            b, a = signal.iirnotch(freq, Q, sample_rate)
            for i in range(data.shape[0]):
                filtered_data[i] = signal.filtfilt(b, a, filtered_data[i])
    
    return filtered_data

def plot_multi_channel_spectrum(data,state,trial, sample_rate, freq_range, title_suffix=""):
    """
    绘制多通道重叠频谱图（所有通道在同一图中）
    参数:
        freq_range: 要显示的频率范围元组 (low, high)
    """
    freq_low, freq_high = freq_range
    n_samples = data.shape[1]
    freqs = fftfreq(n_samples, 1/sample_rate)
    
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

def analyze_all_bands(data, state, trial, sample_rate=2500):
    """分析所有频段的多通道频谱"""
    # 应用增强型陷波滤波器
    filtered_data = enhanced_notch_filter(data, sample_rate)
    
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
            sample_rate, 
            freq_range=(low, high),
            title_suffix="\n(After Enhanced Notch Filtering)"
        )

def plot_2_50hz_spectrum(data, state, trial, sample_rate=2500, title_suffix=""):
    """
    专门绘制2-50Hz频段的多通道重叠频谱图
    参数:
        data: 输入信号 (n_channels, n_samples)
        sample_rate: 采样率
        title_suffix: 标题后缀
    """
    # 设置频率范围
    freq_low, freq_high = 2, 50
    n_samples = data.shape[1]
    freqs = fftfreq(n_samples, 1/sample_rate)
    
    # 创建频率掩码 (2-50Hz)
    freq_mask = (freqs >= freq_low) & (freqs <= freq_high)
    freqs = freqs[freq_mask]
    n_channels = data.shape[0]
    
    # 创建颜色映射 (使用plasma色图更易区分低频)
    colors = cm.plasma(np.linspace(0, 1, n_channels))
    
    # 计算并绘制每个通道的频谱
    all_spectra = []
    for i in range(n_channels):
        fft_result = fft(data[i])
        spectrum = np.abs(fft_result[freq_mask])
        all_spectra.append(spectrum)
        
        # 绘制曲线（关键通道加粗显示）
        linewidth = 1.5 if i in [0, 25, 49] else 0.8  # 突出显示首、中、末通道
        plt.plot(freqs, spectrum, 
                color=colors[i], 
                alpha=0.7, 
                linewidth=linewidth,
                label=f'Ch{i+1}' if i % 10 == 0 else "")  # 每10个通道显示一个标签

    # 设置图形属性
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.xlim(freq_low, freq_high)
    plt.ylim(0, np.max(all_spectra)*1.1)
    plt.grid(True, linestyle='--', alpha=0.4)
    
    # 添加颜色条表示通道编号
    sm = plt.cm.ScalarMappable(cmap=cm.plasma, norm=plt.Normalize(vmin=1, vmax=n_channels))
    sm.set_array([])
    cbar = plt.colorbar(sm, pad=0.02)
    cbar.set_label('Channel Number', rotation=270, labelpad=15)
    
    # 设置标题和图例
    title = f'Multi-channel Spectrum (2-50Hz){title_suffix}'
    plt.title(title, fontsize=14, pad=20)
    plt.legend(loc='upper right', fontsize=8, ncol=2)
    
    plt.tight_layout()
    plt.savefig(save_path+f"/{region_name}_{state}_trial{trial}.png")
    plt.clf()

    return all_spectra

def plot_2_16hz_spectrum(data,state,trial, sample_rate=2500, title_suffix=""):
    """
    绘制2-16Hz频段的多通道重叠频谱图
    参数:
        data: 输入信号 (n_channels, n_samples)
        sample_rate: 采样率
        title_suffix: 标题后缀
    """
    # 设置频率范围
    freq_low, freq_high = 2, 16
    n_samples = data.shape[1]
    freqs = fftfreq(n_samples, 1/sample_rate)
    
    # 创建频率掩码 (2-16Hz)
    freq_mask = (freqs >= freq_low) & (freqs <= freq_high)
    freqs = freqs[freq_mask]
    n_channels = data.shape[0]
    
    # 创建颜色映射 (使用viridis色图)
    colors = cm.viridis(np.linspace(0, 1, n_channels))
    
    # 创建大尺寸图形
    plt.figure(figsize=(16, 8))
    
    # 计算并绘制每个通道的频谱
    all_spectra = []
    for i in range(n_channels):
        fft_result = fft(data[i])
        spectrum = np.abs(fft_result[freq_mask])
        all_spectra.append(spectrum)
        
        # 绘制曲线（关键通道加粗显示）
        linewidth = 1.5 if i in [0, n_channels//2, n_channels-1] else 0.8
        plt.plot(freqs, spectrum, 
                color=colors[i], 
                alpha=0.7, 
                linewidth=linewidth,
                label=f'Ch{i+1}' if i % 10 == 0 else "")  # 每10个通道显示一个标签
    
    # 标记重要频段 (Theta: 4-8Hz, Alpha: 8-12Hz)
    plt.axvspan(4, 8, color='green', alpha=0.1, label='Theta (4-8Hz)')
    plt.axvspan(8, 12, color='blue', alpha=0.1, label='Alpha (8-12Hz)')
    
    # 设置图形属性
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.xlim(freq_low, freq_high)
    plt.ylim(0, np.max(all_spectra)*1.1)
    plt.grid(True, linestyle='--', alpha=0.4)
    
    # 添加颜色条表示通道编号
    sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(vmin=1, vmax=n_channels))
    sm.set_array([])
    cbar = plt.colorbar(sm, pad=0.02)
    cbar.set_label('Channel Number', rotation=270, labelpad=15)
    
    # 设置标题和图例
    title = f'Multi-channel Spectrum (2-16Hz){title_suffix}'
    plt.title(title, fontsize=14, pad=20)
    plt.legend(loc='upper right', fontsize=8, ncol=3)
    
    plt.tight_layout()
    plt.savefig(save_path+f"/{region_name}_{state}_trial{trial}.png")
    plt.clf()

    return all_spectra


def each_trial_FFT_previous():
    plt.figure(figsize=(12, 6))
    for i in range(0,len(treadmill['run_or_stop'])):
        start = int(treadmill['time_interval_left_end'].iloc[i]*sample_rate)   #乘以采样率，转换为采样点
        end = int(treadmill['time_interval_right_end'].iloc[i]*sample_rate)
        state = treadmill['run_or_stop'].iloc[i]
        if state == 0: state_name = 'stop'
        else: state_name = 'run'
        #读取each trial的LFP
        trail_LFP = LFP[:, start:end]
        analyze_all_bands(trail_LFP,state_name,i)
        '''
        # 直接绘制2-16Hz频谱图（无需滤波）
        print("Plotting 2-16Hz spectrum (no notch filtering needed)...")
        spectra = plot_2_16hz_spectrum(
            trail_LFP, 
            state_name,
            i,
            title_suffix="\n(Theta: 4-8Hz, Alpha: 8-12Hz)"
        )
        
        
        # 应用增强型陷波滤波器
        filtered_data = enhanced_notch_filter(trail_LFP)
        
        # 绘制2-50Hz频谱图
        print("Plotting 2-50Hz spectrum with enhanced notch filtering...")
        spectra = plot_2_50hz_spectrum(
            filtered_data, 
            state_name,
            i,
            title_suffix="\n(After Enhanced Notch Filtering)"
        )
        '''
        '''
        freqs,values = fft_channels(trail_LFP)
        plt.plot(freqs, values)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude')
        plt.title(f'{region_name}_{state_name}_trial{i}')
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.05), ncol=2, fontsize='small')
        plt.grid()
        plt.savefig(save_path+f"/{region_name}_{state_name}_trial{i}.png")
        plt.clf()
        '''

each_trial_FFT_previous()

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