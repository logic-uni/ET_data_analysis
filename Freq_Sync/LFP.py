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
import seaborn as sns
np.set_printoptions(threshold=np.inf)

# ------- NEED CHANGE -------
mice_name = '20230113_littermate'
LFP_file = '/Spinal vestibular nucleus_20230113_338_3_liteermate_female_stable_set_g0_t0.exported.imec0.lf.csv'

# ------- NO NEED CHANGE -------
sample_rate = 2500 # 2,499.985345140265   #spikeGLX neuropixel LFP sample rate
treadmill = pd.read_csv(rf'E:\xinchao\Data\useful_data\{mice_name}\Marker\treadmill_move_stop_velocity.csv',index_col=0)
LFP = pd.read_csv(rf'E:\xinchao\Data\useful_data\{mice_name}\LFP' + LFP_file)
region_name = LFP_file[1:]
region_name = region_name.split('_')[0]

fig_path = rf'C:\Users\zyh20\Desktop\ET_data analysis\LFP_frequence\{mice_name}\{region_name}'
savepath_1 = fig_path + r'\trials\raw_tra&fq_spec\run'
savepath_2 = fig_path + r'\trials\mainfq_phas_dif\run'
savepath_3 = fig_path + r'\trials\raw_trace&fq_spec\stop'
savepath_4 = fig_path + r'\trials\mainfq_phas_dif\stop'
savepath_ch = fig_path + r'\trials\fq_spec_ch_overlay'
savepath_trial = fig_path + r'\trials\fq_spec_trial_overlay'
savepath_ch_trail = fig_path + r'\trials\fq_spec_ch_trail_overlay'

savepath_9 = fig_path + r'\all_time\raw_trace&freq_spectrum'
savepath_10 = fig_path + r'\all_time\main_freq_phase_diff'
savepath_11 = fig_path + r'\all_time\freq_spectrum_overlay_ch'

LFP = LFP.T
LFP = LFP.to_numpy()
LFP = LFP[:,800*sample_rate:2200*sample_rate]  # 截取800-2200s
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

def fft_channels(signals,fq_thre_low,fq_thre_high):
    t=np.arange(0,signals.shape[1]/sample_rate,1/sample_rate)
    # 计算所有信号的傅里叶变换
    fft_values = [np.fft.fft(signal) for signal in signals]
    fft_freq = np.fft.fftfreq(len(t), 1/sample_rate)
    # 只考虑正频率部分
    positive_freqs = fft_freq[:len(fft_freq)//2]
    positive_fft_values = [np.abs(fft[:len(fft)//2]) for fft in fft_values]
    # 计算功率谱密度
    power_spectrum_values = [fft**2 / (len(t) * sample_rate) for fft in positive_fft_values]
    # 按范围过滤频率
    filtered_indices = (positive_freqs >= fq_thre_low) & (positive_freqs < fq_thre_high)
    filtered_freqs = positive_freqs[filtered_indices]
    filtered_fft_values = [fft[filtered_indices] for fft in positive_fft_values]
    filtered_power_spectrum = [psd[filtered_indices] for psd in power_spectrum_values]
    #print(fft_values)
    return filtered_freqs,filtered_fft_values,filtered_power_spectrum

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

def trials_channels_FFT(state,freq_low,freq_high):
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
        colors = plt.cm.viridis(np.linspace(0, 1, len(filtered_fft_values)))  # 颜色循环
        for j, (fft_val, color) in enumerate(zip(filtered_fft_values, colors)):
            plt.plot(filtered_freqs, fft_val, label=f'Signal {j+1}', color=color)

    if state == 0: state_name = 'stop'
    else: state_name = 'run'
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.title(f'Frequency Spectrum of {region_name}_{state_name}_Signals')
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.05), ncol=2, fontsize='small')
    plt.grid()
    plt.savefig(savepath_ch+f"/Frequency Spectrum of {region_name}_{state_name}_Signals.png")

## trails
# 当前脑区，所有channel的信号、频谱图（峰值图、热图）、主频率、channel之间的相位差矩阵，区分运动静止
#FFT_freq_phase(run_LFP[a],f'run_{a}',savepath_1,savepath_2)
# 当前脑区，各个channel，频谱峰值trail叠加图，区分运动静止
#overlay_trial_freq_spetrum(2,500) 
# 当前脑区，run trials和stop trials两张图，不同颜色不同channel
freq_low = 2
freq_high = 30
trials_channels_FFT(0,freq_low,freq_high)
trials_channels_FFT(0,freq_low,freq_high)
trials_channels_PSD(1,freq_low,freq_high)
trials_channels_PSD(1,freq_low,freq_high)

## 全时间尺度
# 当前脑区，所有channel的信号、频谱图（峰值图、热图）、主频率、channel之间的相位差矩阵，区分运动静止
#FFT_freq_phase(LFP,f'all_time',savepath_9,savepath_10)
# 当前脑区，各个trail，频谱峰值channel叠加图
#alltime_fft = fft(LFP)
#overlay_ch_freq_spetrum(LFP,f'all_time',savepath_11) 