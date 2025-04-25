
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

'''
def analyse_freq_and_amp(x: np.ndarray, y: np.ndarray):
    """
    分析不同各频率谐波的幅值
    :param x: （几乎）等间隔的升序序列
    :param y:
    :return: 频率序列，幅值序列
    """
    n = len(x)
    sample_freq = (n - 1) / (x[-1] - x[0])  # 信号的采样频率
    freqs = fftfreq(n, 1. / sample_freq)[:n // 2]
    amplitudes = 2. / n * np.abs(fft(y)[:n // 2])
    return freqs, amplitudes
        

y=LFP[0,:]
#signal2=LFP[1,:]

freqs, amps = analyse_freq_and_amp(t, y)
plt.figure()
plt.plot(freqs, amps)
plt.xlabel("Freq./Hz")
plt.ylabel("Amplitude")
plt.show()


# 计算信号的傅里叶变换
fft_result1 = np.fft.fft(signal1)
fft_result2 = np.fft.fft(signal2)

# 计算相位
phase1 = np.angle(fft_result1)
phase2 = np.angle(fft_result2)

# 计算相位差
phase_difference = phase2 - phase1

# 计算频率向量
freqs = np.fft.fftfreq(len(signal1), 1/sampling_rate)

# 仅选择正频率部分
positive_freqs = freqs[:len(freqs)//2]
positive_phase_difference = phase_difference[:len(phase_difference)//2]

# 绘制信号和相位差
plt.figure(figsize=(12, 8))

# 原始信号1
plt.subplot(3, 1, 1)
plt.plot(t, signal1, label="Signal 1")
plt.plot(t, signal2, label="Signal 2")
plt.title("Original Signals")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()

# 相位谱1
plt.subplot(3, 1, 2)
plt.plot(positive_freqs, phase1[:len(positive_freqs)], label="Phase of Signal 1")
plt.plot(positive_freqs, phase2[:len(positive_freqs)], label="Phase of Signal 2")
plt.title("Phase Spectrum of Signals")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Phase [radians]")
plt.legend()

# 相位差
plt.subplot(3, 1, 3)
plt.plot(positive_freqs, positive_phase_difference)
plt.title("Phase Difference Spectrum")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Phase Difference [radians]")

plt.tight_layout()
plt.show()

# CSD

s1=data[i]
s2=data[i+1]
dt=3
fs = 1/dt

# 计算CSD
f, csd = signal.csd(s1, s2, fs=fs, nperseg=len(t))

# 绘制CSD
plt.figure()
plt.title('Cross Spectral Density')
plt.plot(f, np.abs(csd), label='CSD Amplitude')
plt.xlabel('Frequency (Hz)')
plt.ylabel('CSD Amplitude')
plt.grid()
plt.legend()

# 计算相位差
phase_diff = np.angle(csd)

# 绘制相位差
plt.figure()
plt.title('Phase Difference')
plt.plot(f, phase_diff, label='Phase Difference')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase Difference (radians)')
plt.grid()
plt.legend()

plt.show()
'''