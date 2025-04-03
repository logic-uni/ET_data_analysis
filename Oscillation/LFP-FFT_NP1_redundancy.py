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