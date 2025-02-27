"""
# coding: utf-8
@author: Yuhao Zhang
last updated: 09/18/2024
data from: Xinchao Chen
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import welch
from scipy.signal import stft
import pywt
from scipy.signal import hilbert, chirp
from mpl_toolkits.mplot3d import Axes3D
from PyEMD import EMD
import matplotlib.pyplot as plt
import numpy as np
from PyEMD import EEMD,EMD,Visualisation
from vmdpy import VMD

data_path = r'E:\xinchao\cage1_training_nonmotorized\PV-SYT2-training.txt'
fig_save_path = r'C:\Users\zyh20\Desktop\ET_data analysis\loadcell_FFT\run_stop\FFT\PV-SYT2-training'

df = pd.read_csv(data_path, sep=',', header=None, names=['counter', 'step', 'loadcell', 'reward'])
df = df.apply(pd.to_numeric, errors='coerce')
sample_rate = 1000  #loadcell采样率1000

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
    print(fft_values)

    return filtered_freqs,filtered_fft_values,filtered_power_spectrum

def freq_heat_map(times,frequencies,Sxx,trail_num):
    # 频谱热图
    #plt.figure(figsize=(10, 6))
    plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')
    plt.colorbar(label='Intensity [dB]')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.title(f'{trail_num}__Spectrogram')
    plt.savefig(fig_save_path+f"/{trail_num}_Spectrogram.png")
    plt.clf()

def hilbert_ana(t,x,trail_num):  #希尔伯特变换在信号边缘可能会产生伪频率，需要谨慎解释边缘区域的结果。
    # 应用希尔伯特变换
    analytic_signal = hilbert(x)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * sample_rate

    # 绘制原始信号
    plt.subplot(2, 1, 1)
    plt.plot(t, x)
    plt.title('Original Signal')
    plt.xlabel('Time [sec]')
    plt.ylabel('Amplitude')

    # 绘制瞬时频率
    plt.subplot(2, 1, 2)
    plt.plot(t[1:], instantaneous_frequency)
    plt.title(f'{trail_num}_Instantaneous Frequency')
    plt.xlabel('Time [sec]')
    plt.ylabel('Frequency [Hz]')
    plt.ylim(0, 20)  # 限制频率范围
    plt.savefig(fig_save_path+f"/{trail_num}_Instantaneous Frequency.png")
    plt.clf()

def stft_ana(t,sig,trail_num):
    # 计算短时傅里叶变换
    f, t, Zxx = stft(sig, fs=sample_rate, nperseg=256)
    # 只保留 2 Hz 到 20 Hz 的频率部分
    freq_range = (f >= 2) & (f <= 15)
    f_filtered = f[freq_range]
    Zxx_filtered = Zxx[freq_range, :]

    # 可视化 STFT 结果，仅显示 2-20 Hz
    plt.pcolormesh(t, f_filtered, np.abs(Zxx_filtered), shading='gouraud')
    plt.title('STFT Magnitude (2-20 Hz)')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='Magnitude')
    plt.savefig(fig_save_path+f"/STFT_{trail_num}.png")
    plt.clf()

def PSD_ana(sig,fq_thre_low,fq_thre_high):
    # 计算PSD使用Welch方法
    frequencies, psd = welch(sig, fs=sample_rate, nperseg=1024)

    # 只保留1到15 Hz范围内的数据
    mask = (frequencies >= fq_thre_low) & (frequencies <= fq_thre_high)
    frequencies_filtered = frequencies[mask]
    psd_filtered = psd[mask]

    # 创建散点图
    plt.scatter(frequencies_filtered, psd_filtered, c='blue', marker='o', s=50)

    # 设置轴标签和标题
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (V²/Hz)')
    plt.title('Power Spectral Density (1-15 Hz)')
    plt.grid(True)

def CWT_ana(x):
    # 定义小波
    wavelet = 'cmor'  # 复Morlet小波

    # 选择尺度范围
    scales = np.arange(1, 128)

    # 进行小波变换
    coefficients, frequencies = pywt.cwt(x, scales, wavelet, sampling_period=1/sample_rate)

    # 可视化小波变换结果
    plt.figure(figsize=(10, 6))
    plt.imshow(np.abs(coefficients), extent=[0, 3, frequencies[-1], frequencies[0]], cmap='jet', aspect='auto')
    plt.colorbar(label='Magnitude')
    plt.title('Wavelet Transform (CWT) Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.ylim(2, 20)  # 限制频率范围在2-50 Hz
    plt.show()

def hilbert_3D(time,signal):
    # 执行Hilbert变换
    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = np.diff(instantaneous_phase) / (2.0*np.pi) * np.diff(time)

    # 三维图 - 解析信号
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(time, signal, amplitude_envelope, label='Analytic Signal')
    ax.set_xlabel('Time (Counter)')
    ax.set_ylabel('Original Signal (Loadcell)')
    ax.set_zlabel('Transformed Signal')
    plt.title('3D Analytic Signal with Projections')
    ax.grid()

    # 投影到XY平面（原信号）
    ax.plot(time, signal, np.zeros_like(signal), label='Original Signal')

    # 投影到XZ平面（Hilbert变换后的信号）
    ax.plot(time, np.zeros_like(signal), amplitude_envelope, label='Transformed Signal')

    # 投影到YZ平面（复平面轨迹）
    ax.plot(np.zeros_like(signal), signal, amplitude_envelope, label='Complex Plane Trajectory')

    plt.legend()
    plt.show()

    # 极坐标图
    plt.figure(figsize=(6, 6))
    plt.polar(instantaneous_phase, amplitude_envelope)
    plt.title('Polar Plot of Analytic Signal')
    plt.show()

    # 瞬时频率图
    plt.figure(figsize=(10, 4))
    plt.plot(time[:-1], instantaneous_frequency)
    plt.xlabel('Time (Counter)')
    plt.ylabel('Instantaneous Frequency')
    plt.title('Instantaneous Frequency')
    plt.grid()
    plt.show()

#分解方法（emd、eemd、vmd）
def decompose_lw(signal,t,method='eemd',K=5,draw=1):
    names=['emd','eemd','vmd']
    idx=names.index(method)
    #emd分解
    if idx==0:
        emd = EMD()
        IMFs= emd.emd(signal)

    #vmd分解
    elif idx==2:
        alpha = 2000       # moderate bandwidth constraint
        tau = 0.            # noise-tolerance (no strict fidelity enforcement)
        DC = 0             # no DC part imposed
        init = 1           # initialize omegas uniformly
        tol = 1e-7
        # Run actual VMD code
        IMFs, _, _ = VMD(signal, alpha, tau, K, DC, init, tol)
        
    #eemd分解
    else:
        eemd = EEMD()
        emd = eemd.EMD
        emd.extrema_detection="parabol"
        IMFs= eemd.eemd(signal,t)
    
    #可视化
    if draw==1:
        plt.figure()
        for i in range(len(IMFs)):
            plt.subplot(len(IMFs),1,i+1)
            plt.plot(t,IMFs[i])
            if i==0:
                plt.rcParams['font.sans-serif']='Times New Roman'
                plt.title('Decomposition Signal',fontsize=14)
            elif i==len(IMFs)-1:
                plt.rcParams['font.sans-serif']='Times New Roman'
                plt.xlabel('Time/s')
        # plt.tight_layout()
    return IMFs       

#希尔波特变换及画时频谱
def hhtlw(IMFs,t,f_range=[0,500],t_range=[0,1],ft_size=[128,128],draw=1):
    fmin,fmax=f_range[0],f_range[1]         #时频图所展示的频率范围
    tmin,tmax=t_range[0],t_range[1]         #时间范围
    fdim,tdim=ft_size[0],ft_size[1]         #时频图的尺寸（分辨率）
    dt=(tmax-tmin)/(tdim-1)
    df=(fmax-fmin)/(fdim-1)
    vis = Visualisation()
    #希尔伯特变化
    c_matrix=np.zeros((fdim,tdim))
    for imf in IMFs:
        imf=np.array([imf])
        #求瞬时频率
        freqs = abs(vis._calc_inst_freq(imf, t, order=False, alpha=None))
        #求瞬时幅值
        amp= abs(hilbert(imf))
        #去掉为1的维度
        freqs=np.squeeze(freqs)
        amp=np.squeeze(amp)
        #转换成矩阵
        temp_matrix=np.zeros((fdim,tdim))
        n_matrix=np.zeros((fdim,tdim))
        for i,j,k in zip(t,freqs,amp):
            if i>=tmin and i<=tmax and j>=fmin and j<=fmax:
                temp_matrix[round((j-fmin)/df)][round((i-tmin)/dt)]+=k
                n_matrix[round((j-fmin)/df)][round((i-tmin)/dt)]+=1
        n_matrix=n_matrix.reshape(-1)
        idx=np.where(n_matrix==0)[0]
        n_matrix[idx]=1
        n_matrix=n_matrix.reshape(fdim,tdim)
        temp_matrix=temp_matrix/n_matrix
        c_matrix+=temp_matrix
    
    t=np.linspace(tmin,tmax,tdim)
    f=np.linspace(fmin,fmax,fdim)
    #可视化
    if draw==1:
        fig,axes=plt.subplots()
        plt.rcParams['font.sans-serif']='Times New Roman'
        plt.contourf(t, f, c_matrix,cmap="jet")
        plt.xlabel('Time/s',fontsize=16)
        plt.ylabel('Frequency/Hz',fontsize=16)
        plt.title('Hilbert spectrum',fontsize=20)
        x_labels=axes.get_xticklabels()
        [label.set_fontname('Times New Roman') for label in x_labels]
        y_labels=axes.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in y_labels]
        # plt.show()
    return t,f,c_matrix

def emd_eemd_vmd():
    # 获取时间和信号列
    t = df['counter'].to_numpy()
    signal = df['loadcell'].to_numpy()
    print(t.shape)
    print(signal.shape)

    #画时域图
    plt.figure()
    plt.plot(t,signal)
    plt.rcParams['font.sans-serif']='Times New Roman'
    plt.xlabel('Time/s',fontsize=16)
    plt.title('Original Signal',fontsize=20) 
    plt.show()

    IMFs=decompose_lw(signal,t,method='emd',K=10)                                    #分解信号
    tt,ff,c_matrix=hhtlw(IMFs,t,f_range=[0,500],t_range=[0,1],ft_size=[128,128]) 

def standard_FFT_PSD(sig,fq_thre_low,fq_thre_high,trail_num):
    # 计算信号的傅里叶变换
    fft_result = np.fft.fft(sig)
    # 计算频率向量
    freq_vector = np.fft.fftfreq(len(sig), 1/sample_rate)
    # 只选择正频率部分
    positive_freqs = freq_vector[:len(freq_vector)//2]
    positive_fft_result = np.abs(fft_result[:len(fft_result)//2])
    # 频率范围
    filtered_indices = (positive_freqs >= fq_thre_low) & (positive_freqs < fq_thre_high)
    positive_freqs = positive_freqs[filtered_indices]
    positive_fft_result = positive_fft_result[filtered_indices]
    return positive_freqs,positive_fft_result
    # 功率谱密度
    #power_spectrum_values = positive_fft_result**2 / (len(t) * sample_rate)  
    #filtered_power_spectrum = power_spectrum_values[filtered_indices]
    #plt.scatter(positive_freqs, positive_fft_result)

    '''
    # 频谱峰值图
    '''
    '''
    plt.plot(positive_freqs, positive_fft_result)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.title(f'{trail_num}_Frequency Spectrum')
    plt.grid()
    plt.savefig(fig_save_path+f"/{trail_num}_Frequency Spectrum.png")
    plt.clf()
    '''

def main_function():
    fq_thre_low = 2
    fq_thre_high = 15
    changes = df['reward'].diff() == 1
    changes.iloc[0] = False  # 处理边界情况，确保第一个条目不被错误地标记为变化
    df_changes = df[changes]
    trail_endtimes = df_changes['counter'].values
    trail_endtimes = trail_endtimes[1:]
    trail_num = 0
    amplitude = np.array([])
    frequency = np.array([])

    plt.figure(figsize=(10, 6))
    temp = np.array([])
    for end in trail_endtimes:
        trial = df.loc[(df['counter'] > (end - 4)) & (df['counter'] < (end - 0))]
        sig = trial['loadcell'].values
        t = trial['counter'].values
        #hilbert_ana(t,sig,trail_num)
        #stft_ana(t,sig,trail_num)
        
        x,y = standard_FFT_PSD(sig,fq_thre_low,fq_thre_high,trail_num)
        if len(y) > 50:
            y = y[:-1]
        
        if trail_num == 0:
            temp = y
        else:
            temp = np.vstack([temp, y])
        
        trail_num = trail_num + 1
        
    average_array = np.mean(temp, axis=0)
    x = x[:-1]
    plt.plot(x, average_array)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.title(f'average_Frequency Spectrum')
    plt.grid()
    plt.savefig(fig_save_path+f"/average_Frequency Spectrum.png")
    plt.clf()

    '''
    if trail_num == 0:
        amplitude = positive_fft_result
    else:
        amplitude = np.vstack((amplitude, positive_fft_result))
    
    if trail_num == 0:
        frequency = positive_freqs
    else:
        frequency = np.vstack((frequency, positive_freqs))
    
trials = np.arange(trail_num)  # trial编号

'''

main_function()