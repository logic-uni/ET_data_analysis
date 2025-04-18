import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt
from scipy.fft import fft, fftfreq

# ------- NEED CHANGE -------
mice_name = '20230623_Syt2_conditional_tremor_mice4'
LFP_file = '/Medial vestibular nucleus_202300622_Syt2_512_2_Day18_P79_g0_t0.exported.imec0.lf.csv'

# ------- NO NEED CHANGE -------
# 1. Load data
lfp_path = f"/data1/zhangyuhao/xinchao_data/NP1/{mice_name}/LFP/{LFP_file}"
treadmill = pd.read_csv(f"/data1/zhangyuhao/xinchao_data/NP1/{mice_name}/Marker/treadmill_move_stop_velocity.csv",index_col=0)
save_path = "/home/zhangyuhao/Desktop/Result/ET/LFP_phase_kura/"
fs = 2500  # Hz
lfp_signals = pd.read_csv(lfp_path, header=None).values
lfp_signals = lfp_signals.T
num_channels, num_samples = lfp_signals.shape
print(f'LFP data loaded with {num_channels} channels and {num_samples} samples.')

# 2. 预处理：滤波与相位提取
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """带通滤波器"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, data)
    return filtered

# 3. 从LFP数据中提取固有频率
def extract_natural_frequencies(signals, fs):
    """通过FFT提取主导频率，仅考虑1.5Hz到50Hz之间的频率"""
    freqs = []
    for sig in signals:
        sig = sig - np.mean(sig) 
        n = len(sig)
        y = fft(sig)
        xf = fftfreq(n, 1/fs)[:n//2]
        # 限制频率范围在2Hz到50Hz之间
        valid_idx = (xf >= 2) & (xf <= 50)
        y = np.abs(y[:n//2])
        y[~valid_idx] = 0  # 将无效频率的幅值置为0
        idx = np.argmax(y)
        freqs.append(xf[idx])
    return np.array(freqs)

# 4. Kuramoto模型仿真
def kuramoto_ode(theta, t, omega, K, coupling_matrix):
    """Kuramoto模型的微分方程"""
    n = len(theta)
    dtheta_dt = np.zeros(n)
    for i in range(n):
        coupling = 0
        for j in range(n):
            coupling += coupling_matrix[i,j] * np.sin(theta[j] - theta[i])
        dtheta_dt[i] = omega[i] + (K / n) * coupling
    return dtheta_dt

# 使用欧拉方法数值积分
def simulate_kuramoto(initial_phases, omega, K, coupling_matrix, t, fs):
    dt = 1/fs
    theta = initial_phases.copy()
    theta_history = [theta.copy()]
    for _ in range(1, len(t)):
        dtheta = kuramoto_ode(theta, None, omega, K, coupling_matrix)
        theta += dt * dtheta
        theta_history.append(theta.copy())
    return np.array(theta_history)

# 5. 同步性分析（计算PLV）
def calculate_plv(phases):
    """计算所有振子对的相位锁定值PLV"""
    n = phases.shape[0]
    plv_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                phase_diff = phases[i] - phases[j]
                plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                plv_matrix[i,j] = plv
    return plv_matrix

def workflow(data, trial_num, state_name):
    num_oscillators = data.shape[0]
    duration = data.shape[1] / fs
    t = np.arange(0, duration, 1/fs)
    # 对每个通道进行带通滤波
    filtered_signals = np.array([bandpass_filter(sig, 1.5, 50, fs) for sig in data])
    # 提取相位（希尔伯特变换）
    phases = np.array([np.angle(hilbert(sig)) for sig in filtered_signals])
    # 提取固有频率
    frequencies = extract_natural_frequencies(filtered_signals, fs)
    print(frequencies)
    omega = 2 * np.pi * frequencies

    # 参数设置
    K = 1.0  # 耦合强度
    coupling_matrix = np.ones((num_oscillators, num_oscillators))  # 全连接网络

    # 初始相位（从真实数据中提取）
    initial_phases = phases[:, 0]

    # 运行仿真
    theta_history = simulate_kuramoto(initial_phases, omega, K, coupling_matrix, t, fs)
    
    # 计算仿真后的PLV
    plv_simulated = calculate_plv(theta_history.T)

    # 计算真实LFP数据的PLV
    plv_real = calculate_plv(phases)

    # 可视化
    plt.figure(figsize=(15, 10))
    # 原始LFP信号
    plt.subplot(3, 2, 1)
    for i in range(num_oscillators):
        plt.plot(t, lfp_signals[i] + 3*i, label=f'Channel {i+1}')
    plt.title("Raw LFP Signals")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    # 滤波后的信号
    plt.subplot(3, 2, 2)
    for i in range(num_oscillators):
        plt.plot(t, filtered_signals[i] + 3*i)
    plt.title("Filtered Signals (4-8Hz)")

    # 相位动态
    plt.subplot(3, 2, 3)
    for i in range(num_oscillators):
        plt.plot(t, theta_history[:,i], label=f'Oscillator {i+1}')
    plt.title("Phase Dynamics (Simulated)")
    plt.xlabel("Time (s)")
    plt.ylabel("Phase (rad)")

    # PLV矩阵（真实数据）
    plt.subplot(3, 2, 4)
    plt.imshow(plv_real, vmin=0, vmax=1, cmap='hot')
    plt.colorbar()
    plt.title("PLV Matrix (Real Data)")

    # PLV矩阵（仿真结果）
    plt.subplot(3, 2, 5)
    plt.imshow(plv_simulated, vmin=0, vmax=1, cmap='hot')
    plt.colorbar()
    plt.title("PLV Matrix (Simulated)")

    plt.tight_layout()
    plt.savefig(save_path+f"/trail{trial_num}_{state_name}.png")

def main():
    for trial_num in range(0,len(treadmill['run_or_stop'])):
        start = int(treadmill['time_interval_left_end'].iloc[trial_num]*fs)   #乘以采样率，转换为采样点
        end = int(treadmill['time_interval_right_end'].iloc[trial_num]*fs)
        state = treadmill['run_or_stop'].iloc[trial_num]
        if state == 0: state_name = 'stop'
        else: state_name = 'run'
        #读取each trial的LFP
        trail_LFP = lfp_signals[:, start:end]
        print(trail_LFP.shape)
        workflow(trail_LFP, trial_num, state_name)

main()