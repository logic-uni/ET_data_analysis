"""
# coding: utf-8
@author: Yuhao Zhang
last updated: 04/19/2025
data from: Xinchao Chen
"""
import neo
import quantities as pq
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.pyplot import *
from ast import literal_eval
from elephant.conversion import BinnedSpikeTrain
import os
import warnings
np.set_printoptions(threshold=np.inf)
np.seterr(divide='ignore',invalid='ignore')

fr_bin = 1  # unit: ms
fs = 1000 / fr_bin  # fr sample rate = 1000ms / bin size 如果bin是10ms，则采样率为100hz，根据香农采样定理，FFT时候会自动把最大频率确定在100hz以内
freq_low, freq_high = 1, 25
neuron_id = 1

### ------------------ Load Data-------------------

## --------- NP2 ----------
mice_name = '20250310_VN_tremor'
sorting_path = f"/data1/zhangyuhao/xinchao_data/NP2/{mice_name}/Sorted/"
neurons = pd.read_csv(f"/data1/zhangyuhao/xinchao_data/NP2/{mice_name}/filtered_quality_metrics.csv")  # QC neurons
#neurons = pd.read_csv(sorting_path + '/cluster_group.tsv', sep='\t')   # all neurons
marker = pd.read_csv(f"/data1/zhangyuhao/xinchao_data/NP2/{mice_name}/Marker/static_motion_segement.csv")
save_path = f"/home/zhangyuhao/Desktop/Result/ET/ACF_FFT/NP2/{mice_name}/"

'''
## --------- NP1 ----------
mice_name = '20230602_Syt2_conditional_tremor_mice2_lateral'
region_name = 'Superior vestibular nucleus'
# ------ Easysort ------  With QC
#  NEED CHANGE
mapping_file = 'unit_ch_dep_region_QC_isi_violations_ratio_pass_rate_60.17316017316017%.csv'
#  NO NEED CHANGE
sorting_path = f"/data1/zhangyuhao/xinchao_data/NP1/{mice_name}/Sorted/Easysort/results_KS2/sorter_output/"
save_path = f"/home/zhangyuhao/Desktop/Result/ET/Fr_FFT/NP1/Easysort/{mice_name}/"
neurons = pd.read_csv(f"/data1/zhangyuhao/xinchao_data/NP1/{mice_name}/Sorted/Easysort/mapping/{mapping_file}")  # different sorting have different nueron id
# ------ Xinchao_sort ------  Without QC
#  NO NEED CHANGE 
#sorting_path = f"/data1/zhangyuhao/xinchao_data/NP1/{mice_name}/Sorted/Xinchao_sort/"
#save_path = "/home/zhangyuhao/Desktop/Result/ET/Fr_FFT/NP1/Xinchao_sort/{mice_name}/"
#neurons = pd.read_csv(sorting_path + '/neuron_id_region_firingrate.csv')  # different sorting have different nueron id
#marker = pd.read_csv(f"/data1/zhangyuhao/xinchao_data/NP1/{mice_name}/Marker/treadmill_move_stop_velocity_segm_trial.csv",index_col=0)
marker = pd.read_csv(f"/data1/zhangyuhao/xinchao_data/NP1/{mice_name}/Marker/treadmill_move_stop_velocity.csv",index_col=0)
'''

# ---------- Load electrophysiology data ----------
sample_rate = 30000 #spikeGLX neuropixel sample rate
identities = np.load(sorting_path + '/spike_clusters.npy') # time series: unit id of each spike
times = np.load(sorting_path + '/spike_times.npy')  # time series: spike time of each spike
print("Test if electrophysiology duration is equal to treadmill duration ...")
print(f"Marker duration: {marker['time_interval_right_end'].iloc[-1]}")

## Change NP1 or NP2
print(f"Electrophysiology duration: {times[-1] / sample_rate}")     # NP2
#print(f"Electrophysiology duration: {(times[-1] / sample_rate)[0]}")  # NP1

### ------------------ Main Program -------------------
def singleneuron_spiketimes(id):
    x = np.where(identities == id)
    y=x[0]
    spike_times=np.zeros(len(y))
    for i in range(0,len(y)):
        z=y[i]
        spike_times[i]=times[z]/sample_rate
    return spike_times

## -------- FFT ---------
def neuron_spiketrain(neuron_id,marker_start,marker_end):
    spike_times = singleneuron_spiketimes(neuron_id)
    spike_times_trail = spike_times[(spike_times > marker_start) & (spike_times < marker_end)]
    spiketrain = neo.SpikeTrain(spike_times_trail,units='sec',t_start=marker_start, t_stop=marker_end)
    fr = BinnedSpikeTrain(spiketrain, bin_size=fr_bin*pq.ms,tolerance=None)  # had been qualified that elephant can generate correct spike counts
    trial_neuron_fr = fr.to_array().astype(int)[0]
    return trial_neuron_fr

def ACF(data,start,end,trial_type,unit_id):
    sample_rate = 1000  # 采样率1000Hz
    max_lag_time = 1    # 最大滞后时间1秒
    frequency_resolution = 0.1  # 频率分辨率0.1Hz

    # 计算对应的样本数和FFT长度
    max_lag_samples = int(max_lag_time * sample_rate)
    n_fft = int(sample_rate / frequency_resolution)  # 10000点FFT
    # 计算自相关函数（ACF）
    acf_full = np.correlate(data, data, mode='full')
    # 提取滞后0到max_lag_samples部分
    acf = acf_full[len(data)-1 : len(data)-1 + max_lag_samples + 1]
    # -------------------- 绘制自相关函数图 --------------------
    time_lags = np.arange(max_lag_samples + 1) / sample_rate  # 将滞后转换为秒
    plt.plot(time_lags, acf, color='blue')
    plt.xlabel("Lag Time (s)", fontsize=12)
    plt.ylabel("Autocorrelation", fontsize=12)
    plt.title("Autocorrelation Function (0 to 1 sec lags)", fontsize=14)
    plt.xlim(0, max_lag_time)  # 限制横轴范围
    plt.ylim(0, 40)  # 新增纵轴范围限制
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{start}_{end}_{trial_type}_Neuron_id_{unit_id}_ACF.png'))
    plt.clf()
    # -------------------- 后续FFT处理（原代码） --------------------
    # 应用FFT并补零到n_fft以提升频率分辨率
    fft_acf = np.fft.fft(acf, n=n_fft)

    # 计算频率轴
    freq = np.fft.fftfreq(n_fft, d=1/sample_rate)
    # 获取非负频率部分
    positive_freq = freq[:n_fft//2]
    magnitude = np.abs(fft_acf[:n_fft//2])  # 取幅度谱

    plt.plot(positive_freq, magnitude)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('FFT of ACF with 0.1 Hz Resolution')
    plt.xlim(freq_low, freq_high)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{start}_{end}_{trial_type}_Neuron_id_{unit_id}_ACF_FFT.png'))
    plt.clf()

def prominance_compute():
    #  Peak detection with 1%‐of‐mean prominence
    prom_thresh = 0.01 * np.mean(spec_smooth)
    peaks, props = find_peaks(spec_smooth, prominence=prom_thresh)

    #  Collect peak info
    peak_freqs       = freqs[peaks]
    peak_heights     = spec_smooth[peaks]
    peak_prominences = props['prominences']

    prominence_array = np.zeros_like(freqs)
    for freq, prom in zip(peak_freqs, peak_prominences):
        idx = np.argmin(np.abs(freqs - freq))  # Find the closest index in freqs
        prominence_array[idx] = prom
    print(prominence_array)
    return prominence_array


def prominance():
            promi = prominan(vs)
            #promi = promi * promi * promi
            current_sum += promi  # Accumulate smoothed vector strength
            #current_sum_subt = current_sum - np.min(current_sum)  # normalize for ploting heatmap
            current_sum_history.append(current_sum.copy())  # 保存当前状态
    
    # 转换为二维数组（神经元数 x 频率）
    data = np.array(current_sum_history)

    # 创建热图
    plt.figure(figsize=(14, 8))
    plt.imshow(
        data.T,  # 转置，使频率作为纵轴
        aspect='auto',
        cmap='viridis',
        origin='lower',
        extent=[0.5, len(data)+0.5, freqs[0], freqs[-1]],  # 横轴范围
        interpolation='nearest'  # 避免插值
    )

    # 设置坐标轴和标签
    plt.colorbar(label='Cumulative Vector Strength')
    plt.xlabel('Number of Accumulated Neurons')
    plt.ylabel('Frequency (Hz)')

    # 调整横轴刻度为整数
    plt.xticks(np.arange(1, len(data)+1, step=max(1, len(data)//10)))

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'heatmap_accumulation.png'), dpi=300)
    plt.close()

    # 获取最后一个累积状态的数据
    last_slice = current_sum_history[-1]  # 最后一行数据

    # 创建二维曲线图
    plt.figure(figsize=(12, 5))
    plt.plot(freqs, last_slice, 
            color='#2E86C1',  # 使用与热图协调的颜色
            lw=1.5, 
            alpha=0.9)

    # 增强可视化效果
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Cumulative Vector Strength', fontsize=12)
    plt.title('Final Accumulated Vector Strength', fontsize=14, pad=15)

    # 如果频率范围较大，可以设置为对数坐标
    if np.max(freqs) / np.min(freqs) > 100:
        plt.xscale('log')
        plt.xticks([1, 10, 100, 1000], ['1', '10', '100', '1000'])

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'final_accumulation_curve.png'), dpi=300)
    plt.close()

def enumarate_neurons(start,end,trial_type):
    # NP2
    for index, row in neurons.iterrows():
        unit_id = row['cluster_id']
        unit_id = int(unit_id)
        spiketrain = neuron_spiketrain(unit_id,start,end)
        ACF(spiketrain,start,end,trial_type,unit_id)
    '''
    # NP1
    result = neurons.groupby('region')['cluster_id'].apply(list).reset_index(name='cluster_ids')
    for index, row in result.iterrows():
        region = row['region']
        popu_ids = row['cluster_ids']
        if region == region_name:
            for j in range(len(popu_ids)): #第j个neuron
                spiketrain = neuron_spikecounts(popu_ids[j],start,end)
                ACF(spiketrain)
    '''

def enumarate_trials():
    plt.figure(figsize=(10, 6))
    for index, row in marker.iterrows():
        start = row['time_interval_left_end']
        end = row['time_interval_right_end']
        if end - start < 2:
            continue
        status = row['run_or_stop']
        if status == 0:
            trial_type = 'static'
        else:
            trial_type = 'run'
        enumarate_neurons(start,end,trial_type)

enumarate_trials()