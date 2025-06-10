"""
# coding: utf-8
data from: Xinchao Chen
@author: Yuhao Zhang
last updated: 05/17/2025
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import expon
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks
np.set_printoptions(threshold=np.inf)
np.seterr(divide='ignore',invalid='ignore')

# ------- NEED CHANGE -------
data_path = '/data2/zhangyuhao/xinchao_data/Givenme/1670-2-tremor-Day5-bank_4CVC-FM_g0'
save_path = "/home/zhangyuhao/Desktop/Result/ET/ISI/NP2/givenme/1670-2-tremor-Day5-bank_4CVC-FM_g0"
trial_interval = 40 # unit s
fr_filter = 30          
# ------- NO NEED CHANGE -------
type2_threshold = 0.2
cutofdis_max = 250           # 250ms/None  cutoff_distr=0.25代表截断ISI分布大于0.25s的
cutofdis_min = 0.001
histo_bin_num = 100          # 统计图bin的个数
### electrophysiology
fs = 30000 #spikeGLX neuropixel sample rate
identities = np.load(data_path + '/Sorted/kilosort4/spike_clusters.npy') # time series: unit id of each spike
times = np.load(data_path + '/Sorted/kilosort4/spike_times.npy')  # time series: spike time of each spike
#neurons = pd.read_csv(data_path + "/filtered_quality_metrics.csv")  # QC neurons
elec_dura = times[-1] / fs
print(f"Electrophysiology duration: {elec_dura}")

region = 'CbX'  # 感兴趣的region
'''
## Load neuron id: For single region
neuron_info = pd.read_csv(data_path + "/quality_metrics.csv")
neurons = neuron_info[neuron_info['firing_rate'] > fr_filter]
print(neurons)
popu_ids = neurons['cluster_id'].to_numpy()
'''
## Load neuron id: For across region
neurons = pd.read_csv(data_path+'/Sorted/kilosort4/mapping_artifi.csv') 
#neurons = pd.read_csv(data_path+'/Sorted/kilosort4/mapping_artifi_QC.csv') 
#neurons = neurons[neurons['fr'] > fr_filter]
region_groups = neurons.groupby('region')
region_neuron_ids = {}
for reg, group in region_groups:
    # 提取每组的第一列（cluster_id），去除缺失值
    cluster_ids = group.iloc[:, 0].dropna().astype(int).values
    region_neuron_ids[reg] = cluster_ids

popu_ids = region_neuron_ids[region]  # 获取该region的neuron_ids
print(popu_ids.shape)

# ------- Main Program -------
# get single neuron spike train
def singleneuron_spiketimes(id):
    x = np.where(identities == id)
    y=x[0]
    spike_times=np.empty(len(y))
    for i in range(0,len(y)):
        z=y[i]
        spike_times[i]=times[z]/fs
    return spike_times

# ------- Plot -------
def save_selected_neuron_spike_times(unit_id):
    spike_times = singleneuron_spiketimes(unit_id)
    np.save(save_path + f"/{mice_name}_VN_spike_time_neuron_{unit_id}.npy", spike_times)

def ISI_single_neuron_session(unit_id):
    spike_times = singleneuron_spiketimes(unit_id)
    intervals = np.array([])
    # 如果整个session的spike个数，小于elec_dura电生理时长（s），即每秒钟的spike个数小于1，则不进行统计
    if len(spike_times) > (fr_filter*elec_dura): 
        intervals = np.diff(spike_times) * 1000  # 计算时间间隔  # 转为ms单位
        # 绘制时间间隔的直方图
        if cutoff_distr is not None:
            intervals = intervals[(intervals > 0.001) & (intervals <= cutoff_distr)]
            if len(intervals) != 0:  #截取区间可能导致没有interval
                plt.hist(intervals, bins=histo_bin_num, density=False,alpha=0.6)  #如果只看0.25s间隔以内细致的，画counts更适合
        else:
            plt.hist(intervals, bins=20, density=True, alpha=0.6)

    plt.xlabel('Inter-spike Interval (ms)')
    plt.ylabel('Counts')
    plt.title(f'unit id {unit_id}')
    plt.text(0.95, 0.95, f'firing filter > {fr_filter} spike/s\n\nhisto bin: {histo_bin_num}\n\ndistr cutoff > {cutoff_distr}\n\n', ha='right', va='top', transform=plt.gca().transAxes)
    plt.savefig(save_path+f"/neuron_id_{unit_id}.png",dpi=600,bbox_inches = 'tight')
    plt.clf()

def each_neuron_ISI(units):
    plt.figure(figsize=(10, 6))
    for index, row in units.iterrows():
        unit_id = row['cluster_id']
        ISI_single_neuron_session(unit_id)

# ------- Counting -------
def classify_isi_distribution(intervals, bin_num=100):
    """
    1. 泊松过程只需满足：
       - 单峰分布（允许微小抖动）
       - 主峰位置≤10ms
       - 偏度>0.8（适度右偏）
       - 峰度>2（不严格限制尖峰）
    2. 其他情况判为非泊松
    """
    counts, bins = np.histogram(intervals, bins=bin_num, range=(0, 80))
    peaks, _ = find_peaks(counts, height=np.max(counts)*0.1, prominence=np.max(counts)*0.15)
    
    # 计算统计量
    skewness = skew(intervals)
    kurt = kurtosis(intervals)
    main_peak_pos = bins[peaks[0]] if len(peaks) > 0 else 0
    
    # 宽松泊松条件（满足任意3条即可）
    is_poisson = (
        len(peaks) <= 1 and           
        main_peak_pos <= 10 and       # 主峰位置放宽
        skewness > 0.3 and            # 偏度要求降低
        kurt > 1                      # 峰度要求降低
    )

    return not is_poisson

def ISI_counting(spike_times,unit_id,trialnum=None):
    intervals = np.array([])
    is_non_poisson = None
    intervals = np.diff(spike_times) * 1000  #单位换算
    intervals = intervals[(intervals > cutofdis_min) & (intervals <= cutofdis_max)]
    if len(intervals) > 0:
        # 分类判断
        is_non_poisson = classify_isi_distribution(intervals)
        plt.hist(intervals, bins=histo_bin_num,range = (cutofdis_min,cutofdis_max), alpha=0.6)
        stats_text = f"skew={skew(intervals):.2f}\nkurtosis={kurtosis(intervals):.2f}"
        plt.xlabel('Inter-spike Interval (ms)')
        plt.ylabel('Counts')
        plt.title(f'unit {unit_id} - {"Non-Poisson" if is_non_poisson else "Poisson"}\n{stats_text}')
        plt.savefig(f"{save_path}/neuronid_{unit_id}_truncate_{trialnum}.png", dpi=600, bbox_inches='tight')
        plt.clf()
    return is_non_poisson

def neurons_ISI_counting_trunc():
    non_poisson_count = 0
    total_processed = 0
    trunc_num = int(elec_dura // trial_interval)
    print(trunc_num)
    plt.figure(figsize=(10, 6))
    for neuron_id in popu_ids: #第j个neuron
        spike_times = singleneuron_spiketimes(neuron_id)
        for trial_num in range(0,trunc_num):
            start = trial_num * trial_interval
            end = (trial_num + 1) * trial_interval
            trunc_sptimes = spike_times[(spike_times > start) & (spike_times < end)]
            result = ISI_counting(trunc_sptimes,neuron_id,trial_num)
            if result is not None:
                total_processed += 1
                if result:
                    non_poisson_count += 1
    
    print(f"非泊松神经元比例: {non_poisson_count}/{total_processed} ({non_poisson_count/total_processed:.1%})")
    with open(f"{save_path}/non_poisson_ratio.txt", "w") as f:
        f.write(f"非泊松神经元比例: {non_poisson_count}/{total_processed} ({non_poisson_count/total_processed:.1%})\n")

def neurons_ISI_counting():
    non_poisson_count = 0
    total_processed = 0
    plt.figure(figsize=(10, 6))
    for neuron_id in popu_ids: #第j个neuron
        spike_times = singleneuron_spiketimes(neuron_id)
        result = ISI_counting(spike_times,neuron_id)
        if result is not None:
            total_processed += 1
            if result:
                non_poisson_count += 1
    
    print(f"非泊松神经元比例: {non_poisson_count}/{total_processed} ({non_poisson_count/total_processed:.1%})")
    with open(f"{save_path}/non_poisson_ratio.txt", "w") as f:
        f.write(f"非泊松神经元比例: {non_poisson_count}/{total_processed} ({non_poisson_count/total_processed:.1%})\n")

neurons_ISI_counting()

#each_neuron_ISI(neurons)
#save_selected_neuron_spike_times(189)