"""
# coding: utf-8
@author: Yuhao Zhang
last updated: 11/11/2024
data from: Xinchao Chen
"""
import neo
import quantities as pq
import matplotlib.pyplot as plt
from elephant.conversion import BinnedSpikeTrain
from viziphant.spike_train_correlation import plot_cross_correlation_histogram
from elephant.spike_train_correlation import cross_correlation_histogram  # noqa
from elephant.spike_train_synchrony import spike_contrast
from elephant import statistics
from collections import Counter
from scipy.signal import find_peaks
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import os
import glob
import pandas as pd
from scipy.signal import correlate, correlation_lags
import matplotlib.pyplot as plt

import random
np.set_printoptions(threshold=np.inf)
np.seterr(divide='ignore',invalid='ignore')

### path
mice = '20230113_littermate'
save_path = r'C:\Users\zyh20\Desktop\ET_data analysis\delay\sudo_spike_test'

sample_rate=30000 #spikeGLX neuropixel sample rate

# get single neuron spike train
def generate_poisson_process(rate, duration):
    """
    生成一个泊松过程的时间序列
    
    参数:
    rate: float, 泊松过程的事件发生率 (每秒的事件数)
    duration: float, 模拟的总时间长度 (秒)
    
    返回:
    times: list, 事件发生的时间点
    """
    times = []
    current_time = 0
    
    # 生成时间间隔，直到达到指定的总时间长度
    while current_time < duration:
        # 生成指数分布的随机数，作为下一个事件的时间间隔
        interval = np.random.exponential(1 / rate)
        current_time += interval
        if current_time < duration:
            times.append(current_time)
    
    return times

def TLCC(spiketrain1,spiketrain2):
    #plt.figure()
    binned_spiketrain1 = BinnedSpikeTrain(spiketrain1, bin_size=0.5*pq.ms,tolerance=None)
    binned_spiketrain2 = BinnedSpikeTrain(spiketrain2, bin_size=0.5*pq.ms,tolerance=None)
    # scipy包计算cross correlation
    array1 = binned_spiketrain1.to_array().astype(int)[0]
    array2 = binned_spiketrain2.to_array().astype(int)[0]
    # 计算相关性
    corr = correlate(array2, array1, mode='full')
    # 计算相关性的时间滞后范围
    lags = correlation_lags(len(array1), len(array2), mode='full')

    lag_indices = (lags >=  int(-100)) & (lags <= int(100))  # 选择滞后在-50ms to 50ms
    corr = corr[lag_indices]
    lags = lags[lag_indices]
    delay = lags[np.argmax(corr)]
    if np.sum(corr) == 0:  
        delay = 0
    '''
    plt.figure(figsize=(10, 6))
    plt.plot(lags, corr)
    plt.show()
    '''
    delay = delay/2
    return delay

def max_corr(time_series_1,time_series_others):
    # 初始化最大相关系数和对应的序列索引
    max_corr = -1
    best_series_index = -1

    # 计算第一个时间序列与每个其他时间序列的相关性
    for i in range(time_series_others.shape[0]):
        corr = np.corrcoef(time_series_1, time_series_others[i])[0, 1]  # 相关性系数
        #print(f"Correlation between time_series_1 and time_series_{i+1}: {corr:.4f}")
        
        # 更新最大相关性和对应索引
        if corr > max_corr:
            max_corr = corr
            best_series_index = i

    return best_series_index

def ProbD(delay,region_A,region_B):
    values, counts = np.unique(delay, return_counts=True)
    probabilities = counts / len(delay)
    df = pd.DataFrame({
        'delays':  pd.Series(delay),
        'sorted_delays': pd.Series(values),
        'sorted_counts': pd.Series(probabilities)
    })
    df.to_csv(save_path+f'/Delay_A_{region_A}_B_{region_B}.csv', index=False)
    # 绘制概率密度分布
    plt.figure()
    plt.bar(values, probabilities, width=0.6, color='skyblue')
    plt.title('Delay Distribution')
    plt.xlabel('Delay(ms)')
    plt.ylabel('Probability')
    plt.xticks(np.arange(-50, 51, 10))
    plt.text(0.99, 0.9, 'sudo spike trains', horizontalalignment='right', verticalalignment='top',transform=plt.gca().transAxes,fontsize=9)
    plt.savefig(save_path+f"/DelayDistribution_A_{region_A}_B_{region_B}.png",dpi=600,bbox_inches = 'tight')

def neuron_pair_corre(neuron_A_id,neuron_B_id,region_A,region_B):
    #取出第一组中的所有neuron，分别计算每个neuron和第二组neurons的最高相关度neuron
    m=0
    for neuron_A in neuron_A_id:
        spike_times_A = singleneuron_spiketrain(neuron_A)
        spiketrain_A = neo.SpikeTrain(spike_times_A,units='sec',t_start=0, t_stop=(times[-1]/sample_rate)[0])
        binned_spiketrain_A_neo = BinnedSpikeTrain(spiketrain_A, bin_size=0.5*pq.ms,tolerance=None)
        binned_spiketrain_A = binned_spiketrain_A_neo.to_array().astype(int)[0]
        j=0
        for neuron_B in neuron_B_id:
            spike_times_B = singleneuron_spiketrain(neuron_B)
            spiketrain_B = neo.SpikeTrain(spike_times_B,units='sec',t_start=0, t_stop=(times[-1]/sample_rate)[0])
            binned_spiketrain_B_neo = BinnedSpikeTrain(spiketrain_B, bin_size=0.5*pq.ms,tolerance=None)
            binned_spiketrain_B = binned_spiketrain_B_neo.to_array().astype(int)[0]
            if j==0:
                binned_spikes_B = binned_spiketrain_B
            else:
                binned_spikes_B = np.vstack((binned_spikes_B, binned_spiketrain_B))
            j=j+1
        
        best_series_index = max_corr(binned_spiketrain_A,binned_spikes_B)
        combination = np.array([neuron_A,neuron_B_id[best_series_index]])
        if m == 0:
            combinations = combination
        else:
            combinations = np.vstack((combinations, combination))
        m=m+1

    combinations_save = pd.DataFrame(combinations, columns=[f'{region_A}',f'{region_B}'])
    combinations_save.to_csv(save_path+f'/combinations_A_{region_A}_B_{region_B}.csv', index=False)
    return combinations

def delay_compu_session(neuron_a_id,neuron_b_id,marker):
    spike_times1 = singleneuron_spiketrain(neuron_a_id)
    spike_times2 = singleneuron_spiketrain(neuron_b_id)
    
    ## ET
    marker_start = marker['time_interval_left_end'].iloc[0]
    marker_end = marker['time_interval_right_end'].iloc[-1]
    spike_times1_trail = spike_times1[(spike_times1 > marker_start) & (spike_times1 < marker_end)]
    spike_times2_trail = spike_times2[(spike_times2 > marker_start) & (spike_times2 < marker_end)]
    if len(spike_times1_trail) > 2 and len(spike_times2_trail) > 2:   #筛选两个neuron的spike数目均大于2个的
        spiketrain1 = neo.SpikeTrain(spike_times1_trail,units='sec',t_start=marker_start, t_stop=marker_end)
        spiketrain2 = neo.SpikeTrain(spike_times2_trail,units='sec',t_start=marker_start, t_stop=marker_end)
        delay_trial = TLCC(spiketrain1,spiketrain2)
    else: delay_trial = False
    
    return delay_trial

def test_two_series():
    #计算两组存在delya的spike trian的TLCC图
    spike_times1_trail = np.array(generate_poisson_process(rate=100, duration=10))
    spike_times2_trail = spike_times1_trail + 0.02
    spike_times2_trail = spike_times2_trail[spike_times2_trail <= 10]
    spike_times2_trail = spike_times2_trail[spike_times2_trail >= 0]
    spike_times1_trail = spike_times1_trail.tolist()
    spike_times2_trail = spike_times2_trail.tolist()
    spiketrain1 = neo.SpikeTrain(spike_times1_trail,units='sec',t_start=0, t_stop=10)
    spiketrain2 = neo.SpikeTrain(spike_times2_trail,units='sec',t_start=0, t_stop=10)
    delay = TLCC(spiketrain1,spiketrain2)
    print('delay =',delay,'ms')

def test_two_group():
    delay = np.array([])

    # A脑区一百个神经元spike train，每个神经元发放率随机的Poisson process
    spike_times_A_region = [generate_poisson_process(rate=random.randint(50, 100), duration=10) for _ in range(100)]
    # B脑区一百个神经元spike train，每个神经元为A对应神经元spike train平移
    spike_times_B_region = [[x + 0.02 for x in sublist] for sublist in spike_times_A_region]  #B滞后A 20ms
    print(spike_times_A_region[0])
    print(spike_times_B_region[0])
    ## 对neurons两两组合
    result = []
    for i in spike_times_A_region:
        for j in spike_times_B_region:
            result.append((i, j))
    index = 0
    ## 对combination计算TLCC
    for comb in result:
        spike_times1 = comb[0]
        spike_times2 = comb[1]
        if index == 0:
            print(spike_times1)
            print(spike_times2)
        spike_times2 = np.array(spike_times2)
        spike_times2 = spike_times2[spike_times2 <= 10]
        spike_times2 = spike_times2[spike_times2 >= 0]
        spike_times2 = spike_times2.tolist()
        spiketrain1 = neo.SpikeTrain(spike_times1,units='sec',t_start=0, t_stop=10)
        spiketrain2 = neo.SpikeTrain(spike_times2,units='sec',t_start=0, t_stop=10)
        delay_onetrial = TLCC(spiketrain1,spiketrain2)
        if delay_onetrial != False:
            delay = np.append(delay, delay_onetrial)
        index = index + 1
    ##计算完所有delay后，统计
    ProbD(delay,'region_A','region_B')
    
test_two_group()