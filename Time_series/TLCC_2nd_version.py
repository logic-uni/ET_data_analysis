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

np.set_printoptions(threshold=np.inf)
np.seterr(divide='ignore',invalid='ignore')

### path
mice_name = '20230113_littermate'
main_path = r'E:\xinchao\sorted neuropixels data\useful_data\20230113_littermate\data'
save_path = r'C:\Users\zyh20\Desktop\ET_data analysis\delay\20230113_littermate\run_trials\all_pairs\A_Interposed nucleus_B_Spinal vestibular nucleus'

### marker
treadmill = pd.read_csv(main_path+'/marker/treadmill_move_stop_velocity.csv',index_col=0)
print(treadmill)

### electrophysiology
sample_rate=30000 #spikeGLX neuropixel sample rate
identities = np.load(main_path+'/spike_train/spike_clusters.npy') #存储neuron的编号id,对应phy中的第一列id
times = np.load(main_path+'/spike_train/spike_times.npy')  #
channel = np.load(main_path+'/spike_train/channel_positions.npy')
neurons = pd.read_csv(main_path+'/spike_train/region_neuron_id.csv', low_memory = False,index_col=0)#防止弹出警告
print(neurons)
print("检查treadmill总时长和电生理总时长是否一致")
print("电生理总时长")
print((times[-1]/sample_rate)[0])
print("跑步机总时长") 
print(treadmill['time_interval_right_end'].iloc[-1])
neuron_num = neurons.count().transpose().values

# get single neuron spike train
def singleneuron_spiketrain(id):
    x = np.where(identities == id)
    y=x[0]
    spike_times=np.empty(len(y))
    for i in range(0,len(y)):
        z=y[i]
        spike_times[i]=times[z]/sample_rate
    return spike_times

def TLCC(spiketrain1,spiketrain2):
    #plt.figure()
    binned_spiketrain1 = BinnedSpikeTrain(spiketrain1, bin_size=0.5*pq.ms,tolerance=None)
    binned_spiketrain2 = BinnedSpikeTrain(spiketrain2, bin_size=0.5*pq.ms,tolerance=None)
    # scipy包计算cross correlation
    array1 = binned_spiketrain1.to_array().astype(int)[0]
    array2 = binned_spiketrain2.to_array().astype(int)[0]
    # 计算相关性
    corr = correlate(array1, array2, mode='full')
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

def Prob_distri(delay,region_A,region_B,mice_name):
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
    plt.title(f'{region_A} -> {region_B}')
    plt.xlabel('Delay(ms)')
    plt.ylabel('Probability')
    plt.xticks(np.arange(-50, 51, 10))
    plt.text(0.99, 0.9, f'{mice_name}', horizontalalignment='right', verticalalignment='top',transform=plt.gca().transAxes,fontsize=9)
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

def delay_runtrials_cond_ET(neuron_a_id,neuron_b_id,marker):
    spike_times1 = singleneuron_spiketrain(neuron_a_id)
    spike_times2 = singleneuron_spiketrain(neuron_b_id)
    ## ET
    for trial in np.arange(1,len(marker['time_interval_left_end']),2):
        marker_start = marker['time_interval_left_end'].iloc[trial]
        if marker['time_interval_right_end'].iloc[trial] - marker_start > 30:
            marker_end = marker_start+30
            spike_times1_trail = spike_times1[(spike_times1 > marker_start) & (spike_times1 < marker_end)]
            spike_times2_trail = spike_times2[(spike_times2 > marker_start) & (spike_times2 < marker_end)]
            if len(spike_times1_trail) > 2 and len(spike_times2_trail) > 2:   #筛选在30s的trial里，两个neuron的spike数目均大于2个的
                spiketrain1 = neo.SpikeTrain(spike_times1_trail,units='sec',t_start=marker_start, t_stop=marker_end)
                spiketrain2 = neo.SpikeTrain(spike_times2_trail,units='sec',t_start=marker_start, t_stop=marker_end)
                #计算这两个neuron在这个trial下的delay
                delay_trial = TLCC(spiketrain1,spiketrain2)
            else: delay_trial = False
    return delay_trial

def delay_runtrials_littermate(neuron_a_id,neuron_b_id,marker):
    spike_times1 = singleneuron_spiketrain(neuron_a_id)
    spike_times2 = singleneuron_spiketrain(neuron_b_id)
    ## LITTERMATE 人为切分30s的trial
    for marker_start in np.arange(105,495,30):
        marker_end = marker_start + 30
        spike_times1_trail = spike_times1[(spike_times1 > marker_start) & (spike_times1 < marker_end)]
        spike_times2_trail = spike_times2[(spike_times2 > marker_start) & (spike_times2 < marker_end)]
        if len(spike_times1_trail) > 2 and len(spike_times2_trail) > 2:  #筛选在30s的trial里，两个neuron的spike数目均大于2个的
            spiketrain1 = neo.SpikeTrain(spike_times1_trail,units='sec',t_start=marker_start, t_stop=marker_end)
            spiketrain2 = neo.SpikeTrain(spike_times2_trail,units='sec',t_start=marker_start, t_stop=marker_end)
            #计算这两个neuron在这个trial下的delay
            delay_trial = TLCC(spiketrain1,spiketrain2)
        else: delay_trial = False
    for marker_start in np.arange(705,1095,30):
        marker_end = marker_start + 30
        spike_times1_trail = spike_times1[(spike_times1 > marker_start) & (spike_times1 < marker_end)]
        spike_times2_trail = spike_times2[(spike_times2 > marker_start) & (spike_times2 < marker_end)]
        if len(spike_times1_trail) > 2 and len(spike_times2_trail) > 2:   #筛选在30s的trial里，两个neuron的spike数目均大于2个的
            spiketrain1 = neo.SpikeTrain(spike_times1_trail,units='sec',t_start=marker_start, t_stop=marker_end)
            spiketrain2 = neo.SpikeTrain(spike_times2_trail,units='sec',t_start=marker_start, t_stop=marker_end)
            #计算这两个neuron在这个trial下的delay
            delay_trial = TLCC(spiketrain1,spiketrain2)
        else: delay_trial = False
    return delay_trial

def delay_runtrials_PV_Syt2(neuron_a_id,neuron_b_id,marker):
    spike_times1 = singleneuron_spiketrain(neuron_a_id)
    spike_times2 = singleneuron_spiketrain(neuron_b_id)
    ## PV_Syt2 人为切分30s的trial
    for marker_start in np.arange(0,360,30):
        marker_end = marker_start + 30
        spike_times1_trail = spike_times1[(spike_times1 > marker_start) & (spike_times1 < marker_end)]
        spike_times2_trail = spike_times2[(spike_times2 > marker_start) & (spike_times2 < marker_end)]
        if len(spike_times1_trail) > 2 and len(spike_times2_trail) > 2:   #筛选在30s的trial里，两个neuron的spike数目均大于2个的
            spiketrain1 = neo.SpikeTrain(spike_times1_trail,units='sec',t_start=marker_start, t_stop=marker_end)
            spiketrain2 = neo.SpikeTrain(spike_times2_trail,units='sec',t_start=marker_start, t_stop=marker_end)
            #计算这两个neuron在这个trial下的delay
            delay_trial = TLCC(spiketrain1,spiketrain2)
        else: delay_trial = False
    for marker_start in np.arange(555,585,30):
        marker_end = marker_start + 30
        spike_times1_trail = spike_times1[(spike_times1 > marker_start) & (spike_times1 < marker_end)]
        spike_times2_trail = spike_times2[(spike_times2 > marker_start) & (spike_times2 < marker_end)]
        if len(spike_times1_trail) > 2 and len(spike_times2_trail) > 2:   #筛选在30s的trial里，该neuron的spike数目大于2个的
            spiketrain1 = neo.SpikeTrain(spike_times1_trail,units='sec',t_start=marker_start, t_stop=marker_end)
            spiketrain2 = neo.SpikeTrain(spike_times2_trail,units='sec',t_start=marker_start, t_stop=marker_end)
            #计算这两个neuron在这个trial下的delay
            delay_trial = TLCC(spiketrain1,spiketrain2)
        else: delay_trial = False
    return delay_trial

def delay_session(neuron_a_id,neuron_b_id,marker):
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
        
def main(neurons,marker,region_A,region_B,mice_name,mice_type):
    delay = np.array([])
    ## 分别取两个脑区全部的neuron id 
    for i in range(neurons.shape[1]):  
        region_name = neurons.columns.values[i]
        if region_name == region_A:
            region_A_id = np.array(neurons.iloc[:, i].dropna()).astype(int)
        elif region_name == region_B:
            region_B_id = np.array(neurons.iloc[:, i].dropna()).astype(int)
    ## 对关注脑区之间的neurons先对A脑区每个neuron计算和B脑区中相关度最高的pair
    #combinations = neuron_pair_corre(region_A_id,region_B_id,region_A,region_B)   
    ## 对关注脑区之间的neurons两两组合
    X, Y = np.meshgrid(region_A_id, region_B_id)
    combinations = np.vstack([X.ravel(), Y.ravel()]).T
    ## 对combination计算TLCC
    for comb in combinations:  
        neuron_a_id = comb[0]
        neuron_b_id = comb[1]
        if mice_type == 'cond_ET':
            delay_onetrial = delay_runtrials_cond_ET(neuron_a_id,neuron_b_id,marker)
        elif mice_type == 'littermate':
            delay_onetrial = delay_runtrials_littermate(neuron_a_id,neuron_b_id,marker)
        elif mice_type == 'PV_Syt2':
            delay_onetrial = delay_runtrials_PV_Syt2(neuron_a_id,neuron_b_id,marker)
        #delay_onetrial = delay_session(neuron_a_id,neuron_b_id,marker)  # session 也就是all time 算的很慢 而且结果不一定有参考价值
        if delay_onetrial != False:
            delay = np.append(delay, delay_onetrial)
    ## 计算完所有delay后，统计
    Prob_distri(delay,region_A,region_B,mice_name)

region_A = 'Interposed nucleus'
region_B = 'Spinal vestibular nucleus'
#mice_type = 'cond_ET'
#mice_type = 'PV_Syt2'
mice_type = 'littermate'
main(neurons,treadmill,region_A,region_B,mice_name,mice_type)


#A_Interposed nucleus_B_Spinal vestibular nucleus