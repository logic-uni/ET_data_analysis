"""
# coding: utf-8
@author: Yuhao Zhang
last updated: 06/02/2025
data from: Xinchao Chen
"""
# 设置核心线程参数
import os
import sys
os.environ["OPENBLAS_NUM_THREADS"] = "48"   # 总核心数的 1/3
os.environ["OMP_NUM_THREADS"] = "1"        # 禁用 OpenMP 多线程
os.environ["MKL_NUM_THREADS"] = "1"         # 禁用 MKL 多线程
os.environ["OPENBLAS_MAX_THREADS"] = "144"  # 提示库最大支持数

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import neo
import quantities as pq
from elephant import statistics
from quantities import ms, s, Hz
from sklearn.metrics import pairwise_distances
from sklearn.manifold import Isomap
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from elephant.conversion import BinnedSpikeTrain
np.set_printoptions(threshold=np.inf)

# ------- NEED CHANGE -------
data_path = '/data1/zhangyuhao/xinchao_data/NP2/demo/20250308_control_Mice_1423-15-VN-head_fixation_demo'
save_path = '/home/zhangyuhao/Desktop/Result/ET/Manifold/NP2/demo/20250308_control_Mice_1423-15-VN-head_fixation_demo'
# ------- NO NEED CHANGE -------
fr_bin = 800  #ms
trial_length = 30  # trial_length unit s
segment_len_s = 60  # 每段60秒
### Behavior
marker = pd.read_csv(data_path+'/Behavior/marker.csv') 
print(marker)

### Electrophysiology
fs = 30000  # spikeGLX neuropixel sample rate
identities = np.load(data_path+'/Sorted/kilosort4/spike_clusters.npy') # time series: unit id of each spike
times = np.load(data_path+'/Sorted/kilosort4/spike_times.npy')  # time series: spike time of each spike
'''
## For across region
neurons = pd.read_csv(data_path+'/Sorted/kilosort4/mapping_artifi.csv') 
neurons = neurons[neurons['fr'] >= 0.01]  # 去掉发放率小于0.01的神经元
# 按region分组，提取每组的第一列cluster_id
region_groups = neurons.groupby('region')
region_neuron_ids = {}
for region, group in region_groups:
    # 提取每组的第一列（cluster_id），去除缺失值
    cluster_ids = group.iloc[:, 0].dropna().astype(int).values
    region_neuron_ids[region] = cluster_ids
region = 'CbX'  # 感兴趣的region
neuron_ids = region_neuron_ids[region]  # 获取该region的neuron_ids
'''
## For single region
neurons = pd.read_csv(data_path + '/quality_metrics.csv')
region = 'VN'  # 感兴趣的region
neurons = neurons[neurons['firing_rate'] >= 0.01]  # 去掉发放率小于0.01的神经元
neuron_ids = neurons['cluster_id'].dropna().astype(int).values

print(neurons)
print(f"region: {region}")
print(f"neuron_ids: {neuron_ids}")
print("Test if Ephys duration same as motion duration...")
print(f"Ephys duration: {(times[-1]/fs)} s")  # for NP1, there's [0] after times[-1]/fs
print(f"motion duration: {marker['time_interval_right_end'].iloc[-1]} s")
neuron_num = neurons.count().transpose().values

def singleneuron_spiketimes(id):
    x = np.where(identities == id)
    y = x[0]
    spike_times = np.empty(len(y))
    for i in range(0,len(y)):
        z = y[i]
        spike_times[i] = times[z]/fs
    return spike_times

def popu_fr_onetrial(neuron_ids,marker_start,marker_end):   
    for j in range(len(neuron_ids)): #第j个neuron
        spike_times = singleneuron_spiketimes(neuron_ids[j])
        spike_times_trail = spike_times[(spike_times > marker_start) & (spike_times < marker_end)]
        spiketrain = neo.SpikeTrain(spike_times_trail, units = 'sec', t_start = marker_start, t_stop = marker_end)
        #fr = BinnedSpikeTrain(spiketrain, bin_size = fr_bin*pq.ms, tolerance = None)
        ## covolutioned firing rate in each bin
        cov_rate = statistics.instantaneous_rate(spiketrain,sampling_period = fr_bin * pq.ms,kernel='auto')  
        fr0 = cov_rate.magnitude.flatten()
        ## spike count in each bin
        #histogram_count = statistics.time_histogram(spiketrain, bin_size = fr_bin * ms) 
        #fr1 = histogram_count.magnitude.flatten()
        ## firing rate in each bin
        #histogram_rate = statistics.time_histogram(spiketrain, bin_size = fr_bin * ms, output='rate')  
        #fr2 = histogram_rate.magnitude.flatten()
        #one_neruon = fr.to_array().astype(int)[0]
        if j == 0:
            neurons = fr0
        else:
            neurons = np.vstack((neurons, fr0))
    return neurons

def normalize_fr(data2dis):
    '''
    #标准化方法1 z-score 会出现负值, PCA不适应报错
    # 计算每行的均值和标准差
    means = np.mean(data2dis, axis=1, keepdims=True)
    stds = np.std(data2dis, axis=1, keepdims=True)
    # 计算z-score
    z_scores = (data2dis - means) / stds
    '''
    #标准化方法2 标准化到0-1
    normalized_data = (data2dis - data2dis.min(axis=1, keepdims=True)) / (data2dis.max(axis=1, keepdims=True) - data2dis.min(axis=1, keepdims=True))
    return normalized_data

def PCA_explained_var(rate,stage,dim): # extract linear structure
    pca = PCA(n_components = dim)
    X_pca = pca.fit_transform(rate.values)   #对应的是Explained variance
    explained_variance_ratio = pca.explained_variance_ratio_   #每个主成分所解释的方差比例
    explained_variance_sum = np.cumsum(explained_variance_ratio)  #计算累积解释方差比例
    #画explained_variance图
    x = list(range(len(explained_variance_ratio)))
    plt.figure()
    plt.plot(x,explained_variance_ratio, color = 'b', label = 'each PC ratio')
    plt.plot(x,explained_variance_sum, color = 'r', label = 'ratio sum')
    plt.title(f"{region}_{stage}_PC_explained_var_ratio")
    plt.xlabel('PC')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(save_path+f"/{region}_{stage}_PCA_explainedvar.png",dpi = 600,bbox_inches = 'tight')
    plt.close()
    return X_pca

def redu_dim(fr,stage):
    fr = pd.DataFrame(fr)
    ##​ 1. ​稳定方差 非常重要 切不可去除
    #放电率高的神经元方差大，放电率低的神经元方差小 （异方差性）
    #这种异方差性 影响后续的降维（如PCA）或聚类分析的效果，因为这些分析通常假设数据的方差是恒定的
    #平方根变换 可以使得方差趋于稳定，即变换后的数据的方差不再依赖于均值
    fr_sqrt = np.sqrt(fr)  
    ## 2. 平滑数据 默认: window=50  min_periods=1 
    # window 滑动窗口大小 100个点 
    # std 高斯核的标准差
    # min_periods允许窗口最小数据点为5
    fr_smooth = fr_sqrt.rolling(window=100, win_type='gaussian', center=True, min_periods=5).mean(std=2) 
    X_pca = PCA_explained_var(fr_smooth,stage,3)
    return X_pca

def Plot_2D_PC1_PC2(redu_dim_data,stage):  #静态流形，无时间区间颜色标记，用于trial_average，只需输入降维后的，无需marker
    plt.figure()
    plt.plot(redu_dim_data[:,0], redu_dim_data[:,1])
    plt.title(f"{region}_{stage}_PC1_PC2")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.savefig(save_path+f"/{region}_{stage}_PC1_PC2.png", dpi=600, bbox_inches='tight')
    plt.close()

def Plot_2D_PC2_PC3(redu_dim_data,stage):  #静态流形，无时间区间颜色标记，用于trial_average，只需输入降维后的，无需marker
    plt.figure()
    plt.plot(redu_dim_data[:,1], redu_dim_data[:,2])
    plt.title(f"{region}_{stage}_PC2_PC3")
    plt.xlabel("PC2")
    plt.ylabel("PC3")
    plt.savefig(save_path+f"/{region}_{stage}_PC2_PC3.png", dpi=600, bbox_inches='tight')
    plt.close()

def Plot_color_trial(redu_dim_data):  #静态流形，时间区间颜色标记
    #colors = ['#ffcccc', '#ff6666', '#ff3333', '#cc0000'] #从浅红到深红的颜色列表，用于不同速度挡位画图区分
    #velocity_level=np.array(marker['velocity_level'][1::2])
    #分别画前三个PC
    for PC in range(0,3):
        q = 0 # q控制颜色
        plt.figure()
        for i in range(0,len(marker['run_or_stop'])-1):
            start_time = marker['time_interval_left_end'].iloc[i] * 1000 # 转换为毫秒
            end_time = marker['time_interval_right_end'].iloc[i] * 1000
            left = int(start_time / fr_bin) # 实际时间除以fr_bin得到对应画图的索引
            right = int(end_time / fr_bin)
            x_run = np.arange(left,right + 1)
            x_stop = np.arange(left,right)
            if marker['run_or_stop'].iloc[i] == 1:
                plt.plot(x_run,redu_dim_data[left:right+1,PC],color='r')  # color=colors[int(velocity_level[q])]
                q = q+1
            else:
                plt.plot(x_stop,redu_dim_data[left:right,PC],color='b')
        plt.title(f"{region}_PCA_colored_intervals_PC{PC+1}")
        plt.xlabel("t")
        plt.savefig(save_path+f"/{region}_session_PC{PC+1}.png",dpi=600,bbox_inches = 'tight')
        plt.close()

    #画三维manifold
    p = 0 # p控制颜色
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    ax.view_init(elev=25, azim=120)  # 调整仰角和方位角
    ax.set_title(f"{region}_PCA_color_trials")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    for i in range(0,len(marker['run_or_stop'])-1):
        start_time = marker['time_interval_left_end'].iloc[i] * 1000 # 转换为毫秒
        end_time = marker['time_interval_right_end'].iloc[i] * 1000
        left = int(start_time / fr_bin) # 实际时间除以fr_bin得到对应的索引
        right = int(end_time / fr_bin)
        if marker['run_or_stop'].iloc[i] == 1:
            ax.plot3D(redu_dim_data[left:right+1, 0],
                      redu_dim_data[left:right+1, 1],
                      redu_dim_data[left:right+1, 2],'r')  # colors[int(velocity_level[p])]
            p += 1
        else:
            ax.plot3D(redu_dim_data[left:right+1, 0],
                      redu_dim_data[left:right+1, 1],
                      redu_dim_data[left:right+1, 2],'b')
    # 这两句仅用于NP1的makrer,最后一段时间区间因为没有在marker中标记，所以取最后一个marker作为开始，时间长度作为结束，需要额外输入行为总时长也就是time_len_int_aft_bin
    #end_inter_start = int(marker['time_interval_left_end'].iloc[-1]/fr_bin)
    #ax.plot3D(redu_dim_data[end_inter_start:time_len_int_aft_bin,0],redu_dim_data[end_inter_start:time_len_int_aft_bin,1],redu_dim_data[end_inter_start:time_len_int_aft_bin,2],'blue')
    plt.tight_layout()
    plt.savefig(save_path+f"/{region}_session_PCA_3D.png",dpi = 600,bbox_inches = 'tight')
    plt.close()

def interval_cut():
    # Beacuse the time length of running or stopping is not the same, we need to cut the marker into equal intervals trial_length
    # Only select the time interval that is not less than trial_length
    # Align all the trials with same start point and end point
    # our hyothesis is that the neural state is similar when motion transtion
    # 分别提取运动和静止的区间
    run_trials = marker[marker['run_or_stop'] == 1].copy()
    stop_trials = marker[marker['run_or_stop'] == 0].copy()
    # 计算区间长度并过滤
    run_trials['duration'] = run_trials['time_interval_right_end'] - run_trials['time_interval_left_end']
    stop_trials['duration'] = stop_trials['time_interval_right_end'] - stop_trials['time_interval_left_end']
    filtered_run = run_trials[run_trials['duration'] > trial_length]
    filtered_stop = stop_trials[stop_trials['duration'] > trial_length]

    # 将过滤后的区间存入数组 (存储为元组列表)
    run_intervals = list(filtered_run[['time_interval_left_end', 'time_interval_right_end']].itertuples(index=False, name=None))
    stop_intervals = list(filtered_stop[['time_interval_left_end', 'time_interval_right_end']].itertuples(index=False, name=None))

    # 输出结果
    print(f"运动区间(>{trial_length}):", run_intervals)
    print(f"静止区间(>{trial_length}):", stop_intervals)

    # 处理运动区间 - 只保留每个区间的前trial_length秒
    processed_run = [(left, left + trial_length) for left, _ in run_intervals]
    # 处理静止区间 - 只保留每个区间的前trial_length秒
    processed_stop = [(left, left + trial_length) for left, _ in stop_intervals]
    # 打印结果
    print(f"处理后的运动区间(前{trial_length}秒):", processed_run)
    print(f"处理后的静止区间(前{trial_length}秒):", processed_stop)

    return processed_run,processed_stop

def trail_aver(data,run_time_dura,stop_time_dura):
    # run_time_dura: list of (start, end) in seconds
    # data: shape (neurons, timepoints), timepoints sampled at fr_bin ms
    run_bin_intervals = [(int(start * 1000 // fr_bin), int(end * 1000 // fr_bin)) for start, end in run_time_dura]
    stop_bin_intervals = [(int(start * 1000 // fr_bin), int(end * 1000 // fr_bin)) for start, end in stop_time_dura]

    # Find minimum length to align all trials
    min_run_len = min(end - start for start, end in run_bin_intervals)
    min_stop_len = min(end - start for start, end in stop_bin_intervals)
    # Stack all run trials
    run_trials = []
    for start, end in run_bin_intervals:
        trial = data[:, start:start + min_run_len]
        if trial.shape[1] == min_run_len:
            run_trials.append(trial)
    run_average = np.mean(np.stack(run_trials, axis=0), axis=0)  # shape: (neurons, min_run_len)

    # Stack all stop trials
    stop_trials = []
    for start, end in stop_bin_intervals:
        trial = data[:, start:start + min_stop_len]
        if trial.shape[1] == min_stop_len:
            stop_trials.append(trial)
    stop_average = np.mean(np.stack(stop_trials, axis=0), axis=0)  # shape: (neurons, min_stop_len)

    return run_average, stop_average

def Plot_trial_aver(redu_dim_data,stage):  #静态流形，无时间区间颜色标记，用于trial_average，只需输入降维后的，无需marker
    for i in range(0,3): #独立画前三个PC
        plt.figure()
        plt.plot(redu_dim_data[:,i])
        plt.title(f"{region}_{stage}_trail_average_PC{i+1}_triallen{trial_length}s")
        plt.xlabel("t")
        plt.savefig(save_path+f"/{region}_{stage}_PC{i+1}_trail_average.png",dpi = 600,bbox_inches = 'tight')
        plt.close()
    fig = plt.figure()   #画三维manifold
    ax = fig.add_subplot(projection = '3d')
    ax.plot3D(redu_dim_data[:,0],redu_dim_data[:,1],redu_dim_data[:,2],'blue')
    ax.set_title(f"{region}_{stage}_trail_average_triallen{trial_length}s")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.savefig(save_path+f"/{region}_{stage}_trail_average.png",dpi = 600,bbox_inches = 'tight')
    plt.close()

def main():
    session_start = marker['time_interval_left_end'].iloc[0]
    session_end = marker['time_interval_right_end'].iloc[-1]
    data = popu_fr_onetrial(neuron_ids,session_start,session_end)
    data_norm = normalize_fr(data)
    ### 1. session  对整个session即所有trial进行降维
    data2redu_trails = data_norm.T
    X_pca = redu_dim(data2redu_trails,stage='session')
    Plot_color_trial(X_pca)
    Plot_2D_PC1_PC2(X_pca,'session')
    Plot_2D_PC2_PC3(X_pca,'session')
    ### 2. trial average  对trial average后的进行降维
    run_trials,static_trials = interval_cut()
    run_average,stop_average = trail_aver(data_norm,run_trials,static_trials)
    run2redu_trialave = run_average.T
    run_redu_dim_aver = redu_dim(run2redu_trialave,stage = 'Run_trial_average')
    Plot_trial_aver(run_redu_dim_aver,'Run_trial_average')
    Plot_2D_PC1_PC2(run_redu_dim_aver,f'Run_trial_average_triallen{trial_length}s')
    Plot_2D_PC2_PC3(run_redu_dim_aver,f'Run_trial_average_triallen{trial_length}s')
    stop2redu_trialave = stop_average.T
    stop_redu_dim_aver = redu_dim(stop2redu_trialave,stage = 'Stop_trial_average')
    Plot_trial_aver(stop_redu_dim_aver,'Stop_trial_average')
    Plot_2D_PC1_PC2(stop_redu_dim_aver,f'Stop_trial_average_triallen{trial_length}s')
    Plot_2D_PC2_PC3(stop_redu_dim_aver,f'Stop_trial_average_triallen{trial_length}s')
    ### 3. truncate 100s average 对整个session每100s截断，平均后进行降维
    segment_len_bins = int(segment_len_s * 1000 / fr_bin)  # 转换为对应的bin数
    num_segments = int(data_norm.shape[1] / segment_len_bins)
    truncated_segments = []
    for i in range(num_segments):
        start = i * segment_len_bins
        end = start + segment_len_bins
        segment = data_norm[:, start:end]
        if segment.shape[1] == segment_len_bins:
            truncated_segments.append(segment)
    truncated_averages = np.mean(np.stack(truncated_segments, axis=0), axis=0)
    trun2redu_trialave = truncated_averages.T  # shape: (segments, neurons)
    truncated_redu_dim = redu_dim(trun2redu_trialave, stage=f'Truncate{segment_len_s}s_average')
    Plot_trial_aver(truncated_redu_dim, f'Truncate{segment_len_s}s_average')
    Plot_2D_PC1_PC2(truncated_redu_dim, f'Truncate{segment_len_s}s_average')
    Plot_2D_PC2_PC3(truncated_redu_dim, f'Truncate{segment_len_s}s_average')

main()