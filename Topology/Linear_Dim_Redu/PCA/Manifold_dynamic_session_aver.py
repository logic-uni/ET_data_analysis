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
from matplotlib.animation import FuncAnimation
from sklearn.metrics import pairwise_distances
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
from elephant.conversion import BinnedSpikeTrain
np.set_printoptions(threshold=np.inf)

# ------- NEED CHANGE -------
data_path = '/data2/zhangyuhao/xinchao_data/Givenme/1410-1-tremor-Day3-3CVC-FM_g0'
save_path = '/home/zhangyuhao/Desktop/Result/ET/Manifold/NP2/givenme/1410-1-tremor-Day3-3CVC-FM_g0'
# ------- NO NEED CHANGE -------
fr_bin = 10  #ms
### Behavior
marker = pd.read_csv(data_path+'/Behavior/marker.csv') 
print(marker)

### Electrophysiology
fs = 30000  # spikeGLX neuropixel sample rate
identities = np.load(data_path+'/Sorted/kilosort4/spike_clusters.npy') # time series: unit id of each spike
times = np.load(data_path+'/Sorted/kilosort4/spike_times.npy')  # time series: spike time of each spike
'''
## For across region
#neurons = pd.read_csv(data_path+'/Sorted/kilosort4/mapping_artifi.csv') 
# 按region分组，提取每组的第一列cluster_id
region_groups = neurons.groupby('region')
region_neuron_ids = {}
for region, group in region_groups:
    # 提取每组的第一列（cluster_id），去除缺失值
    cluster_ids = group.iloc[:, 0].dropna().astype(int).values
    region_neuron_ids[region] = cluster_ids
region = 'VN'  # 选择感兴趣的region
neuron_ids = region_neuron_ids[region]  # 获取该region的neuron_ids
'''
## For single region
neurons = pd.read_csv(data_path + '/Sorted/kilosort4/cluster_group.tsv', sep='\t')  # for single region

print(neurons)
region = 'VN'  # 选择感兴趣的region
neuron_ids = neurons['cluster_id'].dropna().astype(int).values
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
        spiketrain = neo.SpikeTrain(spike_times_trail,
                                    units = 'sec',
                                    t_start = marker_start,
                                    t_stop = marker_end)
        fr = BinnedSpikeTrain(spiketrain, bin_size = fr_bin*pq.ms,tolerance = None)
        one_neruon = fr.to_array().astype(int)[0]
        if j == 0:
            neurons = one_neruon
        else:
            neurons = np.vstack((neurons, one_neruon))
    return neurons

def adjust_array(arr):
    if any(x < 0 for x in arr):
        min_val = min(arr)
        diff = -min_val
        arr = [x + diff for x in arr]
    return np.array(arr)

def oneDdynamic(count,bin_size,region_name): # 默认: 0.1 感觉改bin_size影响不大，改firing rate的bin size影响较大
    #smooth data
    count = pd.DataFrame(count)
    rate = np.sqrt(count/bin_size)
    #对数据做均值  默认: window=50  min_periods=1  感觉改这些值影响不大，改firing的bin size影响较大
    rate = rate.rolling(window = 50,
                        win_type = 'gaussian',
                        center = True,
                        min_periods = 1).mean(std=2)  # axis 无需输入默认0
    #reduce dimension
    '''
    ## PCA
    pca = PCA(n_components=1)
    X_pca = pca.fit_transform(rate.values)   #对应的是Explained variance
    explained_variance_ratio = pca.explained_variance_ratio_   #每个主成分所解释的方差比例
    explained_variance_sum = np.cumsum(explained_variance_ratio)  #计算累积解释方差比例
    #画explained_variance图
    x=list(range(len(explained_variance_ratio)))
    '''
    fig = plt.figure()
    X_isomap = Isomap(n_components = 1, n_neighbors = 21).fit_transform(rate.values)  #对应的是Residual variance

    ### 原始降维到一维后的值随时间变化
    # 小球距离曲线的偏移量
    offset = 7
    array = np.transpose(X_isomap)[0]
    adjusted_array = adjust_array(array)
    # 初始化动画
    y = adjusted_array
    x = np.arange(0,len(y))
    fig, ax = plt.subplots()
    line, = ax.plot(x, y, linestyle='-', color='b')
    ball, = ax.plot([], [], marker='o', markersize=7, color='r')
    # 设置图形界面属性
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(y) - 1, np.max(y) + 1)
    ax.set_title(f"{region_name}_dynamic_1D")
    ax.set_xlabel("time")
    ax.set_ylabel("neural state")
    ax.grid(True)
    # 更新函数，用于每一帧的更新
    def update(frame):
        ball_x = x[frame]
        ball_y = y[frame] + offset  # 小球在曲线上方的位置偏移
        ball.set_data([ball_x],[ball_y])
        return line, ball,
    # 创建动画
    ani = FuncAnimation(fig, update, frames=len(x), interval=20, blit=True)
    # 使用imagemagick将动画保存为GIF图片
    ani.save(save_path+f"/1Ddynamic_{region_name}_dynamic_raw.gif", writer='pillow')

    ### y=x^2 李雅普诺夫能量函数
    # 小球距离曲线的偏移量
    offset = 7
    array = np.transpose(X_isomap)[0]
    adjusted_array = adjust_array(array)
    # 初始化动画
    x = adjusted_array
    y = x*x
    fig, ax = plt.subplots()
    line, = ax.plot(x, y, linestyle='-', color='b')
    ball, = ax.plot([], [], marker='o', markersize=7, color='r')
    # 设置图形界面属性
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(y) - 1, np.max(y) + 1)
    ax.set_title(f"{region_name}_dynamic_1D")
    ax.set_xlabel("time")
    ax.set_ylabel("neural energy")
    ax.grid(True)
    # 更新函数，用于每一帧的更新
    def update(frame):
        ball_x = x[frame]
        ball_y = y[frame] + offset  # 小球在曲线上方的位置偏移
        ball.set_data([ball_x], [ball_y])
        return line, ball,
    # 创建动画
    ani = FuncAnimation(fig, update, frames=len(x), interval=20, blit=True)
    # 使用imagemagick将动画保存为GIF图片
    ani.save(save_path+f"/1Ddynamic_{region_name}_dynamic_x^2.gif", writer='pillow')

def redu_dim(count,bin_size,region_name,stage): # 默认: 0.1 感觉改bin_size影响不大，改firing rate的bin size影响较大
    #smooth data
    count = pd.DataFrame(count)
    rate = np.sqrt(count/bin_size)
    #对数据做均值  默认: window=50  min_periods=1  感觉改这些值影响不大，改firing的bin size影响较大
    rate = rate.rolling(window = 50,
                        win_type = 'gaussian',
                        center = True,
                        min_periods = 1).mean(std=2)  # axis 无需输入默认0
    #reduce dimension
    ## PCA
    pca = PCA(n_components = 3)
    X_pca = pca.fit_transform(rate.values)   #对应的是Explained variance
    #X_isomap = Isomap(n_components = 3, n_neighbors = 21).fit_transform(rate.values)  #对应的是Residual variance
    #X_tsne = TSNE(n_components=3,random_state=21,perplexity=20).fit_transform(rate.values)  #t-SNE没有Explained variance，t-SNE 旨在保留局部结构而不是全局方差
    explained_variance_ratio = pca.explained_variance_ratio_   #每个主成分所解释的方差比例
    explained_variance_sum = np.cumsum(explained_variance_ratio)  #计算累积解释方差比例
    #画explained_variance图
    x=list(range(len(explained_variance_ratio)))
    plt.figure()
    plt.plot(x,explained_variance_ratio, color = 'b', label = 'each PC ratio')
    plt.plot(x,explained_variance_sum, color = 'r', label = 'ratio sum')
    plt.title(f"{region_name}_{stage}_PC_explained variance ratio")
    plt.xlabel('PC')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(save_path+f"/{region_name}_{stage}_PC_explained_var_ratio.png",dpi = 600,bbox_inches = 'tight')
    plt.close()
    return X_pca

def redu_dim_ISOMAP(count,smooth_bin,region_name,stage): # extract nolinear structure # 默认: 0.1 感觉改bin_size影响不大，改firing rate的bin size影响较大
    #smooth data
    count = pd.DataFrame(count)
    rate = np.sqrt(count/smooth_bin)
    #对数据做均值  默认: window=50  min_periods=1  感觉改这些值影响不大，改firing的bin size影响较大
    rate = rate.rolling(window = 50,
                        win_type = 'gaussian',
                        center = True,
                        min_periods = 1).mean(std=2)   # axis 无需输入默认0
    #reduce dimension
    X_isomap = Isomap(n_components = 3, n_neighbors = 21).fit_transform(rate.values)  #对应的是Residual variance
    X = rate.values
    D_high = pairwise_distances(X, metric = 'euclidean')
    residual_variances = []
    # Calculate residual variance for different embedding dimensions
    for n_components in range(1, 6):
        isomap = Isomap(n_neighbors = 5, n_components = n_components)
        X_low = isomap.fit_transform(X)
        D_low = pairwise_distances(X_low, metric = 'euclidean')
        residual_variance = 1 - np.sum((D_high - D_low) ** 2) / np.sum(D_high ** 2)
        residual_variances.append(residual_variance)
    # Plot residual variance
    plt.figure()
    plt.plot(range(1, 6), residual_variances, marker='o')
    plt.xlabel('Number of Dimensions')
    plt.ylabel('Residual Variance')
    plt.title(f"{region_name}_{stage}_Isomap Residual Variance")
    plt.savefig(save_path+f"/{region_name}_{stage}_Isomap_Residual_Var.png",dpi=600,bbox_inches = 'tight')
    plt.close()
    return X_isomap

def manifold_color_trial(redu_dim_data,marker,region_name,redu_method):  #静态流形，时间区间颜色标记
    #colors = ['#ffcccc', '#ff6666', '#ff3333', '#cc0000'] #从浅红到深红的颜色列表，用于不同速度挡位画图区分
    #velocity_level=np.array(marker['velocity_level'][1::2])
    #分别单独画前三个PC
    for PC in range(0,3):
        q = 0 # q控制颜色
        plt.figure()
        for i in range(0,len(marker['run_or_stop'])-1):
            left = int(marker['time_interval_left_end'].iloc[i] / fr_bin)
            right = int(marker['time_interval_right_end'].iloc[i] / fr_bin)
            #x_run = np.arange(left-2,right+2)
            x_run = np.arange(left,right + 1)
            x_stop = np.arange(left,right)
            if marker['run_or_stop'].iloc[i] == 1:
                #plt.plot(x_run,redu_dim_data[left-2:right+2,PC],color=colors[int(velocity_level[q])])
                plt.plot(x_run,redu_dim_data[left:right+1,PC],color='r')
                q = q+1
            else:
                plt.plot(x_stop,redu_dim_data[left:right,PC],color='blue')
        plt.title(f"{region_name}_{redu_method}_manifold_colored_intervals_PC{PC+1}")
        plt.xlabel("t")
        plt.savefig(save_path+f"/{region_name}_{redu_method}_trials_PC{PC+1}.png",dpi=600,bbox_inches = 'tight')
        plt.close()

    #画三维manifold
    p = 0 # p控制颜色
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    ax.set_title(f"{region_name}_{redu_method}_manifold_colored_intervals")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    for i in range(0,len(marker['run_or_stop'])-1):
        left = int(marker['time_interval_left_end'].iloc[i] / fr_bin)
        right = int(marker['time_interval_right_end'].iloc[i] / fr_bin)
        if marker['run_or_stop'].iloc[i] == 1:
            #ax.plot3D(redu_dim_data[left-2:right+2,0],redu_dim_data[left-2:right+2,1],redu_dim_data[left-2:right+2,2],colors[int(velocity_level[p])])
            ax.plot3D(redu_dim_data[left-2:right+2,0],redu_dim_data[left-2:right+2,1],redu_dim_data[left-2:right+2,2],'r')
            p = p + 1
        else:
            ax.plot3D(redu_dim_data[left:right,0],redu_dim_data[left:right,1],redu_dim_data[left:right,2],'blue')
    # 这两句仅用于NP1的makrer,最后一段时间区间因为没有在marker中标记，所以取最后一个marker作为开始，时间长度作为结束，需要额外输入行为总时长也就是time_len_int_aft_bin
    #end_inter_start = int(marker['time_interval_left_end'].iloc[-1]/fr_bin)
    #ax.plot3D(redu_dim_data[end_inter_start:time_len_int_aft_bin,0],redu_dim_data[end_inter_start:time_len_int_aft_bin,1],redu_dim_data[end_inter_start:time_len_int_aft_bin,2],'blue')
    ax.set_xlim([-0.2, 0.2])
    ax.set_ylim([-0.2, 0.2])
    ax.set_zlim([-0.2, 0.2])
    plt.savefig(save_path+f"/{region_name}_{redu_method}_trials.png",dpi = 600,bbox_inches = 'tight')
    plt.close()

def manifold_trial_aver(redu_dim_data,stage,region_name):  #静态流形，无时间区间颜色标记，用于trial_average，只需输入降维后的，无需marker
    #分别单独画前三个PC
    for i in range(0,3):
        plt.figure()
        plt.plot(redu_dim_data[:,i])
        plt.title(f"{region_name}_{stage}_manifold_trail_average_PC{i+1}")
        plt.xlabel("t")
        plt.savefig(save_path+f"/{region_name}_{stage}_PC{i+1}_trail_average.png",dpi = 600,bbox_inches = 'tight')
        plt.close()
    
    #画三维manifold
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    ax.plot3D(redu_dim_data[:,0],redu_dim_data[:,1],redu_dim_data[:,2],'blue')
    ax.set_title(f"{region_name}_{stage}_manifold_trail_average")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.savefig(save_path+f"/{region_name}_{stage}_trail_average.png",dpi = 600,bbox_inches = 'tight')
    plt.close()

def interval_cut(marker):
    run_during = np.array([])
    stop_during = np.array([])
    run_time_dura = np.empty((0, 2)).astype(int) 
    stop_time_dura = np.empty((0, 2)).astype(int) 
    run=marker[marker['run_or_stop'] == 1]
    stop=marker[marker['run_or_stop'] == 0]
    for i in range(0,len(marker['run_or_stop'])):
        start = int(marker['time_interval_left_end'].iloc[i])
        end = int(marker['time_interval_right_end'].iloc[i])
        if marker['run_or_stop'].iloc[i] == 1:
            run_during = np.append(run_during,end-start)  #获得所有运动区间的持续时间长度
        else:
            stop_during = np.append(stop_during,end-start) #获得所有静止区间的持续时间长度
    min_run = np.min(run_during)      #获得运动/静止 最小区间的时间长度
    min_stop = np.min(stop_during)
    run_multiple = np.floor(run_during/min_run).astype(int)      #获得每个时间区间可以被划分为最小时间区间的几倍
    stop_multiple = np.floor(stop_during/min_stop).astype(int)
    #获取所有以最小运动时间长度为基准的运动区间
    for j in range(0,len(run_multiple)):
        if run_multiple[j] != 1:
            for n in range(1,run_multiple[j] + 1):  
                left = int(run['time_interval_left_end'].iloc[j]) + min_run*(n-1)
                right = left+min_run
                time_dura = [int(left),int(right)]
                run_time_dura = np.vstack([run_time_dura, time_dura])
        else:
            left = int(run['time_interval_left_end'].iloc[j])
            right = left + min_run
            time_dura = [int(left),int(right)]
            run_time_dura = np.vstack([run_time_dura, time_dura])
    #获取所有以最小静止时间长度为基准的静止区间
    for k in range(0,len(stop_multiple)):
        if stop_multiple[k] != 1:
            for m in range(1,stop_multiple[k]+1):  
                left = int(stop['time_interval_left_end'].iloc[k]) + min_stop*(m-1)
                right = left + min_stop
                time_dura = [int(left),int(right)]
                stop_time_dura = np.vstack([stop_time_dura, time_dura])
        else:
            left = int(stop['time_interval_left_end'].iloc[k])
            right = left + min_stop
            time_dura = [int(left),int(right)]
            stop_time_dura = np.vstack([stop_time_dura, time_dura])

    return run_time_dura,stop_time_dura

def trail_aver(data,run_time_dura,stop_time_dura):
    ## run
    #run is a matrix with trials * neurons * timepoint, each value is the firing rate in this time point
    run = np.zeros((run_time_dura.shape[0], data.shape[0], run_time_dura[0][1] - run_time_dura[0][0]))
    for ti in range(0,run_time_dura.shape[0]):
        neuron_runpiece = data[:, run_time_dura[ti][0]:run_time_dura[ti][1]]   #firing rate*neurons矩阵，按区间切片
        if neuron_runpiece.shape == run[ti, :, :].shape:
            run[ti, :, :] = neuron_runpiece
    # 三维run矩阵沿着第一个维度，对应相加求平均
    run_average=np.mean(run, axis=0)
    ## stop
    stop = np.zeros((stop_time_dura.shape[0], data.shape[0], stop_time_dura[0][1] - stop_time_dura[0][0]))
    for ti_stop in range(0,stop_time_dura.shape[0]):
        neuron_stoppiece = data[:, stop_time_dura[ti_stop][0]:stop_time_dura[ti_stop][1]]   #firing rate*neurons矩阵，按区间切片
        if neuron_stoppiece.shape == stop[ti_stop - 1, :, :].shape:
            stop[ti_stop, :, :] = neuron_stoppiece
    # 三维stop矩阵沿着第一个维度，对应相加求平均
    stop_average = np.mean(stop, axis = 0)
    return run_average,stop_average

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

def main():
    marker_start = marker['time_interval_left_end'].iloc[0]
    marker_end = marker['time_interval_right_end'].iloc[-1]
    data = popu_fr_onetrial(neuron_ids,marker_start,marker_end)
    data_norm = normalize_fr(data)
    ### 1d
    #oneDdynamic(data2pca,0.1,region_name)
    ### each trail
    # PCA
    data2redu_trails = data_norm.T
    redu_dim_data = redu_dim(data2redu_trails,5,region,stage='trials')
    manifold_color_trial(redu_dim_data,marker,region,redu_method='PCA')
    # ISOMAP 
    redu_dim_data_ISOMAP = redu_dim_ISOMAP(data2redu_trails,1,region,stage='trials')
    manifold_color_trial(redu_dim_data_ISOMAP,marker,region,redu_method='ISOMAP')
    
    ### trial average
    # 由于运动和静止交替的持续时间随机 因此检测持续时间的最小长度 作为一个trail的长度 各个持续时间如果大于最小区间的X倍 按照X个trial计入
    # 当然最小时间长度不得小于30s
    run_time_dura,stop_time_dura = interval_cut(marker)
    run_average,stop_average = trail_aver(data,run_time_dura,stop_time_dura)
    run2pca = run_average.T
    run_redu_dim_aver = redu_dim(run2pca,0.1,region,stage = 'Run_trial_average')
    manifold_trial_aver(run_redu_dim_aver,'Run_trial_average',region)
    stop2pca = stop_average.T
    stop_redu_dim_aver = redu_dim_ISOMAP(stop2pca,0.1,region,stage = 'Stop_trial_average')
    manifold_trial_aver(stop_redu_dim_aver,'Stop_trial_average',region)
    
main()