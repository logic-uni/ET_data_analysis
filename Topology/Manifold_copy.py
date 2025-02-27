"""
# coding: utf-8
@author: Yuhao Zhang
last updated: 06/26/2024
data from: Xinchao Chen
"""
import torch
import numpy as np
import pandas as pd
import pynapple as nap
import pynacollada as pyna
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns 
import networkx as nx
from matplotlib.animation import FuncAnimation, PillowWriter 
import csv
import os
from itertools import count
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
from sklearn.datasets import load_iris,load_digits
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from matplotlib.colors import hsv_to_rgb
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from math import log
from sklearn.mixture import GaussianMixture
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
from scipy.stats import norm
from scipy.signal import savgol_filter
import warnings
import scipy.io as sio
np.set_printoptions(threshold=np.inf)
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d
from mayavi import mlab

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mice = '20230623-condictional tremor4'
fig_save_path = r'C:\Users\zyh20\Desktop\ET_data analysis\manifold\20230623-condictional tremor4'
dis_save_path = r'C:\Users\zyh20\Desktop\ET_data analysis\manifold\runstop_center_distance'
### marker
treadmill_marker_path = r'E:\chaoge\sorted neuropixels data\20230623-condictional tremor4\202300622_Syt2_512_2_Day18_P79_g0'
treadmill = pd.read_csv(treadmill_marker_path+'/treadmill_move_stop_velocity.csv',index_col=0)
print(treadmill)

### electrophysiology
sample_rate=30000 #spikeGLX neuropixel sample rate
file_directory=r'E:\chaoge\sorted neuropixels data\20230623-condictional tremor4\202300622_Syt2_512_2_Day18_P79_g0\202300622_Syt2_512_2_Day18_P79_g0_imec0'
identities = np.load(file_directory+'/spike_clusters.npy') #存储neuron的编号id,对应phy中的第一列id
times = np.load(file_directory+'/spike_times.npy')  #
channel = np.load(file_directory+'/channel_positions.npy')
neurons = pd.read_csv(file_directory+'/region_neuron_id.csv', low_memory = False,index_col=0)#防止弹出警告
print(neurons)
print("检查treadmill总时长和电生理总时长是否一致")
print("电生理总时长")
print((times[-1]/sample_rate)[0])
print("跑步机总时长") 
print(treadmill['time_interval_right_end'].iloc[-1])

#### spike train & firing rates

# get single neuron spike train
def singleneuron_spiketrain(id):
    x = np.where(identities == id)
    y=x[0]
    spike_times=np.empty(len(y))
    for i in range(0,len(y)):
        z=y[i]
        spike_times[i]=times[z]/sample_rate
    return spike_times

# +-2 nuerons around selected id
def neurons_spiketrain(id):
    x = np.where(identities == id)
    y=x[0]
    spike_times=np.empty((6,len(y)))
    for m in range(-2,2):
        x = np.where(identities == id+m)
        y=x[0]
        for i in range(0,len(y)):
            z=y[i]
            spike_times[m,i]=times[z]/sample_rate
    return spike_times


#split spike times into trials, now is the +-0.5s of start pushing the rod
def Trials_spiketrain(spike_times,marker):
    for i in range(len(marker)):
        Trials_spiketrain=np.array([])
        
        for j in range(len(spike_times)):
            if marker[i,0]<spike_times[j] and spike_times[j]<marker[i,1]:
                Trials_spiketrain=np.append(Trials_spiketrain,spike_times[j])
        if Trials_spiketrain.size != 0:
            for k in range(1,len(Trials_spiketrain)):
                Trials_spiketrain[k]=Trials_spiketrain[k]-Trials_spiketrain[0]
            Trials_spiketrain[0]=0
        y=np.full((len(Trials_spiketrain),1),i)      
        plt.plot(Trials_spiketrain,y, '|', color='gray') 

    plt.title('neuron') 
    plt.xlabel("time") 
    plt.xlim(0,1)
    plt.ylim(-2,len(marker)+5)
    plt.show()

# spike counts
def build_time_window_domain(bin_edges, offsets, callback=None):
    callback = (lambda x: x) if callback is None else callback
    domain = np.tile(bin_edges[None, :], (len(offsets), 1))
    domain += offsets[:, None]
    return callback(domain)

def build_spike_histogram(time_domain,
                          spike_times,
                          dtype=None,
                          binarize=False):

    time_domain = np.array(time_domain)

    tiled_data = np.zeros(
        (time_domain.shape[0], time_domain.shape[1] - 1),
        dtype=(np.uint8 if binarize else np.uint16) if dtype is None else dtype
    )

    starts = time_domain[:, :-1]
    ends = time_domain[:, 1:]

    data = np.array(spike_times)

    start_positions = np.searchsorted(data, starts.flat)
    end_positions = np.searchsorted(data, ends.flat, side="right")
    counts = (end_positions - start_positions)

    tiled_data[:, :].flat = counts > 0 if binarize else counts

    return tiled_data

def spike_counts(
    spike_times,
    bin_edges,
    movement_start_time,
    binarize=False,
    dtype=None,
    large_bin_size_threshold=0.001,
    time_domain_callback=None
):

    #build time domain
    bin_edges = np.array(bin_edges)
    domain = build_time_window_domain(
        bin_edges,
        movement_start_time,
        callback=time_domain_callback)

    out_of_order = np.where(np.diff(domain, axis=1) < 0)
    if len(out_of_order[0]) > 0:
        out_of_order_time_bins = \
            [(row, col) for row, col in zip(out_of_order)]
        raise ValueError("The time domain specified contains out-of-order "
                            f"bin edges at indices: {out_of_order_time_bins}")

    ends = domain[:, -1]
    starts = domain[:, 0]
    time_diffs = starts[1:] - ends[:-1]
    overlapping = np.where(time_diffs < 0)[0]

    if len(overlapping) > 0:
        # Ignoring intervals that overlaps multiple time bins because
        # trying to figure that out would take O(n)
        overlapping = [(s, s + 1) for s in overlapping]
        warnings.warn("You've specified some overlapping time intervals "
                        f"between neighboring rows: {overlapping}, "
                        "with a maximum overlap of"
                        f" {np.abs(np.min(time_diffs))} seconds.")
        
    #build_spike_histogram
    tiled_data = build_spike_histogram(
        domain,
        spike_times,
        dtype=dtype,
        binarize=binarize
    )
    return tiled_data

def binary_spiketrain(id,marker):  #each trial
    # bin
    bin_width = 0.0007
    duration = 5  #一个trial的时间，或你关注的时间段的长度
    pre_time = -0.1
    post_time = duration
    bins = np.arange(pre_time, post_time+bin_width, bin_width)   

    histograms=spike_counts(
        singleneuron_spiketrain(id),
        bin_edges=bins,
        movement_start_time=marker,
        )
    print(histograms)

    return histograms

def eachtrial_average_firingrate(histograms,bin_width):
    firing_rate=histograms.mean(1)/bin_width
    print(firing_rate)
    sio.savemat('/firing_rate/20230414/fir_%d.mat'%id, {'fir_%d'%id:firing_rate}) #存成matlab格式，方便后续辨识传递函数

    return firing_rate

def firingrate_time(id,marker,duration,bin_width):
    # bin
    pre_time = 0
    post_time = duration
    bins = np.arange(pre_time, post_time+bin_width,bin_width)  # bin_width默认 0.14
    # histograms
    histograms=spike_counts(
        singleneuron_spiketrain(id),
        bin_edges=bins,
        movement_start_time=marker,
        )
    return histograms

def firingrate_shortime(id,marker):
    # bin
    bin_width = 0.05
    duration = 0.5   #一个trial的时间，或你关注的时间段的长度
    pre_time = -0.3
    post_time = duration
    bins = np.arange(pre_time, post_time+bin_width, bin_width)  
    # histograms
    histograms=spike_counts(
        singleneuron_spiketrain(id),
        bin_edges=bins,
        movement_start_time=marker,
        )
    print(histograms)
    return histograms

def population_spikecounts(neuron_id,marker_start,marker_end,Artificial_time_division,bin):  
    #这里由于allen的spike counts函数是针对视觉的，因此对trial做了划分，必要trialmarker作为参数，因此这里分假trial，再合并
    #Artificial_time_division是把整个session人为划分为一个个时间段trial
    #bin是对firing rate的滑窗大小，单位s
    marker=np.array(range(int(marker_start),int(marker_end)-int(marker_end)%Artificial_time_division,Artificial_time_division))
    #get a 2D matrix with neurons, trials(trials contain times), trials and times are in the same dimension
    for j in range(len(neuron_id)): #第j个neuron
        #每个neuron的tials水平append
        for i in range(len(marker)):
            if i == 0:
                one_neruon = firingrate_time(neuron_id[j],marker,Artificial_time_division,bin)[0]
            else:
                trail = firingrate_time(neuron_id[j],marker,Artificial_time_division,bin)[i]
                one_neruon = np.append(one_neruon, trail)
        if j == 0:
            neurons = one_neruon
        else:
            neurons = np.vstack((neurons, one_neruon))
    '''
    print(neurons)
    print(neurons.shape)
    '''
    time_len=(int(marker_end)-int(marker_end)%Artificial_time_division)/bin
    return neurons,time_len

### manifold
def adjust_array(arr):
    if any(x < 0 for x in arr):
        min_val = min(arr)
        diff = -min_val
        arr = [x + diff for x in arr]
    return np.array(arr)

def reduce_dimension_to1(count,bin_size,region_name): # 默认: 0.1 感觉改bin_size影响不大，改firing rate的bin size影响较大
    #smooth data
    count = pd.DataFrame(count)
    rate = np.sqrt(count/bin_size)
    #对数据做均值  默认: window=50  min_periods=1  感觉改这些值影响不大，改firing的bin size影响较大
    rate = rate.rolling(window=50,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=2) 
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
    y=adjusted_array
    x=np.arange(0,len(y))
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
    ani.save(fig_save_path+f"/1Ddynamic/{region_name}_dynamic_raw.gif", writer='pillow')

    ### y=x^2 李雅普诺夫能量函数
    # 小球距离曲线的偏移量
    offset = 7
    array = np.transpose(X_isomap)[0]
    adjusted_array = adjust_array(array)
    # 初始化动画
    x=adjusted_array
    y=x*x
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
    ani.save(fig_save_path+f"/1Ddynamic/{region_name}_dynamic_x^2.gif", writer='pillow')

def reduce_dimension(count,bin_size,region_name,stage): # 默认: 0.1 感觉改bin_size影响不大，改firing rate的bin size影响较大
    #smooth data
    count = pd.DataFrame(count)
    rate = np.sqrt(count/bin_size)
    #对数据做均值  默认: window=50  min_periods=1  感觉改这些值影响不大，改firing的bin size影响较大
    rate = rate.rolling(window=50,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=2) 
    #reduce dimension
    ## PCA
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(rate.values)   #对应的是Explained variance
    explained_variance_ratio = pca.explained_variance_ratio_   #每个主成分所解释的方差比例
    explained_variance_sum = np.cumsum(explained_variance_ratio)  #计算累积解释方差比例
    #画explained_variance图
    x=list(range(len(explained_variance_ratio)))
    fig = plt.figure()
    plt.plot(x,explained_variance_ratio, color='blue', label='each PC ratio')
    plt.plot(x,explained_variance_sum, color='red', label='ratio sum')
    plt.title(f"{region_name}_{stage}_PC_explained variance ratio")
    plt.xlabel('PC')
    plt.ylabel('Value')
    plt.legend()
    #plt.show()
    plt.savefig(fig_save_path+f"/{region_name}_{stage}_PC_explained variance ratio.png",dpi=600,bbox_inches = 'tight')
    X_isomap = Isomap(n_components = 3, n_neighbors = 21).fit_transform(rate.values)  #对应的是Residual variance
    #X_tsne = TSNE(n_components=3,random_state=21,perplexity=20).fit_transform(rate.values)  #t-SNE没有Explained variance，t-SNE 旨在保留局部结构而不是全局方差
    return X_isomap

def reduce_dimension_ISOMAP(count,bin_size,region_name,stage): # 默认: 0.1 感觉改bin_size影响不大，改firing rate的bin size影响较大
    #smooth data
    count = pd.DataFrame(count)
    rate = np.sqrt(count/bin_size)
    #对数据做均值  默认: window=50  min_periods=1  感觉改这些值影响不大，改firing的bin size影响较大
    rate = rate.rolling(window=50,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=2) 
    #reduce dimension
    X_isomap = Isomap(n_components = 3, n_neighbors = 21).fit_transform(rate.values)  #对应的是Residual variance
    X=rate.values
    D_high = pairwise_distances(X, metric='euclidean')
    residual_variances = []
    # Calculate residual variance for different embedding dimensions
    for n_components in range(1, 6):
        isomap = Isomap(n_neighbors=5, n_components=n_components)
        X_low = isomap.fit_transform(X)
        D_low = pairwise_distances(X_low, metric='euclidean')
        residual_variance = 1 - np.sum((D_high - D_low) ** 2) / np.sum(D_high ** 2)
        residual_variances.append(residual_variance)
    fig = plt.figure()
    # Plot residual variance
    plt.plot(range(1, 6), residual_variances, marker='o')
    plt.xlabel('Number of Dimensions')
    plt.ylabel('Residual Variance')
    plt.title(f"{region_name}_{stage}_Isomap Residual Variance")
    #plt.show()
    plt.savefig(fig_save_path+f"/{region_name}_{stage}_Isomap Residual Variance.png",dpi=600,bbox_inches = 'tight')

    return X_isomap

def manifold_fixed_colored_intervals(redu_dim_data,marker,bin,time_len_int_aft_bin,region_name,redu_method):  #静态流形，时间区间颜色标记
    colors = ['#ffcccc', '#ff6666', '#ff3333', '#cc0000'] #从浅红到深红的颜色列表，用于不同速度挡位画图区分
    velocity_level=np.array(marker['velocity_level'][1::2])
    #分别单独画前三个PC
    for a in range(0,3):
        q=0 # q控制颜色
        fig = plt.figure()
        for i in range(0,len(marker['run_or_stop'])-1):
            left=int(marker['time_interval_left_end'].iloc[i]/bin)
            right=int(marker['time_interval_right_end'].iloc[i]/bin)
            x_run=np.arange(left-2,right+2)
            x_stop=np.arange(left,right)
            if marker['run_or_stop'].iloc[i] == 1:
                plt.plot(x_run,redu_dim_data[left-2:right+2,a],color=colors[int(velocity_level[q])])
                q=q+1
            else:
                plt.plot(x_stop,redu_dim_data[left:right,a],color='blue')
        plt.title(f"{region_name}_{redu_method}_manifold_colored_intervals_PC{a+1}")
        plt.xlabel("t")
        #plt.show()
        plt.savefig(fig_save_path+f"/{region_name}_{redu_method}_manifold_colored_intervals_PC{a+1}.png",dpi=600,bbox_inches = 'tight')

    #画三维manifold
    p=0 # p控制颜色
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    ax.set_title(f"{region_name}_{redu_method}_manifold_colored_intervals")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    for i in range(0,len(marker['run_or_stop'])-1):
        left=int(marker['time_interval_left_end'].iloc[i]/bin)
        right=int(marker['time_interval_right_end'].iloc[i]/bin)
        if marker['run_or_stop'].iloc[i] == 1:
            ax.plot3D(redu_dim_data[left-2:right+2,0],redu_dim_data[left-2:right+2,1],redu_dim_data[left-2:right+2,2],colors[int(velocity_level[p])])
            p=p+1
        else:
            ax.plot3D(redu_dim_data[left:right,0],redu_dim_data[left:right,1],redu_dim_data[left:right,2],'blue')
    end_inter_start=int(marker['time_interval_left_end'].iloc[-1]/bin)
    ax.plot3D(redu_dim_data[end_inter_start:time_len_int_aft_bin,0],redu_dim_data[end_inter_start:time_len_int_aft_bin,1],redu_dim_data[end_inter_start:time_len_int_aft_bin,2],'blue')

    #plt.show()
    plt.savefig(fig_save_path+f"/{region_name}_{redu_method}_manifold_colored_intervals.png",dpi=600,bbox_inches = 'tight')

    

def manifold_dynamic_colored_intervals(redu_dim_data,marker,bin,time_len_int_aft_bin,region_name,redu_method):  #动态流形，时间区间颜色标记
    p=0 # p控制颜色
    colors = ['#ffcccc', '#ff6666', '#ff3333', '#cc0000'] #从浅红到深红的颜色列表，用于不同速度挡位画图区分
    velocity_level=np.array(marker['velocity_level'][1::2])
    fig = plt.figure()
    ax =  fig.add_subplot(projection = '3d')
    ax.set_title(f"{region_name}_{redu_method}_manifold_colored_intervals")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.grid(True)
    plt.ion()  # interactive mode on!!!! 很重要,有了他就不需要plt.show()了
    for i in range(0,len(marker['run_or_stop'])-1):
        left=int(marker['time_interval_left_end'].iloc[i]/bin)
        right=int(marker['time_interval_right_end'].iloc[i]/bin)
        for j in range(left,right):
            if marker['run_or_stop'].iloc[i] == 1:
                ax.plot3D(redu_dim_data[j:j+2,0],redu_dim_data[j:j+2,1],redu_dim_data[j:j+2,2],colors[int(velocity_level[p])])
            else:
                ax.plot3D(redu_dim_data[j:j+2,0],redu_dim_data[j:j+2,1],redu_dim_data[j:j+2,2],'blue')
            plt.pause(0.01)
        if marker['run_or_stop'].iloc[i] == 1:
            p=p+1
    end_inter_start=int(marker['time_interval_left_end'].iloc[-1]/bin)
    for m in range(end_inter_start,time_len_int_aft_bin):
        ax.plot3D(redu_dim_data[m:m+2,0],redu_dim_data[m:m+2,1],redu_dim_data[m:m+2,2],'blue')

def manifold_fixed(redu_dim_data,stage,region_name):  #静态流形，无时间区间颜色标记，用于trial_average，只需输入降维后的，无需marker
    #分别单独画前三个PC
    for i in range(0,3):
        fig = plt.figure()
        plt.plot(redu_dim_data[:,i])
        plt.title(f"{region_name}_{stage}_manifold_trail_average_PC{i+1}")
        plt.xlabel("t")
        #plt.show()
        plt.savefig(fig_save_path+f"/{region_name}_{stage}_PC{i+1}_manifold_trail_average.png",dpi=600,bbox_inches = 'tight')
    #画三维manifold
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    ax.plot3D(redu_dim_data[:,0],redu_dim_data[:,1],redu_dim_data[:,2],'blue')
    ax.set_title(f"{region_name}_{stage}_manifold_trail_average")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.savefig(fig_save_path+f"/{region_name}_{stage}_manifold_trail_average.png",dpi=600,bbox_inches = 'tight')
    

def manifold_dynamic(redu_dim_data,stage):  #静态流形，无时间区间颜色标记，用于trial_average，只需输入降维后的，无需marker
    fig = plt.figure()
    ax =  fig.add_subplot(projection = '3d')
    ax.set_title(f"Essential Tremor Manifold, {stage}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.grid(True)
    plt.ion()  # interactive mode on!!!! 很重要,有了他就不需要plt.show()了
    for j in range(0,len(redu_dim_data)):
        ax.plot3D(redu_dim_data[j:j+2,0],redu_dim_data[j:j+2,1],redu_dim_data[j:j+2,2],'blue')
        plt.pause(0.01)

'''
def manifold_fixed_colored_intervals(X_isomap,marker,bin,time_len_int_aft_bin): 
    colors=[None] * time_len_int_aft_bin
    for i in range(0,len(marker['run_or_stop'])-1):
        t_left_withbin=int(marker['time_interval_left_end'].iloc[i]/bin)
        t_right_withbin=int(marker['time_interval_right_end'].iloc[i]/bin)
        if marker['run_or_stop'].iloc[i] == 1:
            colors[t_left_withbin:t_right_withbin] = ['red'] * (t_right_withbin-t_left_withbin)
        else:
            colors[t_left_withbin:t_right_withbin] = ['blue'] * (t_right_withbin-t_left_withbin)

    end_inter_start=int(marker['time_interval_left_end'].iloc[-1]/bin)
    colors[end_inter_start:time_len_int_aft_bin] = ['blue'] * (time_len_int_aft_bin-end_inter_start)
    print(len(colors))
    manifold_fixed(X_isomap,colors)
'''

def splicing_neural_data(time,data):
    phy = np.zeros((1, 3))
    for i in range(0,time.shape[0]):
        left=int(time[i][0])
        right=int(time[i][1])
        temp=np.array([data[left:right,0],data[left:right,1],data[left:right,2]]).T
        if i == 0:
            phy = temp
        else:
            phy=np.concatenate((phy, temp), axis=0)
    return phy

def splicing_neural_data_high_dim(time,data):
    for i in range(0,time.shape[0]):
        left=int(time[i][0])
        right=int(time[i][1])
        temp=data[:,left:right]
        if i == 0:
            phy = temp
        else:
            phy=np.hstack((phy, temp))
    return phy

def manifold_center_distance(data,marker,region_name):
    data2pca=data.T
    redu_dim_data=reduce_dimension(data2pca,0.1,region_name,stage='all_session')
    run = marker[marker['run_or_stop'] == 1]
    stop = marker[marker['run_or_stop'] == 0]
    run_time = np.array([run['time_interval_left_end'],run['time_interval_right_end']]).T
    stop_time = np.array([stop['time_interval_left_end'],stop['time_interval_right_end']]).T
    run_reduc_di_phy = splicing_neural_data(run_time,redu_dim_data)
    stop_reduc_di_phy = splicing_neural_data(stop_time,redu_dim_data)
    run_phy = splicing_neural_data_high_dim(run_time,data)
    stop_phy = splicing_neural_data_high_dim(stop_time,data)
    
    # 计算运动簇的中心
    center_run_higdim = np.mean(run_phy.T, axis=0)
    center_run_3d = np.mean(run_reduc_di_phy, axis=0)
    # 计算静止簇的中心
    center_stop_higdim = np.mean(stop_phy.T, axis=0)
    center_stop_3d = np.mean(stop_reduc_di_phy, axis=0)
    # 计算两个中心之间的距离
    high_dist = np.linalg.norm(center_run_higdim - center_stop_higdim)  #高维空间点的二范数，欧式距离
    three_dist = np.linalg.norm(center_run_3d - center_stop_3d) #低维空间点的二范数，欧式距离
    '''
    #plot 3d scatter
    fig = plt.figure()
    ax = plt.subplot(projection = '3d')
    # 绘制簇的中心点
    ax.scatter(run_reduc_di_phy[:,0],run_reduc_di_phy[:,1],run_reduc_di_phy[:,2], c='gold',alpha = 0.5, zorder=2)
    ax.scatter(stop_reduc_di_phy[:,0],stop_reduc_di_phy[:,1],stop_reduc_di_phy[:,2], c='green',alpha = 0.5, zorder=2)
    
    ax.scatter(center_run_3d[0], center_run_3d[1], center_run_3d[2], c='red',  s=150, label='run', zorder=1)
    ax.scatter(center_stop_3d[0], center_stop_3d[1], center_stop_3d[2], c='blue',  s=150, label='stop', zorder=1)
    # 计算中心点的方位角，为了朝向中心点呈现
    x_origin, y_origin, z_origin = center_stop_3d[0], center_stop_3d[1], center_stop_3d[2]
    x_target, y_target, z_target = center_run_3d[0], center_run_3d[1], center_run_3d[2]
    azim = np.arctan2(y_target - y_origin, x_target - x_origin)
    azim = np.degrees(azim)  # 转换为度
    # 计算仰角
    elev = np.arccos(z_target / np.sqrt((x_target - x_origin)**2 + (y_target - y_origin)**2 + z_target**2))
    elev = np.degrees(elev)  # 转换为度
    ax.view_init(elev=elev, azim=azim % 360)  # 确保方位角在0到360度之间
    # 设置图形标题和标签
    ax.set_title(f'{region_name}')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.legend()
    #plt.show()
    plt.savefig(fig_save_path+f"/run and stop center of neural manifold_{region_name}.png",dpi=600,bbox_inches = 'tight')
    '''
    return high_dist,three_dist

def plot_hyper_plane(X, y,X_pca):
    # 使用线性回归拟合超平面
    model = LinearRegression()
    model.fit(X, y)

    # 超平面的系数
    w = model.coef_
    b = model.intercept_

    # 计算超平面在PCA降维后的空间中的斜率和截距
    slope = -w[0] / w[1]
    intercept = -b / w[1]

    # 绘制数据点和超平面在PCA降维后的空间中
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolors='k')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    # 绘制超平面
    x_vals = np.array(plt.gca().get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--', color='red', label='Hyperplane')

    plt.legend()
    plt.title('Hyperplane in PCA-reduced Space')
    plt.show()

def interp_helper(values, num=50, kind='quadratic'):
    interp_i = np.linspace(min(values), max(values), num)
    return interp1d(np.linspace(min(values), max(values), len(values)), values, kind=kind)(interp_i)

def manifold_fitplane(X_isomap):
    for i in range(0,len(X_isomap)):
        ax.scatter(x_track[:,0], x_track[:,1], x_track[:,2], 'blue')
        x_track_s=[X_isomap[i,0],X_isomap[i,1],X_isomap[i,2]]
        x_track = np.vstack((x_track, x_track_s))
        plt.pause(0.01)
    #plot fixed 3D trajectory
    fig = plt.figure()   
    ax = fig.gca(projection='3d')
    ax.scatter(X_isomap[:,0],X_isomap[:,1],X_isomap[:,2],label='Essential Tremor Neural Manifold')  #分别取三列的值作为x,y,z的值
    ax.legend()
    plt.show()
    # plot 3D colored smooth trajectory
    x_new, y_new, z_new = (interp_helper(i,800) for i in (X_isomap[:,0], X_isomap[:,1], X_isomap[:,2]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    zmax = np.array(z_new).max()
    zmin = np.array(z_new).min()
    for i in range(len(z_new) - 1):
        ax.plot(x_new[i:i + 2], y_new[i:i + 2], z_new[i:i + 2],
                color=plt.cm.jet(int((np.array(z_new[i:i + 2]).mean() - zmin) * 255 / (zmax - zmin))))
    plt.title('Essential Tremor Neural Manifold')
    plt.show()
    
    #fit plane
    v1=fit_plane(X_isomap[0:215,0],X_isomap[0:215,1],X_isomap[0:215,2],'r')
    #normal vector
    #vv1 = np.vstack((v1, v2))
    #print(vv14)
    # normlize x to same symbol
    for i in range(0,len(v)):
        if v[i,2] > 0:
            v[i,0]=(-1)*v[i,0]
            v[i,1]=(-1)*v[i,1]
            v[i,2]=(-1)*v[i,2]
    print(v)
    # amplify
    v=v*10
    print(v)
    plot_normal_vector(v)

    plt.show()
    return X_isomap

def plot_surface(x,y,z,region_name):
    x_grid, y_grid = np.meshgrid(x, y)
    z_grid = np.empty((len(x),len(y))) 
    for i in range(len(x)):
        z_grid[i][i] = z[i]
    s = mlab.mesh(x_grid, y_grid, z_grid
                  )
    mlab.show()
    '''
    x = savgol_filter(x, window_length=5, polyorder=3)
    y = savgol_filter(y, window_length=5, polyorder=3)
    z = savgol_filter(z, window_length=5, polyorder=3)


    # 创建一个新的图像
    fig = plt.figure()
    # 创建一个3D绘图区域
    ax = fig.add_subplot(111, projection='3d')
    # 绘制三维曲面图
    ax.plot_surface(x_grid, y_grid, z_grid, rstride = 15, cstride = 15,cmap = plt.get_cmap('rainbow'), alpha=None, antialiased=True)
    # 设置标签
    ax.set_title(f"manifold_surface_{region_name}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.savefig(fig_save_path+f"/manifold_surface_{region_name}.png",dpi=600,bbox_inches = 'tight')
    '''

def plot_surface_2(x,y,z):
    f = interpolate.interp2d(x, y, z, kind='cubic')
    znew = f(x, y)

    #修改x,y，z输入画图函数前的shape
    xx1, yy1 = np.meshgrid(x, y)
    newshape = (xx1.shape[0])*(xx1.shape[0])
    y_input = xx1.reshape(newshape)
    x_input = yy1.reshape(newshape)
    z_input = znew.reshape(newshape)

    #画图
    sns.set(style='white')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(x_input,y_input,z_input,cmap=cm.coolwarm)
    plt.show()

def fit_plane(xs,ys,zs,color_name):
    ax = plt.subplot(projection = '3d')
    # do fit
    tmp_A = [] #存储x和y
    tmp_b = [] #存储z
    for i in range(len(xs)):
        tmp_A.append([xs[i], ys[i], 1])
        tmp_b.append(zs[i])
    b = np.matrix(tmp_b).T
    A = np.matrix(tmp_A)

    # Manual solution
    fit = (A.T * A).I * A.T * b  #该式由最小二乘法推导得出
    errors = b - A * fit #计算估计值与真实值的误差
    residual = np.linalg.norm(errors)  #求误差矩阵的范数，即残差平方和SSE
    error_withmean = np.mean(b)- A * fit #计算估计值与平均值的误差
    regression = np.linalg.norm(error_withmean)  #求回归误差矩阵的范数，即回归平方和SSR
    SST=residual+regression
    R2=1-(residual/SST)

    print("solution: %f x + %f y + %f = z" % (fit[0], fit[1], fit[2]))
    print("residual:" ,residual)
    print("R2:" ,R2)

    # plot plane
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    X,Y = np.meshgrid(np.arange(xlim[0], xlim[1]),
                    np.arange(ylim[0], ylim[1]))
    Z = np.zeros(X.shape)
    for r in range(X.shape[0]):
        for c in range(X.shape[1]):
            Z[r,c] = fit[0] * X[r,c] + fit[1] * Y[r,c] + fit[2]
    ax.plot_surface(X,Y,Z, color=color_name,alpha=0.5)
    normal_v=np.array([fit[0,0], fit[1,0], fit[2,0]])
    return normal_v

def plot_normal_vector(normal_vector):
    #plot normal vector
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    a=[0,0,0]
    a_count=0
    b=[0,0,0]
    b_count=0
    for i in range(0,len(normal_vector)):
        if i ==0 or i ==5 or i ==10 or i ==11 or i ==18 or i ==19 or 24 <= i <= 26 or 33 <= i <= 34 or 38 <= i <= 40 or 44 <= i <= 46:
            ax.quiver(0,0,0,normal_vector[i,0],normal_vector[i,1],normal_vector[i,2],arrow_length_ratio=0.1,color='r',length=5, normalize=True)
            a=a+normal_vector[i]
            a_count=a_count+1
        if 1 <= i <= 4 or 6 <= i <= 9 or 12 <= i <= 17 or 20 <= i <= 23 or 27 <= i <= 32 or 35 <= i <= 37 or 41 <= i <= 43:
            ax.quiver(0,0,0,normal_vector[i,0],normal_vector[i,1],normal_vector[i,2],arrow_length_ratio=0.1,color='b',length=5, normalize=True)
            b=b+normal_vector[i]
            b_count=b_count+1
    # compute average motion vectors and average rest vectors
    a_av=a/a_count
    b_av=b/b_count
    ax.quiver(0,0,0,a_av[0],a_av[1],a_av[2],arrow_length_ratio=0.1,color='g',length=8, normalize=True)
    ax.quiver(0,0,0,b_av[0],b_av[1],b_av[2],arrow_length_ratio=0.1,color='y',length=8, normalize=True)
    #compute included angle
    cos_angle = np.dot(a_av, b_av) / (np.linalg.norm(a_av) * np.linalg.norm(b_av))
    angle = np.arccos(cos_angle)
    print('夹角为：', angle * 180 / np.pi, '度')
    ax.set_xlim(-8.05,8.05)
    ax.set_ylim(-8.05,8.05)
    ax.set_zlim(-8.05,8.05)
    plt.show()

def interval_cuttage(marker):
    run_during=np.array([])
    stop_during=np.array([])
    run_time_dura=np.empty((0, 2)).astype(int) 
    stop_time_dura=np.empty((0, 2)).astype(int) 
    run=marker[marker['run_or_stop'] == 1]
    stop=marker[marker['run_or_stop'] == 0]
    for i in range(0,len(marker['run_or_stop'])):
        start=int(marker['time_interval_left_end'].iloc[i])
        end=int(marker['time_interval_right_end'].iloc[i])
        if marker['run_or_stop'].iloc[i] == 1:
            #由于treadmill运动和静止交替的持续时间随机，因此检测持续时间的最小长度，作为一个trail的长度，各个持续时间如果大于最小区间的X倍，按照X个trial计入
            run_during=np.append(run_during,end-start)  #获得所有运动区间的持续时间长度
        else:
            stop_during=np.append(stop_during,end-start) #获得所有静止区间的持续时间长度
    min_run=np.min(run_during)      #获得运动/静止 最小区间的时间长度
    min_stop=np.min(stop_during)
    run_multiple=np.floor(run_during/min_run).astype(int)      #获得每个时间区间可以被划分为最小时间区间的几倍
    stop_multiple=np.floor(stop_during/min_stop).astype(int)
    #获取所有以最小运动时间长度为基准的运动区间
    for j in range(0,len(run_multiple)):
        if run_multiple[j] != 1:
            for n in range(1,run_multiple[j]+1):  
                left=int(run['time_interval_left_end'].iloc[j])+min_run*(n-1)
                right=left+min_run
                time_dura=[int(left),int(right)]
                run_time_dura=np.vstack([run_time_dura, time_dura])
        else:
            left=int(run['time_interval_left_end'].iloc[j])
            right=left+min_run
            time_dura=[int(left),int(right)]
            run_time_dura=np.vstack([run_time_dura, time_dura])
    #获取所有以最小静止时间长度为基准的静止区间
    for k in range(0,len(stop_multiple)):
        if stop_multiple[k] != 1:
            for m in range(1,stop_multiple[k]+1):  
                left=int(stop['time_interval_left_end'].iloc[k])+min_stop*(m-1)
                right=left+min_stop
                time_dura=[int(left),int(right)]
                stop_time_dura=np.vstack([stop_time_dura, time_dura])
        else:
            left=int(stop['time_interval_left_end'].iloc[k])
            right=left+min_stop
            time_dura=[int(left),int(right)]
            stop_time_dura=np.vstack([stop_time_dura, time_dura])

    return run_time_dura,stop_time_dura

def trail_average(data,run_time_dura,stop_time_dura):
    ## run
    #run is a matrix with trials * neurons * timepoint, each value is the firing rate in this time point
    run = np.zeros((run_time_dura.shape[0], data.shape[0], run_time_dura[0][1]-run_time_dura[0][0]))
    for ti in range(0,run_time_dura.shape[0]):
        neuron_runpiece = data[:, run_time_dura[ti][0]:run_time_dura[ti][1]]   #firing rate * neurons矩阵，按照区间切片
        if neuron_runpiece.shape == run[ti, :, :].shape:
            run[ti, :, :] = neuron_runpiece
    # 三维run矩阵沿着第一个维度，对应相加求平均
    run_average=np.mean(run, axis=0)

    ## stop
    stop = np.zeros((stop_time_dura.shape[0], data.shape[0], stop_time_dura[0][1]-stop_time_dura[0][0]))
    for ti_stop in range(0,stop_time_dura.shape[0]):
        neuron_stoppiece = data[:, stop_time_dura[ti_stop][0]:stop_time_dura[ti_stop][1]]   #firing rate * neurons矩阵，按照区间切片
        if neuron_stoppiece.shape == stop[ti_stop-1, :, :].shape:
            stop[ti_stop, :, :] = neuron_stoppiece
    # 三维stop矩阵沿着第一个维度，对应相加求平均
    stop_average=np.mean(stop, axis=0)
    
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

def main_function(neurons,marker):
    high_dim_dist_all = []
    three_dim_dist_all = []
    for i in range(neurons.shape[1]):  #遍历所有的脑区
        bin=1
        region_name = neurons.columns.values[i]
        
        neuron_id = np.array(neurons.iloc[:, i].dropna()).astype(int)  #提取其中一个脑区的neuron id
        marker_start = marker['time_interval_left_end'].iloc[0]
        marker_end = marker['time_interval_right_end'].iloc[-1]
        
        '''
        ### manifold 1PC
        data,time_len = population_spikecounts(neuron_id,marker_start,marker_end,30,bin)
        data_norm=normalize_fr(data)
        data2pca=data_norm.T
        reduce_dimension_to1(data2pca,0.1,region_name)
        
        ### manifold surface
        data2pca=data.T
        redu_dim_data=reduce_dimension(data2pca,0.1,region_name,stage='all_session')
        #plot_surface_2(redu_dim_data[:,0],redu_dim_data[:,1],redu_dim_data[:,2])
        plot_surface(redu_dim_data[:,1],redu_dim_data[:,2],redu_dim_data[:,0],region_name)  #PC2=x,PC3=y,PC1=z
        
        ### manifold each trail
        data2pca_each_trail=data.T
        redu_dim_data=reduce_dimension(data2pca_each_trail,0.1,region_name,stage='all_session')
        manifold_fixed_colored_intervals(redu_dim_data,marker,bin,int(time_len),region_name,redu_method='PCA')  #fixed & colored intervals
        # ISOMAP extract nolinear structure
        redu_dim_data_ISOMAP=reduce_dimension_ISOMAP(data2pca_each_trail,0.1,region_name,stage='all_session')
        manifold_fixed_colored_intervals(redu_dim_data_ISOMAP,marker,bin,int(time_len),region_name,redu_method='ISOMAP')  #fixed & colored intervals
        
        #### manifold 动态图
        if region_name == 'Superior vestibular nucleus':
            neuron_id = np.array(neurons.iloc[:, i].dropna()).astype(int)  #提取其中一个脑区的neuron id
            marker_start = marker['time_interval_left_end'].iloc[0]
            marker_end = marker['time_interval_right_end'].iloc[-1]
            data,time_len = population_spikecounts(neuron_id,marker_start,marker_end,30,bin)
            data2pca_each_trail=data.T
            redu_dim_data_ISOMAP=reduce_dimension_ISOMAP(data2pca_each_trail,0.1,region_name,stage='all_session')
            manifold_dynamic_colored_intervals(redu_dim_data_ISOMAP,marker,bin,int(time_len),region_name,redu_method='ISOMAP')
        
        ### manifold trial average
        run_time_dura,stop_time_dura=interval_cuttage(marker)
        run_average,stop_average=trail_average(data,run_time_dura,stop_time_dura)
        #print(run_average.shape)
        #print(stop_average.shape)

        run2pca=run_average.T
        run_redu_dim_aver=reduce_dimension(run2pca,0.1,region_name,stage='Run')
        manifold_fixed(run_redu_dim_aver,'Run',region_name)
        #manifold_dynamic(run_redu_dim_aver,'Run')

        stop2pca=stop_average.T
        stop_redu_dim_aver=reduce_dimension(stop2pca,0.1,region_name,stage='Stop')
        manifold_fixed(stop_redu_dim_aver,'Stop',region_name)
        #manifold_dynamic(stop_redu_dim_aver,'Stop')
        '''
        ### manifold_distance
        data2dis,time_len = population_spikecounts(neuron_id,marker_start,marker_end,30,0.1)
        #高维距离,三维距离
        normalized_data = normalize_fr(data2dis)#对原始firing rate进行normalize，以便于分析距离差异
        high_dist,three_dist = manifold_center_distance(normalized_data,marker,region_name)
        
        high_dim_dist_all.append(high_dist)
        three_dim_dist_all.append(three_dist)

    # manifold_distance
    manifold_dist = {'region': neurons.columns.values, 'high_dim_dist': high_dim_dist_all,'three_dim_dist': three_dim_dist_all}
    df = pd.DataFrame(manifold_dist)
    df.to_csv(dis_save_path+f"/{mice}_manifold_run_stop_distance.csv", index=False)
    

main_function(neurons,treadmill)