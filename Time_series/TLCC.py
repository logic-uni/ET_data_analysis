"""
# coding: utf-8
@author: Yuhao Zhang
last updated: 09/20/2024
data from: Xinchao Chen
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from matplotlib.animation import FuncAnimation, PillowWriter 
from sklearn.metrics import pairwise_distances
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from matplotlib import cm
from scipy import interpolate
from scipy.interpolate import interp1d
from mayavi import mlab
np.set_printoptions(threshold=np.inf)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### path
mice = '20230113_littermate'
main_path = r'E:\xinchao\sorted neuropixels data\useful_data\20230113_littermate\data'

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
neuron_spike_A = np.array([])
neuron_spike_B = np.array([])

#### spike train & firing rates
# get single neuron spike train
def singleneuron_spiketimes(id):
    x = np.where(identities == id)
    y=x[0]
    spike_times=np.empty(len(y))
    for i in range(0,len(y)):
        z=y[i]
        spike_times[i]=times[z]/sample_rate
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
        singleneuron_spiketimes(id),
        bin_edges=bins,
        movement_start_time=marker,
        )
    print(histograms)

    return histograms

def firingrate_time(id,marker,duration,bin_width):
    # bin
    pre_time = 0
    post_time = duration
    bins = np.arange(pre_time, post_time+bin_width,bin_width)  # bin_width默认 0.14
    # histograms
    histograms=spike_counts(
        singleneuron_spiketimes(id),
        bin_edges=bins,
        movement_start_time=marker,
        )
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

    time_len=(int(marker_end)-int(marker_end)%Artificial_time_division)/bin
    return neurons,time_len

def TLCC(popuspike_A,popuspike_B):
    # 计算每个神经元群体的平均活动 (逐时间点求平均值)
    mean_activity_A = np.mean(popuspike_A, axis=0)
    mean_activity_B = np.mean(popuspike_B, axis=0)

    # 计算互相关
    correlation = np.correlate(mean_activity_A, mean_activity_B, mode='full')

    # 滤除滞后在0到1秒之外的滞后值
    lag_indices = (lags >=  int(-5)) & (lags <= int(5))  # 选择滞后在-2.5ms to 2.5ms
    correlation = correlation[lag_indices]
    lags = lags[lag_indices]

    # 找到互相关的峰值以及对应的时间滞后
    delay = lags[np.argmax(correlation)]

    # 可视化
    plt.figure(figsize=(10, 5))
    plt.plot(lags, correlation)
    plt.title(f'Cross-correlation between Neuron Groups A and B\nEstimated delay within 0-1s: {delay:.2f} seconds')
    plt.xlabel('Time lag (seconds)')
    plt.ylabel('Cross-correlation')
    plt.grid(True)
    plt.show()

    print(f"Estimated delay between the two neuron groups (within 0-1s): {delay:.2f} seconds")

def main_function(neurons,marker):
    for i in range(neurons.shape[1]):  #遍历所有的脑区
        bin=0.0005
        region_name = neurons.columns.values[i]
        if region_name == 'Spinal vestibular nucleus':
            neuron_id = np.array(neurons.iloc[:, i].dropna()).astype(int)  #提取其中一个脑区的neuron id
            marker_start = marker['time_interval_left_end'].iloc[0]
            marker_end = marker['time_interval_right_end'].iloc[-1]
            neuron_spike_A,time_len_A = population_spikecounts(neuron_id,marker_start,marker_end,30,bin)
            print(neuron_spike_A)
            print(neuron_spike_A.shape)
            print(time_len_A)

        elif region_name == 'Lobules IV-V':
            neuron_id = np.array(neurons.iloc[:, i].dropna()).astype(int)  #提取其中一个脑区的neuron id
            marker_start = marker['time_interval_left_end'].iloc[0]
            marker_end = marker['time_interval_right_end'].iloc[-1]
            neuron_spike_B,time_len_B = population_spikecounts(neuron_id,marker_start,marker_end,30,bin)
            print(neuron_spike_B.shape)
            print(time_len_B)

    TLCC(neuron_spike_A,neuron_spike_B)

main_function(neurons,treadmill)