"""
# coding: utf-8
@author: Yuhao Zhang
last updated: 08/27/2024
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
mice = '20230623_Syt2_conditional_tremor_mice4'
main_path = r'E:\xinchao\sorted neuropixels data\useful_data\20230623_Syt2_conditional_tremor_mice4\data'
fig_save_path = r'C:\Users\zyh20\Desktop\ET_data analysis\manifold\20230623_Syt2_conditional_tremor_mice4'
dis_save_path = r'C:\Users\zyh20\Desktop\ET_data analysis\manifold\runstop_center_distance'

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
    #smooth data
    data2pca = pd.DataFrame(data2pca)
    rate = np.sqrt(data2pca/0.1)
    #对数据做均值  默认: window=50  min_periods=1  感觉改这些值影响不大，改firing的bin size影响较大
    rate = rate.rolling(window=50,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=2) 
    #reduce dimension
    ## PCA 由于是计算距离，因此必须使用没有伸缩变换的PCA来计算
    pca = PCA(n_components=3)
    redu_dim_data = pca.fit_transform(rate.values) 

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