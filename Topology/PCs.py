"""
# coding: utf-8
@author: Yuhao Zhang
last updated: 10/03/2024
data from: Xinchao Chen
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import pairwise_distances
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
import scipy.io as sio
np.set_printoptions(threshold=np.inf)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### path
mice = '20230623_Syt2_conditional_tremor_mice4'
main_path = r'E:\xinchao\sorted neuropixels data\useful_data\20230623_Syt2_conditional_tremor_mice4\data'
save_path = r'C:\Users\zyh20\Desktop\ET_data analysis\PC1\20230623_Syt2_conditional_tremor_mice4'

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

### manifold
def adjust_array(arr):
    if any(x < 0 for x in arr):
        min_val = min(arr)
        diff = -min_val
        arr = [x + diff for x in arr]
    return np.array(arr)

def reduce_dimension(count,bin_size,region_name): # 默认: 0.1 感觉改bin_size影响不大，改firing rate的bin size影响较大
    #smooth data
    count = pd.DataFrame(count)
    rate = np.sqrt(count/bin_size)
    #对数据做均值  默认: window=50  min_periods=1  感觉改这些值影响不大，改firing的bin size影响较大
    rate = rate.rolling(window=50,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=2) 
    ## PCA
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(rate.values)   #对应的是Explained variance
    explained_variance_ratio = pca.explained_variance_ratio_   #每个主成分所解释的方差比例
    explained_variance_sum = np.cumsum(explained_variance_ratio)  #计算累积解释方差比例
    #X_isomap = Isomap(n_components = 1, n_neighbors = 21).fit_transform(rate.values)  #对应的是Residual variance
    return X_pca,explained_variance_ratio

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

    return X_isomap
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

def compute(neuron_id,marker_start,marker_end,bin,region_name):
    data,time_len = population_spikecounts(neuron_id,marker_start,marker_end,30,bin)
    data2pca=data.T
    PC,EVR = reduce_dimension(data2pca,0.1,region_name)
    np.savetxt(save_path+f'\{region_name}\{region_name}_{marker_start}_{marker_end}_EVR.csv', EVR, delimiter=',')
    mat_dict = {
        'MV_PCs': np.column_stack((PC[:,0], PC[:,1], PC[:,2]))
    }
    sio.savemat(save_path+f'/{region_name}/{region_name}_{marker_start}_{marker_end}_PC.mat', mat_dict)

def main_function(neurons,marker):
    for i in range(neurons.shape[1]):  #遍历所有的脑区
        bin=1
        region_name = neurons.columns.values[i]
        if region_name == 'Lobule III':
            neuron_id = np.array(neurons.iloc[:, i].dropna()).astype(int)  #提取neuron id
            
            ## session
            marker_start = marker['time_interval_left_end'].iloc[0]
            marker_end = marker['time_interval_right_end'].iloc[-1]
            compute(neuron_id,marker_start,marker_end,bin,region_name)
            '''
            ### run trials
            ## 1. ET mice
            for trial in np.arange(1,len(marker['time_interval_left_end']),2):
                marker_start = marker['time_interval_left_end'].iloc[trial]
                if marker['time_interval_right_end'].iloc[trial] - marker_start > 29:
                    marker_end = marker_start+29
                    compute(neuron_id,marker_start,marker_end,bin,region_name)
            
            ## 2. Littermate 人为切分29s的trial
            for marker_start in np.arange(105,511,29):
                marker_end = marker_start + 29
                compute(neuron_id,marker_start,marker_end,bin,region_name)
            for marker_start in np.arange(705,1111,29):
                marker_end = marker_start + 29
                compute(neuron_id,marker_start,marker_end,bin,region_name)
            '''

main_function(neurons,treadmill)