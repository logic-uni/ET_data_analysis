import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
import warnings

data_path = '/data2/zhangyuhao/xinchao_data/Givenme/1423_15_control-Day1-1CVC-FM_g0'
bin = 1
fs = 30000
Marker = pd.read_csv(data_path+'/Behavior/marker.csv')
identities = np.load(data_path+'/Sorted/kilosort4/spike_clusters.npy') # time series: unit id of each spike
times = np.load(data_path+'/Sorted/kilosort4/spike_times.npy')  # time series: spike time of each spike

def singleneuron_spiketimes(id):
    x = np.where(identities == id)
    y=x[0]
    #y = np.where(np.isin(identities, id))[0]
    spike_times=np.empty(len(y))
    for i in range(0,len(y)):
        z=y[i]
        spike_times[i]=times[z]/fs
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
            spike_times[m,i]=times[z]/fs
    return spike_times

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

def firingrate_time(id,marker,duration):
    # bin
    pre_time = 0
    post_time = duration
    bins = np.arange(pre_time, post_time+bin,bin)  # bin_width默认 0.14
    # histograms
    histograms=spike_counts(
        singleneuron_spiketimes(id),
        bin_edges=bins,
        movement_start_time=marker,
        )
    return histograms

def eachtrial_average_firingrate(histograms,bin_width):
    firing_rate=histograms.mean(1)/bin_width
    print(firing_rate)
    sio.savemat('/firing_rate/20230414/fir_%d.mat'%id, {'fir_%d'%id:firing_rate}) #存成matlab格式，方便后续辨识传递函数
    return firing_rate

def firingrate_shortime(id,marker):
    # bin
    bin_width = 0.05
    duration = 0.5   #一个trial的时间，或你关注的时间段的长度
    pre_time = -0.3
    post_time = duration
    bins = np.arange(pre_time, post_time+bin_width, bin_width)  
    # histograms
    histograms=spike_counts(
        singleneuron_spiketimes(id),
        bin_edges=bins,
        movement_start_time=marker,
        )
    print(histograms)
    return histograms

def population_spikecounts(neuron_id,marker_start,marker_end,Artificial_time_division):  
    #这里由于allen的spike counts函数是针对视觉的，因此对trial做了划分，必要trialmarker作为参数，因此这里分假trial，再合并
    #Artificial_time_division是把整个session人为划分为一个个时间段trial
    #bin是对firing rate的滑窗大小，单位s
    marker=np.array(range(int(marker_start),int(marker_end)-int(marker_end)%Artificial_time_division,Artificial_time_division))
    #get a 2D matrix with neurons, trials(trials contain times), trials and times are in the same dimension
    for j in range(len(neuron_id)): #第j个neuron
        #每个neuron的tials水平append
        for i in range(len(marker)):
            if i == 0:
                one_neruon = firingrate_time(neuron_id[j],marker,Artificial_time_division)[0]
            else:
                trail = firingrate_time(neuron_id[j],marker,Artificial_time_division)[i]
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

def manifold_dynamic(redu_dim_data,stage):  #动态流形，无时间区间颜色标记，用于trial_average，只需输入降维后的，无需marker
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    ax.set_title(f"Essential Tremor Manifold, {stage}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.grid(True)
    plt.ion()  # interactive mode on!!!! 很重要,有了他就不需要plt.show()了
    for j in range(0,len(redu_dim_data)):
        ax.plot3D(redu_dim_data[j:j+2,0],redu_dim_data[j:j+2,1],redu_dim_data[j:j+2,2],'blue')
        plt.pause(0.01)

def manifold_dynamic_colored_intervals(redu_dim_data,marker,time_len_int_aft_bin,region_name,redu_method):  #动态流形，时间区间颜色标记
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
        left=int(marker['time_interval_left_end'].iloc[i]/fr_bin)
        right=int(marker['time_interval_right_end'].iloc[i]/fr_bin)
        for j in range(left,right):
            if marker['run_or_stop'].iloc[i] == 1:
                ax.plot3D(redu_dim_data[j:j+2,0],redu_dim_data[j:j+2,1],redu_dim_data[j:j+2,2],colors[int(velocity_level[p])])
            else:
                ax.plot3D(redu_dim_data[j:j+2,0],redu_dim_data[j:j+2,1],redu_dim_data[j:j+2,2],'blue')
            plt.pause(0.01)
        if marker['run_or_stop'].iloc[i] == 1:
            p=p+1
    end_inter_start=int(marker['time_interval_left_end'].iloc[-1]/fr_bin)
    for m in range(end_inter_start,time_len_int_aft_bin):
        ax.plot3D(redu_dim_data[m:m+2,0],redu_dim_data[m:m+2,1],redu_dim_data[m:m+2,2],'blue')

'''
def manifold_fixed_colored_intervals(X_isomap,marker,time_len_int_aft_bin): 
    colors=[None] * time_len_int_aft_bin
    for i in range(0,len(marker['run_or_stop'])-1):
        t_left_withbin=int(marker['time_interval_left_end'].iloc[i]/fr_bin)
        t_right_withbin=int(marker['time_interval_right_end'].iloc[i]/fr_bin)
        if marker['run_or_stop'].iloc[i] == 1:
            colors[t_left_withbin:t_right_withbin] = ['red'] * (t_right_withbin-t_left_withbin)
        else:
            colors[t_left_withbin:t_right_withbin] = ['blue'] * (t_right_withbin-t_left_withbin)

    end_inter_start=int(marker['time_interval_left_end'].iloc[-1]/fr_bin)
    colors[end_inter_start:time_len_int_aft_bin] = ['blue'] * (time_len_int_aft_bin-end_inter_start)
    print(len(colors))
    manifold_fixed(X_isomap,colors)
'''

#流形动画
marker_start = marker['time_interval_left_end'].iloc[0]
marker_end = marker['time_interval_right_end'].iloc[-1]
data,time_len = population_spikecounts(neuron_id,marker_start,marker_end,30,fr_bin)
data2pca_each_trail=data.T
redu_dim_data_ISOMAP=reduce_dimension_ISOMAP(data2pca_each_trail,0.1,region_name,stage='all_session')
manifold_dynamic_colored_intervals(redu_dim_data_ISOMAP,marker,fr_bin,int(time_len),region_name,redu_method='ISOMAP')

run_redu_dim_aver=reduce_dimension(run2pca,0.1,region_name,stage='Run')
#manifold_dynamic(run_redu_dim_aver,'Run')
stop_redu_dim_aver=reduce_dimension(stop2pca,0.1,region_name,stage='Stop')
#manifold_dynamic(stop_redu_dim_aver,'Stop') 