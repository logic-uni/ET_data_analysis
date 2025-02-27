"""
@Author: Yuhao Zhang
Last updated : 07/04/2023
Data from: Xinchao Chen
Data collected: 01/13/2023
"""

import torch
import numpy as np
import pandas as pd
import pynapple as nap
import pynacollada as pyna
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import Isomap
from matplotlib.colors import hsv_to_rgb
from mpl_toolkits.axes_grid1 import make_axes_locatable
from math import log
import seaborn as sns
import warnings
import scipy.io as sio
np.set_printoptions(threshold=np.inf)

from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(torch.__version__)
#print(torch.cuda.is_available())
#print(torch.version.cuda)

# Experiment info
t=3041.407
sample_rate=30000 #spikeGLX neuropixel sample rate
marker_path = 'E:/chaoge/sorted neuropixels data/litermate/test.csv'
identities = np.load('E:\chaoge\sorted neuropixels data\litermate\spike_clusters.npy') #存储neuron的编号id,对应phy中的第一列id
times = np.load('E:\chaoge\sorted neuropixels data\litermate\spike_times.npy')  #
channel = np.load('E:\chaoge\sorted neuropixels data\litermate\channel_positions.npy')
#n_spikes = identities[ identities == 11 ].size #统计该id的neuron的发放次数，对应phy中的n_spikes一列
#average_firingrates = n_spikes/t  #对应phy中的fr一列
#print(channel)
#print(identities)
#print(average_firingrates)

def marker():

    #delete "sec"
    mar = pd.read_csv(marker_path)
    array = mar.to_numpy()
    time_mar=[]
    for j in range(len(array)):
        c=str(array[j,0])
        c = float(c.replace(" sec", ""))
        array[j,0]=c

    #Binarization
    for i in range(len(array)):
        if array[i,1]>2:
            array[i,1]=1
        else:
            array[i,1]=0

    #accumulate
    po=0
    for n in range(len(array)):
        po=po+array[n,1]
        array[n,1]=po

    trial_start=[]
    #detect trials start time point
    for i in range(len(array)-1):
        if array[i+1,1]-array[i,1]>0 and array[i,1]-array[i-60,1]==0:
            trial_start.append(array[i,0])
    trial_end=[]
    #detect trials end time point
    for i in range(len(array)-60):
        if array[i+60,1]-array[i,1]==0 and array[i,1]-array[i-1,1]>0:
            trial_end.append(array[i,0])

    print(trial_start)
    print(trial_end)
    '''
    # +-1s 
    marker_1s=np.empty([len(marker),2]) 
    for a in range(len(marker)):
        marker_1s[a,0]=marker[a]-0.5
        marker_1s[a,1]=marker[a]+0.5
    '''
    return array


def find_turning_points_one_diff(data, threshold):
    turning_points = []
    diffs = np.diff(data)
    for i in range(1, len(diffs)):
        if diffs[i-1] * diffs[i] < 0 and abs(diffs[i]) > threshold:
            turning_points.append(i)
    return turning_points

def marker_trialstart():
    #a=np.array([905.0,1315.0,1505.0,1915.0,2105.0])
    a=np.array([105.0,515.0,705.0,1115.0,1305.0])
    return a

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

# plot single neuron spike train
def raster_plot_singleneuron(spike_times):
    y=np.empty(len(spike_times))      
    plt.plot(spike_times,y, '|', color='gray') 
    plt.title('neuron 15') 
    plt.xlabel("time") 
    plt.xlim(0,t)
    plt.show()

# plot neurons around id spike train
def raster_plot_neurons(spike_times,id): 
    y = np.zeros((5, len(spike_times[0])))
    for i in range(0,5):
        y[i,:]=id+i
        plt.plot(spike_times[i] , y[i], '|', color='gray') 
    plt.title('spike train') 
    plt.xlabel("time")
    plt.ylabel("unit id")  
    plt.xlim(500,560)
    plt.ylim(id-1,id+5)
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
    bin_width = 0.001
    duration = 20  #一个trial的时间，或你关注的时间段的长度
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

def firingrate_time(id,marker,duration):

    # bin
    bin_width = 0.14
    pre_time = 0
    post_time = duration
    bins = np.arange(pre_time, post_time+bin_width, bin_width)  
    # histograms
    histograms=spike_counts(
        singleneuron_spiketrain(id),
        bin_edges=bins,
        movement_start_time=marker,
        )
    #print(histograms)

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

# PETH: peri-event time histogram  事件周围时间直方图
def PETH_singleneuron(firing_rate,duration):

    plt.rcParams['xtick.direction'] = 'in'#将x轴的刻度线方向设置向内
    plt.rcParams['ytick.direction'] = 'in'#将y轴的刻度线方向设置向内
    plt.plot(firing_rate)
    plt.axvspan(0, duration, color='gray', alpha=0.1)
    plt.xlim((-1, 50))
    plt.xticks(np.arange(-1,50,1))
    plt.ylabel('Firing rate (spikes/s)')
    plt.xlabel('Time (s)')
    plt.title('Neuron %d, Push rod'%id)
    plt.show()
    
def Pattern_Entropy(data,id):

    # about bin 1 bit = 1 msec 
    # Statistics pattern all trials
    result_dic={}
    for j in range(0,len(data)):
        trial=data[j]  # get a trial
        for i in range(0,len(trial)-len(trial)%8,8):  # delete end bits that can't be divide by 8
            a = np.array(trial[i:i+8])                # slice into 8 bit,  1 byte(1字节)(1B) = 8 bit(8比特)(8位二进制)；1KB = 1024B; 1MB = 1024KB; 1GB = 1024MB
            str1 = ''.join(str(z) for z in a)         # array to str
            if str1 not in result_dic:                # use dic to statistic, key = str, value = number of times
                result_dic[str1]=1
            else:
                result_dic[str1]+=1

    '''
    #delete pattern name contain number > 1 and probability so small that can ignore
    str2='2'
    for i in list(result_dic.keys()):
        if str2 in i:
            del result_dic[i]
    '''

    #compute probability
    total=sum(result_dic.values())
    p={k: v / total for k, v in result_dic.items()}
    del result_dic['00000000']
    total_del0=sum(result_dic.values())
    p_del0={k: v / total_del0 for k, v in result_dic.items()}
    
    '''
    #sorted keys:s
    s0=['00000000']
    s1=[]
    s2=[]
    for i in p.keys():
        if i.count('1')==1:
            s1.append(i)
        if i.count('1')>1:
            s2.append(i)
    s1=sorted(s1)
    s2=sorted(s2)
    s=s0+s1+s2
    sort_p = {key: p[key] for key in s}
    print(sort_p)
    '''

    #del 0 sorted keys:s
    s1=[]
    s2=[]
    for i in p_del0.keys():
        if i.count('1')==1:
            s1.append(i)
        if i.count('1')>1:
            s2.append(i)
    s1=sorted(s1)
    s2=sorted(s2)
    s=s1+s2
    sort_p = {key: p_del0[key] for key in s}
    print(sort_p)
    
    # information entropy
    h=0
    for i in p:
        h = h - p[i]*log(p[i],2)
    print('Shannon Entropy=%f'%h)

    #plot
    x=list(sort_p.keys())
    y=list(sort_p.values())

    plt.bar(x, y)
    plt.title('Encoding pattern distribution, Neuron id %d'%id, fontsize=16)
    plt.xticks(x, rotation=90, fontsize=10)
    plt.yticks(fontsize=16)
    #plt.ylim(0,0.08)
    plt.ylabel("Probability of pattern", fontsize=16)
    plt.show()
    
def PETH_heatmap_1(data): #未调试
    mean_histograms = data.mean(dim="stimulus_presentation_id")
    print(mean_histograms)

    # plot
    fig, ax = plt.subplots(figsize=(8, 8))
    c = ax.pcolormesh(
        mean_histograms["time_relative_to_stimulus_onset"], 
        np.arange(mean_histograms["unit_id"].size),
        mean_histograms, 
        vmin=0,
        vmax=1
    )
    plt.colorbar(c) 
    ax.set_ylabel("unit", fontsize=24)
    ax.set_xlabel("time relative to movement onset (s)", fontsize=24)
    ax.set_title("PSTH for units", fontsize=24)
    plt.show()

def PETH_heatmap_2(data,id):  #已调试

    data_minus_mean=data-np.mean(data)
    # plot
    fig, ax = plt.subplots(figsize=(12, 12))
    div = make_axes_locatable(ax)
    cbar_axis = div.append_axes("right", 0.2, pad=0.05)
    img = ax.imshow(
        data_minus_mean, 
        extent=(-2,30,0,len(data)),  #前两个值是x轴的时间范围，后两个值是y轴的值
        interpolation='none',
        aspect='auto',
        vmin=2, 
        vmax=20 #热图深浅范围
    )
    plt.colorbar(img, cax=cbar_axis)

    cbar_axis.set_ylabel('Spike counts', fontsize=20)
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.set_ylabel('Trials', fontsize=20)
    reltime = np.arange(-2, 30, 1)
    ax.set_xticks(np.arange(-2, 30, 1))
    ax.set_xticklabels([f'{mp:1.3f}' for mp in reltime[::1]], rotation=45)
    ax.set_xlabel('Time(s) Move: 0s', fontsize=20)
    ax.set_title('Spike counts Neuron id %d'%id, fontsize=20)
    plt.show()

def PETH_heatmap_shorttime(data,id):

    t0=-0.3
    t1=0.5
    #data_logtrans=np.log2(data+1)
    #data_minus_mean=data_logtrans-2.2
    data_minus_mean=data-np.median(data)

    print(data_minus_mean)

    # plot
    fig, ax = plt.subplots(figsize=(12, 12))
    div = make_axes_locatable(ax)
    cbar_axis = div.append_axes("right", 0.2, pad=0.05)
    img = ax.imshow(
        data_minus_mean, 
        extent=(t0,t1,0,len(data)),  #前两个值是x轴的时间范围，后两个值是y轴的值
        interpolation='none',
        aspect='auto',
        vmin=0, 
        vmax=3 #热图深浅范围
    )
    plt.colorbar(img, cax=cbar_axis)

    cbar_axis.set_ylabel('spike count', fontsize=20)
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.set_ylabel('Trials', fontsize=20)
    reltime = np.arange(t0,t1, 0.05)
    ax.set_xticks(np.arange(t0,t1, 0.05))
    ax.set_xticklabels([f'{mp:1.3f}' for mp in reltime[::1]], rotation=45,fontsize=12)
    ax.set_xlabel('Time(s) Move: 0s', fontsize=20)
    ax.set_title('Spike Counts Neuron id %d'%id, fontsize=20)
    plt.show()

def InfoPlot():
    x=['PC d:120','PC d:180','PC d:280','PC d:400','IPN d:1580','IPN d:1820','IPN d:1900','IPN d:1960']
    y=[2.3,3.3,3.6,2.8,0.5,0.5,0.3,0.2]

    plt.bar(x, y)
    plt.title('Quantities of information', fontsize=16)
    plt.xticks(x, fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel("Shannon entropy", fontsize=16)
    plt.show()

def population_spikecounts():

    #marker=np.array([105.0,515.0,705.0,1115.0,1305.0])  #拐点
    #marker=np.array([225,315,405,585,765,855,945,1035,1175,1355]) # run:225,315,405,765,855,945,1355  rest:585,1175
    #marker=np.array([135,225,315,405,735,825,915,1005]) #plot 1
    #marker=np.array([545,635,1145,1235]) #plot 2
    #marker=np.array([135,225,315,405,545,635,735,825,915,1005,1145,1235])
    #marker=np.array([165,225,285,345,405,515,575,635,705,765,825,885,945,1005])
    #marker=np.array([165,225,285,345,405,535,595,655,715,775,835,895,955,1015])
    m1=np.array(range(0,90,30))  #3 rest
    m2=np.array(range(115,505,30))  # 13 run
    m3=np.array(range(540,690,30))  # 5 rest
    m4=np.array(range(720,1080,30))  # 12 run
    m5=np.array(range(1120,1300,30)) # 6 rest
    m6=np.array(range(1310,1400,30))  # 3 run
    marker=np.concatenate((m1,m2,m3,m4,m5,m6))
    print(marker)
    print(len(marker))
    # 26 neurons
    neuron_id=np.array([402,403,404,405,406,407,408,411,412,416,417,418,420,421,424,426,428,429,431,439,440,441,442,444,445,447])
    
    #get a 2D matrix with neurons, trials(trials contain times), trials and times are in the same dimension
    for j in range(len(neuron_id)): #第j个neuron

        #每个neuron的tials水平append
        for i in range(len(marker)):
            if i == 0:
                one_neruon = firingrate_time(neuron_id[j],marker,30)[0]
            else:
                trail = firingrate_time(neuron_id[j],marker,30)[i]
                one_neruon = np.append(one_neruon, trail)

        if j == 0:
            neurons = one_neruon
        else:
            neurons = np.vstack((neurons, one_neruon))
    
    neurons_topca=neurons.T


    #combine neurons
    '''
    popu=np.array(firingrate_time(go[0],marker_trialstart())[0])
    for i in range(1,len(go)):
        popu = np.vstack((popu,firingrate_time(go[i],marker_trialstart())[0]))
    
    #combine neurons and trials
    for j in range(0,15):
        for i in range(0,len(go)):
            if i == 0 and j == 0:
                popu_trials=np.array(firingrate_time(go[0],marker_trialstart())[0])
            popu_trials = np.vstack((popu_trials,firingrate_time(go[i],marker_trialstart())[j]))
    '''
    print(neurons_topca.shape)
    return neurons_topca

def interp_helper(values, num=50, kind='quadratic'):
    interp_i = np.linspace(min(values), max(values), num)

    return interp1d(np.linspace(min(values), max(values), len(values)), values, kind=kind)(interp_i)

def manifold(data):
    #smooth data
    bin_size=0.1
    count = data
    count = pd.DataFrame(count)
    rate = np.sqrt(count/bin_size)
    rate = rate.rolling(window=50,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=2) #对数据做均值

    #reduce dimension
    X_isomap = Isomap(n_components = 3, n_neighbors = 21).fit_transform(rate.values)
    #X_tsne = TSNE(n_components=3,random_state=21,perplexity=20).fit_transform(rate.values)
    #X_pca = PCA(n_components=3).fit_transform(rate.values) 
    '''
    #plot dynamic 3D trajectory
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_title("Litermate Manifold")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")

    x_track=np.zeros((1,3))
    x_track[0,0]=X_isomap[0,0]
    x_track[0,1]=X_isomap[0,1]
    x_track[0,2]=X_isomap[0,2]
    plt.grid(True)
    plt.ion()  # interactive mode on!!!! 很重要,有了他就不需要plt.show()了

    for i in range(6032,6790):
        #if 1358 < i < 2716:
        ax.plot3D(x_track[:,0], x_track[:,1], x_track[:,2], 'blue')
        
        #elif 238 < i < 476:
            #ax.plot3D(x_track[:,0], x_track[:,1], x_track[:,2], 'red')
        #elif 476 < i < 714:
            #ax.plot3D(x_track[:,0], x_track[:,1], x_track[:,2], 'gray')
        #else:
            #ax.plot3D(x_track[:,0], x_track[:,1], x_track[:,2], 'yellow')
        
        x_track_s=[X_isomap[i,0],X_isomap[i,1],X_isomap[i,2]]
        x_track = np.vstack((x_track, x_track_s))
        plt.pause(0.001)
    '''
    #plot 3d scatter
    ax = plt.subplot(projection = '3d')
    #m=['#00FFFF','#808080','#008000','#000080','#808000','#FFA500','#FF0000','#800080','#FFD700','#FFFF00','#FF69B4','#800000']  #plot 1
    #m=['#00FFFF','#808080','#008000','#000080']  #plot 2
    m=['#FFD700','#FFD700','#FFD700','#FFD700','#FFD700','#008000','#008000','#008000',
       '#00FFFF','#00FFFF','#00FFFF','#00FFFF','#00FFFF','#00FFFF']  #plot 3
    # #00FFFF cyan, #FFD700 gold, #808080 gray, #000080 navy,#008000 green, #808000 olive, #FFA500 orange, #FF0000 red, #800080 purple, #FFFF00 yellow, #FF69B4 hotpink,#800000 maroon
    for d in range(0,9030,215):  #plot 1: 5144
        #p=int(d/215)
        ax.scatter(X_isomap[d:d+215,0],X_isomap[d:d+215,1],X_isomap[d:d+215,2], c='#FFD700')
        v_temp=fit_plane(X_isomap[d:d+215,0],X_isomap[d:d+215,1],X_isomap[d:d+215,2],'r')
        if d==0:
            v=v_temp
        else:
            v=np.vstack((v, v_temp))
    
    print(v)

    #fit plane
    '''
    v1=fit_plane(X_isomap[0:2145,0],X_isomap[0:2145,1],X_isomap[0:2145,2],'b')
    v2=fit_plane(X_isomap[2145:3432,0],X_isomap[2145:3432,1],X_isomap[2145:3432,2],'r')
    v3=fit_plane(X_isomap[3432:6006,0],X_isomap[3432:6006,1],X_isomap[3432:6006,2],'y')
    #fit_plane(X_isomap[6430:7716,0],X_isomap[6430:7716,1],X_isomap[6430:7716,2],'g') 
    '''

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
    '''
    #plot fixed 3D trajectory
    fig = plt.figure()   
    ax = fig.add_subplot(projection='3d')
    #ax.scatter(X_isomap[:,0],X_isomap[:,1],X_isomap[:,2],label='Essential Tremor Neural Manifold')  #分别取三列的值作为x,y,z的值
    for i in range(0,len(X_isomap)):
        if i < 1358:
            ax.plot(X_isomap[:,0],X_isomap[:,1],X_isomap[:,2], color='blue')
        elif 1358 < i < 2716:
            ax.plot(X_isomap[:,0],X_isomap[:,1],X_isomap[:,2], color='red')
        elif 2716 < i < 4074:
            ax.plot(X_isomap[:,0],X_isomap[:,1],X_isomap[:,2], color='gray')
        else:
            ax.plot(X_isomap[:,0],X_isomap[:,1],X_isomap[:,2], color='yellow')
    ax.legend()

    
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
    '''

    plt.show()
    return X_isomap

def fit_plane(xs,ys,zs,color_name):
    ax = plt.subplot(projection = '3d')
    # do fit
    tmp_A = []
    tmp_b = []
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

    # Or use Scipy
    # from scipy.linalg import lstsq
    # fit, residual, rnk, s = lstsq(A, b)

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
        if i < 3 or 16 < i < 21 or 33 < i < 39:
            ax.quiver(0,0,0,normal_vector[i,0],normal_vector[i,1],normal_vector[i,2],arrow_length_ratio=0.1,color='b',length=5, normalize=True)
            a=a+normal_vector[i]
            a_count=a_count+1
        if 3 < i < 16 or 21 < i < 33 or 39 < i < 42:
            ax.quiver(0,0,0,normal_vector[i,0],normal_vector[i,1],normal_vector[i,2],arrow_length_ratio=0.1,color='r',length=5, normalize=True)
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

#manifold(population_spikecounts())

    
#PETH_heatmap_2(firingrate_time(0,marker_trialstart()),0)
#PETH_heatmap_shorttime(firingrate_shortime(1177,marker()),1177)

'''
    firing_rate=histograms.mean(1)/bin_width
    if k == 0: 
        temp=firing_rate
    else:
        temp=temp+firing_rate

av_firing_rate=temp/len(t)

print(av_firing_rate)
sio.savemat('/firing_rate/20230510/firing_rate_1196.mat', {'firing_rate_1196':firing_rate})
    
plt.plot(av_firing_rate)
'''

#singleneuron_spiketrain(1196)
#print(singleneuron_spiketrain(1196))
#print(binary_spiketrain(426,marker_trialstart()))
#Pattern_Entropy(binary_spiketrain(426,marker_trialstart()),426)
#raster_plot_neurons(neurons_spiketrain(1196),1196)
#raster_plot_singleneuron(singleneuron_spiketrain(1196))
#InfoPlot()