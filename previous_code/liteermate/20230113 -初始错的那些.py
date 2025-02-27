"""
@Author: Yuhao Zhang
Last updated : 06/19/2023
Data from: Xinchao Chen
Data collected: 06/19/2023
"""

import torch
import numpy as np
import pandas as pd
import pynapple as nap
import pynacollada as pyna
import matplotlib.pyplot as plt
import scipy.io as io


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
matrix = io.loadmat('E:\chaoge\sorted neuropixels data\litermate\800to2200marker.mat')
marker_path = ''
identities = np.load('E:\chaoge\sorted neuropixels data\litermate\spike_clusters.npy') #存储neuron的编号id,对应phy中的第一列id
times = np.load('E:\chaoge\sorted neuropixels data\litermate\spike_times.npy')  #
channel = np.load('E:\chaoge\sorted neuropixels data\litermate\channel_positions.npy')
#n_spikes = identities[ identities == 11 ].size #统计该id的neuron的发放次数，对应phy中的n_spikes一列
#average_firingrates = n_spikes/t  #对应phy中的fr一列
#print(channel)
#print(identities)
#print(average_firingrates)
numpy_data = np.transpose(matrix)
#保存为numpy数组文件（.npy文件）
np.save('numpy_data.npy',numpy_data)

def marker():

    with open(marker_path) as file_name:
        array = np.loadtxt(file_name, delimiter=",")

    array=array.astype('int64')
    marker_raw=np.array([])
    marker=np.array([])

    #Binarization
    for i in range(len(array)):
        if array[i]>2:
            array[i]=1
        else:
            array[i]=0
    #Rising edge detection
    for m in range(len(array)):
        if array[m]-array[m-1]==1:
            marker = np.append(marker,m/10593)  #10593 is sample rate of marker of spikeGLX

    #push rod time +-1s
    marker_1s=np.empty([len(marker),2]) 
    for a in range(len(marker)):
        marker_1s[a,0]=marker[a]-0.5
        marker_1s[a,1]=marker[a]+0.5

    #remove start 1000s
    clean=np.array([])
    for e in range(len(marker)):
        if marker[e]>1000:
            clean=np.append(clean,marker[e])

    return clean

def marker_trialstart():
    a=np.array([283.562873599546,313.545257599498,433.562491199306,463.544780799258,583.562108799066,643.561804798970,823.561459198682,
       883.561249598586,1003.56101919839,1093.54366559825,1273.56031199796,1333.56010239787,1423.56002399772,1513.55966239758,1603.55958399743,1693.54326879729])
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

def firingrate_time(id,marker):

    # bin
    bin_width = 0.14
    duration = 30   #一个trial的时间，或你关注的时间段的长度
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
    # channel 211 to 231, neuron id 290 to 363
    # extract neuron id from channel 211 to 231
    # 54 neurons
    #co=[290,293,295,294,292,303,304,305,296,297,301,302,311,298,310,312,313,314,306,308,309,329,332,316,317,320,336,315,321,323,324,325,331,334,335,344,326,327,328,330,333,338,341,342,343,345,339,346,337,357,358,359,360,363]
    go=[293,294,295,296,297,300,306,308,315,316,317,320,325,337,338,339,340,350,351,352,365,368]
    
    #combine neurons
    '''
    popu=np.array(firingrate_time(go[0],marker_trialstart())[0])
    for i in range(1,len(go)):
        popu = np.vstack((popu,firingrate_time(go[i],marker_trialstart())[0]))
    '''
    #combine neurons and trials
    for j in range(0,15):
        for i in range(0,len(go)):
            if i == 0 and j == 0:
                popu_trials=np.array(firingrate_time(go[0],marker_trialstart())[0])
            popu_trials = np.vstack((popu_trials,firingrate_time(go[i],marker_trialstart())[j]))
    
    return popu_trials

def interp_helper(values, num=50, kind='quadratic'):
    interp_i = np.linspace(min(values), max(values), num)

    return interp1d(np.linspace(min(values), max(values), len(values)), values, kind=kind)(interp_i)


def manifold(data):
    bin_size=0.1
    count = data
    count = pd.DataFrame(count)
    rate = np.sqrt(count/bin_size)
    rate = rate.rolling(window=50,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=2)

    projection = Isomap(n_components = 3, n_neighbors = 21).fit_transform(rate.values)
    print(projection)
    x=projection[:,0]
    y=projection[:,1]
    z=projection[:,2]

    '''
    fig = plt.figure()   
    ax = fig.gca(projection='3d')
    ax.plot(projection[:,0],projection[:,1],projection[:,2],label='Essential Tremor Neural Manifold')  #分别取三列的值作为x,y,z的值
    ax.legend()
    plt.show()
    '''
    x_new, y_new, z_new = (interp_helper(i,800) for i in (x, y, z))


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    zmax = np.array(z_new).max()
    zmin = np.array(z_new).min()

    for i in range(len(z_new) - 1):
        ax.plot(x_new[i:i + 2], y_new[i:i + 2], z_new[i:i + 2],
                color=plt.cm.jet(int((np.array(z_new[i:i + 2]).mean() - zmin) * 255 / (zmax - zmin))))
    
    plt.title('Essential Tremor Neural Manifold')
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