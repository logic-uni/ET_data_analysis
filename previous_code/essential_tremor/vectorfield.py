"""
@Author: Yuhao Zhang
Last updated : 05/26/2023
Data from: Xinchao Chen
Data collected: 05/23/2023
"""
import torch
import numpy as np
import pandas as pd
import pynapple as nap
import pynacollada as pyna
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns 
import csv
import os
from itertools import count
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
from sklearn.datasets import load_iris,load_digits
from sklearn.decomposition import PCA
from matplotlib.colors import hsv_to_rgb
from mpl_toolkits.axes_grid1 import make_axes_locatable
from math import log
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from scipy import interpolate

from scipy.interpolate import griddata
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
marker_path = ''
identities = np.load('E:/Data/chaogedata/20230523/sorted/spike_clusters.npy') #存储neuron的编号id,对应phy中的第一列id
times = np.load('E:/Data/chaogedata/20230523/sorted/spike_times.npy')  #
channel = np.load('E:/Data/chaogedata/20230523/sorted/channel_positions.npy')
#n_spikes = identities[ identities == 11 ].size #统计该id的neuron的发放次数，对应phy中的n_spikes一列
#average_firingrates = n_spikes/t  #对应phy中的fr一列
#print(channel)
#print(identities)
#print(average_firingrates)

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

def  binary_spiketrain(id,marker):  #each trial
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

def firingrate_time(id,marker,duration):

    # bin
    #bin_width = 0.5  # Coarse-Grained
    bin_width = 0.14   # fine grained
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

    # Slice data to 8 bit
    res=[]
    for j in range(0,len(data)):
        trial=data[j]  # get a trial
        for i in range(0,len(trial)-len(trial)%8,8):  # delete end bits that can't be divide by 8
            a = list(trial[i:i+8])               # slice into 8 bit,  1 byte(1字节)(1B) = 8 bit(8比特)(8位二进制)；1KB = 1024B; 1MB = 1024KB; 1GB = 1024MB
            strnull = ''
            for item in a:
                strnull = strnull + str(item)
            res.append(strnull)

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

    #save to csv
    my_list = [[key, value] for key, value in sort_p.items()]
    with open('C:/Users/zyh20/Desktop/csv/output.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(my_list)

    '''
    #plot
    x=list(sort_p.keys())
    y=list(sort_p.values())

    plt.bar(x, y)
    plt.title('Encoding pattern distribution, Neuron id %d'%id, fontsize=16)
    plt.xticks(x, rotation=90, fontsize=10)
    plt.yticks(fontsize=16)
    #plt.ylim(0,0.08)
    plt.ylabel("Probability of pattern", fontsize=16)

    
    #MSE拟合曲线
    x_list=np.arange(len(x))
    print(x_list)
    pfit = np.polyfit(x_list,y,15)
    trendline = np.polyval(pfit,x_list)
    print(pfit)
    plt.plot(x_list,trendline,'r')
    print('y = %f x^5 + %f x^4 + %f x^3 + %f x^2 + %f x + %f' %(pfit[0],pfit[1],pfit[2],pfit[3],pfit[4],pfit[5]))
    plt.show()
    

    #MLE拟合曲线
    mu, std = norm.fit(y)
    mu, std = norm.fit_loc_scale(y)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)

    title = "Fit results: mu = {:.2f},  std = {:.2f}".format(mu, std)
    plt.title(title)
    plt.show()
    '''
    result = sort_p.keys()
    de = list(result)
    
    #convert list to array
    
    arr = [i for i in res if i != '00000000']    #delete '00000000' in res
    ress=np.array(arr)   #原始数据切片为八位一组的字符串组成的数组
    de=np.array(de)     #类别库所有模式的类别

    #convert res to num
    num=ress
    for i in range(len(ress)):
        for j in range(len(de)):
            if ress[i]==de[j]:
                num[i]=j

    # covert num type from str to int
    nn=[]
    for i in range(len(num)):
        #nn.append(int(num[i]))
        nn.append(int(num[i]))
    print(nn)
    
    #GMM
    rn=np.array(nn)
    X = rn.reshape((len(rn), 1)) 

    #save to csv
    np.savetxt('C:/Users/zyh20/Desktop/csv/a.csv', X,fmt="%d", delimiter="," )
    
    # fit models with 1-10 components
    N = np.arange(1, 11)
    models = [None for i in range(len(N))]

    for i in range(len(N)):
        models[i] = GaussianMixture(N[i]).fit(X)

    # compute the AIC and the BIC
    AIC = [m.aic(X) for m in models]
    BIC = [m.bic(X) for m in models]

    fig = plt.figure(figsize=(5, 1.7))
    fig.subplots_adjust(left=0.12, right=0.97,
                    bottom=0.21, top=0.9, wspace=0.5)

    # plot 1: data + best-fit mixture
    ax = fig.add_subplot(131)
    M_best = models[np.argmin(AIC)]
    x = np.linspace(-6,10, 800)
    logprob = M_best.score_samples(x.reshape(-1, 1))
    responsibilities = M_best.predict_proba(x.reshape(-1, 1))
    pdf = np.exp(logprob)
    pdf_individual = responsibilities * pdf[:, np.newaxis]

    #ax.hist(X, 29, density=False,histtype='bar')
    ax.plot(x, pdf*1000, '-k')
    ax.plot(x, pdf_individual*1000, '--k')
    ax.text(0.04, 0.96, "Best-fit Mixture",
            ha='left', va='top', transform=ax.transAxes)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$p(x)$')


    # plot 2: AIC and BIC
    ax = fig.add_subplot(132)
    ax.plot(N, AIC, '-k', label='AIC')
    ax.plot(N, BIC, '--k', label='BIC')
    ax.set_xlabel('n. components')
    ax.set_ylabel('information criterion')
    ax.legend(loc=2)


    # plot 3: posterior probabilities for each component
    ax = fig.add_subplot(133)

    p = responsibilities
    p = p[:, (1, 0, 2)]  # rearrange order so the plot looks better
    p = p.cumsum(1).T

    ax.fill_between(x, 0, p[0], color='gray', alpha=0.3)
    ax.fill_between(x, p[0], p[1], color='gray', alpha=0.5)
    ax.fill_between(x, p[1], 1, color='gray', alpha=0.7)
    ax.set_xlim(-6, 6)
    ax.set_ylim(0, 1)
    ax.set_xlabel('$x$')
    ax.set_ylabel(r'$p({\rm class}|x)$')

    ax.text(-5, 0.3, 'class 1', rotation='vertical')
    ax.text(0, 0.5, 'class 2', rotation='vertical')
    ax.text(3, 0.3, 'class 3', rotation='vertical')

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
    #co=[290,293,295,294,292,303,304,305,296,297,301,302,311,298,310,312,313,314,306,308,309,329,332,316,317,320,336,315,321,323,324,325,331,334,335,344,326,327,328,330,333,338,341,342,343,345,339,346,337,357,358,359,360,363]
    #marker=np.array([283.562873599546,313.545257599498,433.562491199306,463.544780799258,583.562108799066,643.561804798970,823.561459198682,
       #883.561249598586,1003.56101919839,1093.54366559825,1273.56031199796,1333.56010239787,1423.56002399772,1513.55966239758,1603.55958399743,1693.54326879729])
    marker=np.array(range(283,1693,30))
    print(marker)
    print(len(marker))
    neuron_id=np.array([293,294,295,296,297,300,306,308,315,316,317,320,325,337,338,339,340,350,351,352,365,368])
    #marker=np.array([283.562873599546,433.562491199306,583.562108799066,823.561459198682,1003.56101919839,1273.56031199796,1423.56002399772,1603.55958399743,
    #313.545257599498,463.544780799258,643.561804798970,883.561249598586,1093.54366559825,1333.56010239787,1513.55966239758,1693.54326879729])
    '''
    #get a 3D matrix with neurons, trials, times
    for j in range(len(marker)): #第j个trial

        #第j个trial的neurons竖直堆叠
        for i in range(len(neuron_id)):
            if i == 0:
                one_trial = firingrate_time(neuron_id[0],marker,30)[j]
            else:
                neuron = firingrate_time(neuron_id[i],marker,30)[j]
                one_trial = np.vstack((one_trial, neuron))

        if j == 0:
            trials = one_trial
        else:
            trials = np.dstack((trials, one_trial))
    '''

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

    '''

    #get a 2D matrix with times, trials(trials contain neurons), trials and neurons are in the same dimension
    for j in range(len(marker)): #第j个trial
        #第j个trial的neurons竖直堆叠
        for i in range(len(neuron_id)):
            if i == 0:
                one_trial = firingrate_time(neuron_id[0],marker,30)[j]
            else:
                neuron = firingrate_time(neuron_id[i],marker,30)[j]
                one_trial = np.vstack((one_trial, neuron))
        
        if j == 0:
            trials = one_trial
        else:
            trials = np.vstack((trials, one_trial))
    

    # 第一次画的时候的代码，和上面这段画出来有细微差别
    for j in range(len(marker)):
        for i in range(len(neuron_id)):
            if i == 0 and j == 0:
                one_trial = firingrate_time(neuron_id[0],marker,30)[0]
            neuron = firingrate_time(neuron_id[i],marker,30)[j]
            one_trial = np.vstack((one_trial, neuron))
    
    '''

    #print(neuron_id.shape)
    #print(marker.shape)
    #print(one_trial.shape)
    #print(trials.shape)
    #print(trials[:,:,h])
    #print(trials[:,:,h].shape)

    print(neurons_topca.shape)
    return neurons_topca
    

def interp_helper(values, num=50, kind='quadratic'):
    interp_i = np.linspace(min(values), max(values), num)

    return interp1d(np.linspace(min(values), max(values), len(values)), values, kind=kind)(interp_i)


def redu_dim(data):
    #smooth data
    #bin_size=0.5  #Coarse-Grained
    bin_size=0.1  #fine-grained
    count = data
    count = pd.DataFrame(count)
    rate = np.sqrt(count/bin_size)
    #Coarse-Grained
    #rate = rate.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=2) #对数据做均值
    #fine-grained
    rate = rate.rolling(window=50,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=2) #对数据做均值

    #reduce dimension
    rd = Isomap(n_components = 2, n_neighbors = 21).fit_transform(rate.values)
    #rd = TSNE(n_components=2,random_state=21,perplexity=20).fit_transform(rate.values)
    #X_pca = PCA(n_components=3).fit_transform(rate.values) 

    return rd

def vectorfield(rd):

    xdata_raw=rd[:,0]
    ydata_raw=rd[:,1]
    xdata=np.clip(xdata_raw,-25,-10,out=None)*10
    ydata=np.clip(ydata_raw,-15,5,out=None)*10
    u=np.diff(xdata)
    u=np.append(u,u[-1])
    v=np.diff(ydata)
    v=np.append(v,v[-1])
    
    x_new = np.linspace(xdata.min(), xdata.max(), len(xdata)+100)
    y_new = np.linspace(ydata.min(), ydata.max(), len(ydata)+100)
    u_new = griddata((xdata, ydata), u, (x_new, y_new), method='nearest')
    v_new = griddata((xdata, ydata), v, (x_new, y_new), method='nearest')

    X,Y=np.meshgrid(x_new, y_new)
    U,V=np.meshgrid(u_new, v_new)

    plt.streamplot(X,Y,U,V,density=15,zorder=1)
    plt.show()
    
def vectorfield_1(rd):
    x=np.arange(-25, -10)
    y=np.arange(-15, 10)
    u=rd[:,0] 
    v=rd[:,1]

    plt.quiver(x,y,u,v,width = 0.005,scale = 100)

    #plt.scatter(x,y,color="b",s=0.05)
    
    plt.xlim(-25, -10)
    plt.ylim(-15, 10)
    plt.show()

def vectorfield_2(rd):

    x=rd[:,0] 
    y=rd[:,1]
    u=np.diff(x, n=1)
    v=np.diff(y, n=1)
    X, Y = np.meshgrid(x, y)
    Z=np.sqrt(u*u+v*v)
    '''
    #sort from small to large, may lose temporal information
    x=np.sort(xdata)
    y=np.sort(ydata)

    #interpolate
    xnew=np.linspace(x[0],x[-1],len(x))
    ynew=np.linspace(y[0],y[-1],len(y))
    X,Y=np.meshgrid(xnew, ynew)
    #f=interpolate.interp1d(x,y,kind='linear')
    #ynew=f(xnew)

    u=np.empty(len(xdata))
    v=np.empty(len(ydata))
    '''
    
    # 填充等高线
    plt.contourf(X, Y, Z, 20, cmap=plt.cm.hot)
    # 添加等高线
    C = plt.contour(X, Y, Z, 20)
    plt.clabel(C, inline=True, fontsize=12)
    # 显示图表
    plt.show()

def streams(xx,yy,u,v):
    x = np.linspace(xx.min(), xx.max(), len(xx))
    y = np.linspace(yy.min(), yy.max(), len(yy))

    xi, yi = np.meshgrid(x,y)

    #then, interpolate your data onto this grid:

    px = xx.flatten()
    py = yy.flatten()
    pu = u.flatten()
    pv = v.flatten()
    speed=np.sqrt(u*u+v*v)
    pspeed = speed.flatten()

    gu = griddata((px,py), pu, (xi,yi))
    gv = griddata((px,py), pv, (xi,yi))
    gspeed = griddata(zip(px,py), pspeed, (xi,yi))

    lw = 6*gspeed/np.nanmax(gspeed)
    #now, you can use x, y, gu, gv and gspeed in streamplot:

    plt.contour(xx,yy,speed, colors='k', alpha=0.4)
    plt.plot(xx,yy,'-k',alpha=0.3)
    plt.plot(xx.T,yy.T,'-k',alpha=0.3)
    plt.plot(xi,yi,'-b',alpha=0.1)
    plt.plot(xi.T,yi.T,'-b',alpha=0.1)
    plt.streamplot(x,y,gu,gv, density=2,
        linewidth=lw, color=gspeed, cmap=plt.cm.jet)
    
    plt.show()


def dict2csv(dic, filename):
    file = open(filename, 'w', encoding='utf-8', newline='')
    csv_writer = csv.DictWriter(file, fieldnames=list(dic.keys()))
    csv_writer.writeheader()
    for i in range(len(dic[list(dic.keys())[0]])):   # 将字典逐行写入csv
        dic1 = {key: dic[key][i] for key in dic.keys()}
        csv_writer.writerow(dic1)
    file.close()

#population_spikecounts()
vectorfield(redu_dim(population_spikecounts()))

#c=redu_dim(population_spikecounts())
#print(c[:,0])