"""
# coding: utf-8
@author: Yuhao Zhang
last updated: 04/22/2024
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
import csv
import os
import numpy as np
import scipy.stats as stats
import warnings
import scipy.io as sio
import scipy.stats as stats
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
from scipy.signal import hilbert, butter, filtfilt
from scipy.fftpack import fft,fftfreq,rfft,irfft,ifft
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d
from itertools import combinations
np.set_printoptions(threshold=np.inf)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Experiment info
sample_rate=30000 #spikeGLX neuropixel sample rate

file_directory=r'E:\chaoge\sorted neuropixels data\20230112-tremor\partly data\350_950sec'
identities = np.load(file_directory+'/spike_clusters.npy') #存储neuron的编号id,对应phy中的第一列id
times = np.load(file_directory+'/spike_times.npy')  #
channel = np.load(file_directory+'/channel_positions.npy')
neuron_info = file_directory+'/cluster_info.tsv'
neurons = pd.read_csv(file_directory+'/neuron_id_region_firingrate.csv', low_memory = False)#防止弹出警告
del neurons[neurons.columns[0]]
print(neurons)

# marker
def marker_for_without_nidqbin(): #before doing this, must export the ap.csv file
    # get matlab recording start time, by ap.bin file to csv file
    session_start = pd.read_csv(r'E:\chaoge\sorted neuropixels data\20230112-tremor\Ephys\20230112_338_1_tremor_female_stable_set_g0\20230112_338_1_tremor_female_stable_set_g0_t0.exported.imec0.ap.csv', low_memory = False,header=None, names=['level'])#防止弹出警告
    session_start = pd.DataFrame(session_start)
    session_start_time = session_start[session_start['level']>0].index[0] / sample_rate
    print(session_start_time)

    # get matlab run and stop marker, by matlab txt file to csv file
    matlab_marker = pd.read_csv(r'E:\chaoge\sorted neuropixels data\20230112-tremor\treadmill_move_stop.csv', low_memory = False,header=None, names=['Run_Stop_Time'])#防止弹出警告
    matlab_marker = pd.DataFrame(matlab_marker)
    matlab_marker = matlab_marker['Run_Stop_Time'].to_numpy()
    print(matlab_marker)

    # compute run stop time in electrophysiology
    marker_time=matlab_marker+session_start_time  #这里得到的是电生理raw data里的电生理时间
    marker_time = np.insert(marker_time, 0, 0)
    print(marker_time)
    return marker_time

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

def firingrate_time(id,marker,duration):
    # bin
    bin_width = 0.1  
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

#(1)希尔伯特变换求瞬时相位同步
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def Instantaneous_phase_synchrony(d1,d2,neuron_a,neuron_b):   #input d1,d2 is pandas.series
    lowcut  = .01
    highcut = .5
    fs = 30.
    order = 1
    y1 = butter_bandpass_filter(d1,lowcut=lowcut,highcut=highcut,fs=fs,order=order)
    y2 = butter_bandpass_filter(d2,lowcut=lowcut,highcut=highcut,fs=fs,order=order)

    al1 = np.angle(hilbert(y1),deg=False)
    al2 = np.angle(hilbert(y2),deg=False)
    phase_synchrony = 1-np.sin(np.abs(al1-al2)/2)
    N = len(al1)

    # 绘制结果
    f,ax = plt.subplots(3,1,figsize=(14,7),sharex=True)
    ax[0].plot(y1,color='r',label='y1')
    ax[0].plot(y2,color='b',label='y2')
    ax[0].legend(bbox_to_anchor=(0., 1.02, 1., .102),ncol=2)
    ax[0].set(xlim=[0,N], title='Filtered Timeseries Data')
    ax[1].plot(al1,color='r')
    ax[1].plot(al2,color='b')
    ax[1].set(ylabel='Angle',title='Angle at each Timepoint',xlim=[0,N])
    phase_synchrony = 1-np.sin(np.abs(al1-al2)/2)
    ax[2].plot(phase_synchrony)
    ax[2].set(ylim=[0,1.1],xlim=[0,N],title='Instantaneous Phase Synchrony',xlabel='Time',ylabel='Phase Synchrony')
    plt.tight_layout()
    plt.savefig(f'C:/Users/zyh20/Desktop/time_series_correlation/{neuron_a}_{neuron_b}_Instantaneous_phase_synchrony.png')
    
#(2)时间滞后互相关
def crosscorr(datax, datay, lag=0, wrap=False):
    """ Lag-N cross correlation. 
    Shifted data filled with NaNs 
    
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length

    Returns
    ----------
    crosscorr : float
    """
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else: 
        return datax.corr(datay.shift(lag))

def Time_lag_crosscorrelation(d1,d2,neuron_a,neuron_b): #input d1,d2 is pandas.series
    df = pd.DataFrame({'S1_Joy': d1, 'S2_Joy': d2})
    
    #时间滞后互相关
    d1 = df['S1_Joy']
    d2 = df['S2_Joy']
    seconds = 5
    fps = 30
    rs = [crosscorr(d1,d2, lag) for lag in range(-int(seconds*fps-1),int(seconds*fps))]
    offset = np.ceil(len(rs)/2)-np.argmax(rs)
    f,ax=plt.subplots(figsize=(14,3))
    ax.plot(rs)
    ax.axvline(np.ceil(len(rs)/2),color='k',linestyle='--',label='Center')
    ax.axvline(np.argmax(rs),color='r',linestyle='--',label='Peak synchrony')
    ax.set(title=f'Offset = {offset} frames\nS1 leads <> S2 leads',ylim=[.1,.31],xlim=[0,300], xlabel='Offset',ylabel='Pearson r')
    ax.set_xticklabels([int(item-150) for item in ax.get_xticks()])
    plt.legend()
    plt.savefig(f'C:/Users/zyh20/Desktop/time_series_correlation/{neuron_a}_{neuron_b}_Time_lag_crosscorrelation.png')

    # 加窗的时间滞后互相关
    seconds = 5
    fps = 30
    no_splits = 20
    samples_per_split = df.shape[0]/no_splits
    rss=[]
    for t in range(0, no_splits):
        d1 = df['S1_Joy'].loc[(t)*samples_per_split:(t+1)*samples_per_split]
        d2 = df['S2_Joy'].loc[(t)*samples_per_split:(t+1)*samples_per_split]
        rs = [crosscorr(d1,d2, lag) for lag in range(-int(seconds*fps-1),int(seconds*fps))]
        rss.append(rs)
    rss = pd.DataFrame(rss)
    f,ax = plt.subplots(figsize=(10,5))
    sns.heatmap(rss,cmap='RdBu_r',ax=ax)
    ax.set(title=f'Windowed Time Lagged Cross Correlation',xlim=[0,300], xlabel='Offset',ylabel='Window epochs')
    ax.set_xticklabels([int(item-150) for item in ax.get_xticks()]);
    plt.savefig(f'C:/Users/zyh20/Desktop/time_series_correlation/{neuron_a}_{neuron_b}_Windowed_Time_lag_crosscorrelation.png')

    # 滑动窗口时间滞后互相关
    seconds = 5
    fps = 30
    window_size = 300 #样本
    t_start = 0
    t_end = t_start + window_size
    step_size = 30
    rss=[]
    while t_end < 5400:
        d1 = df['S1_Joy'].iloc[t_start:t_end]
        d2 = df['S2_Joy'].iloc[t_start:t_end]
        rs = [crosscorr(d1,d2, lag, wrap=False) for lag in range(-int(seconds*fps-1),int(seconds*fps))]
        rss.append(rs)
        t_start = t_start + step_size
        t_end = t_end + step_size
    rss = pd.DataFrame(rss)

    f,ax = plt.subplots(figsize=(10,10))
    sns.heatmap(rss,cmap='RdBu_r',ax=ax)
    ax.set(title=f'Rolling Windowed Time Lagged Cross Correlation',xlim=[0,300], xlabel='Offset',ylabel='Epochs')
    ax.set_xticklabels([int(item-150) for item in ax.get_xticks()])
    plt.savefig(f'C:/Users/zyh20/Desktop/time_series_correlation/{neuron_a}_{neuron_b}_Rolling_Windowed_Time_lag_crosscorrelation.png')
    

#(3)pearson correlation
def pearson_correlation(d1,d2,neuron_a,neuron_b):
    df = pd.DataFrame({'Column1': d1, 'Column2': d2})
    overall_pearson_r = df.corr().iloc[0,1]
    print(f"Pandas computed Pearson r: {overall_pearson_r}")
    # 输出：使用 Pandas 计算皮尔逊相关结果的 r 值：0.2058774513561943

    r, p = stats.pearsonr(df.dropna()['Column1'], df.dropna()['Column2'])
    print(f"Scipy computed Pearson r: {r} and p-value: {p}")
    # 输出：使用 Scipy 计算皮尔逊相关结果的 r 值：0.20587745135619354，以及 p-value：3.7902989479463397e-51

    # 计算滑动窗口同步性
    f,ax=plt.subplots(figsize=(7,3))
    df.rolling(window=30,center=True).median().plot(ax=ax)
    ax.set(xlabel='Time',ylabel='Pearson r')
    ax.set(title=f"Overall Pearson r = {np.round(overall_pearson_r,2)}")
    plt.savefig(f'C:/Users/zyh20/Desktop/time_series_correlation/{neuron_a}_{neuron_b}_Pearson_correlation.png')

    # 设置窗口宽度，以计算滑动窗口同步性
    r_window_size = 120
    # 插入缺失值
    df_interpolated = df.interpolate()
    # 计算滑动窗口同步性
    rolling_r = df_interpolated['Column1'].rolling(window=r_window_size, center=True).corr(df_interpolated['Column2'])
    f,ax=plt.subplots(2,1,figsize=(14,6),sharex=True)
    df.rolling(window=30,center=True).median().plot(ax=ax[0])
    ax[0].set(xlabel='Frame',ylabel='Smiling Evidence')
    rolling_r.plot(ax=ax[1])
    ax[1].set(xlabel='Frame',ylabel='Pearson r')
    plt.suptitle("Smiling data and rolling window correlation")
    plt.savefig(f'C:/Users/zyh20/Desktop/time_series_correlation/{neuron_a}_{neuron_b}_Pearson_correlation_rolling_window.png')

def synchrony_analysis(region_a,region_b,interval):
    focus_reg_neu_01=neurons[neurons['region']==region_a]
    sorted_df_01 = focus_reg_neu_01.sort_values(by='firing_rate', ascending=False)
    focus_reg_neu_02=neurons[neurons['region']==region_b]
    sorted_df_02 = focus_reg_neu_02.sort_values(by='firing_rate', ascending=False)

    ar=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    combinations_list = list(combinations(ar, 2))
    top5_neurons_combina = np.array(combinations_list)  

    for i in top5_neurons_combina:
        fr_neuron_id_01=sorted_df_01.iloc[i[0], 0]  #extract region1 top5 firing rate neuron id
        fr_neuron_id_02=sorted_df_02.iloc[i[1], 0]  #extract region2 top5 firing rate neuron id

        d1=firingrate_time(fr_neuron_id_01,interval,interval[1]-interval[0])[0]
        d2=firingrate_time(fr_neuron_id_02,interval,interval[1]-interval[0])[0]
        d1=pd.Series(d1)
        d2=pd.Series(d2)

        pearson_correlation(d1,d2,fr_neuron_id_01,fr_neuron_id_02)
        Instantaneous_phase_synchrony(d1,d2,fr_neuron_id_01,fr_neuron_id_02)
        Time_lag_crosscorrelation(d1,d2,fr_neuron_id_01,fr_neuron_id_02)


def boundaries(boundaries,num):
    # 方法是判断 num 大于等于哪个区间的下界，并且小于该区间的上界
    index = np.searchsorted(boundaries, num, side='right') - 1

    # 判断数字是否在边界区间内
    if index < 0 or index >= len(boundaries) - 1:
        print(f"错误，时间 {num} 不在任何定义的区间内")
    else:
        lower_bound = boundaries[index]
        upper_bound = boundaries[index + 1]
        if index % 2 == 0:
            print(f"时间 {num} 位于跑步机运转区间 [{lower_bound}, {upper_bound}) 内")
            return 0
        else:
            print(f"时间 {num} 位于跑步机静止区间 [{lower_bound}, {upper_bound}) 内")
            return 1

def Intercept_time_period(marker_time,sorting_start_time,sorting_end_time):
    start_mark=boundaries(marker_time,sorting_start_time)
    marker = marker_time[(marker_time >= sorting_start_time) & (marker_time <= sorting_end_time)]
    marker = np.insert(marker, 0, sorting_start_time)
    marker = np.append(marker, sorting_end_time)
    print(marker)
    # 两两配对成区间
    intervals = np.column_stack((marker[:-1], marker[1:]))
    # 生成所有区间的索引
    indices = np.arange(len(intervals))
    # 筛选出第偶数个区间 (基于1的索引)
    even_index_intervals = intervals[(indices+1) % 2 == 0]
    odd_index_intervals = intervals[(indices) % 2 == 0]
    if start_mark == 0:
        run = even_index_intervals
        stop = odd_index_intervals
    else:
        run = odd_index_intervals
        stop = even_index_intervals
    print("跑步机运动时间（绝对时间）区间：")
    print(run)
    print("跑步机静止时间（绝对时间）区间：")
    print(stop)
    # 减去开始时间，以和电生理文件的时间对齐
    print("跑步机运动时间（相对时间，电生理sorting后的时间）区间：")
    run_rel=run-sorting_start_time
    print(run_rel)
    print("跑步机静止时间（相对时间，电生理sorting后的时间）区间：")
    stop_rel=stop-sorting_start_time
    print(stop_rel)
    return run_rel,stop_rel


marker_time=marker_for_without_nidqbin()
#Intercept_time_period(marker_time,800,2200)  #后两个参数，sorting截取开始时间和sorting截取结束时间  # 20230113-litermate截取了800-2200sec进行sorting
cc=Intercept_time_period(marker_time,350,950)  #20230112-tremor截取了350-950sec进行sorting
print(cc[1][1])

synchrony_analysis('Interposed nucleus','Spinal vestibular nucleus',cc[1][1])

