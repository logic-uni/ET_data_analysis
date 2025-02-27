"""
# coding: utf-8
@author: Yuhao Zhang
last updated: 11/15/2024
data from: Xinchao Chen
"""
import math
import torch
import neo
import quantities as pq
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from matplotlib.pyplot import *
from ast import literal_eval
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from elephant.conversion import BinnedSpikeTrain
from elephant import statistics
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde
from scipy.stats import expon
import numba
np.set_printoptions(threshold=np.inf)

### path
mice_name = '20230113_littermate'
main_path = r'E:\xinchao\sorted neuropixels data\useful_data\20230113_littermate\data'
fig_save_path = r'C:\Users\zyh20\Desktop\ET_data analysis\Stochastic Process distribution\20230113_littermate\Lobules IV-V'

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

# get single neuron spike train
def singleneuron_spiketrain(id):
    x = np.where(identities == id)
    y=x[0]
    #y = np.where(np.isin(identities, id))[0]
    spike_times=np.empty(len(y))
    for i in range(0,len(y)):
        z=y[i]
        spike_times[i]=times[z]/sample_rate
    return spike_times

def Stoch_P_trials(matrices,num_columns,num_matrices):
    print(matrices.shape)

    # 初始化存储结果的数组
    results = []

    # 统计每一列的各个值出现的概率
    for col in range(num_columns):
        col_probabilities = []
        for mat_idx in range(num_matrices):
            values, counts = np.unique(matrices[mat_idx, :, col], return_counts=True)
            probabilities = counts / counts.sum()  # 计算概率
            col_probabilities.append((values, probabilities))
        results.append(col_probabilities)

    # 绘制三维图
    for col in range(num_columns):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # 准备数据
        matrices_indices = np.arange(num_matrices)  # 矩阵索引
        x_values = []
        y_values = []
        z_values = []
        
        for mat_idx in range(num_matrices):
            values, probabilities = results[col][mat_idx]
            x_values.extend([mat_idx] * len(values))
            y_values.extend(values)
            z_values.extend(probabilities)
        
        ax.scatter(x_values, y_values, z_values, marker='o')

        ax.set_title(f'Value Probability Distribution at Time {col}')
        ax.set_xlabel('Neurons')
        ax.set_ylabel('Firing rate(spike/s)')
        ax.set_zlabel('Probability')
        plt.show()

def SP_neuron_fr_trial(data,id,region,fr_bin,fig,mode):
    ax = fig.add_subplot(111, projection='3d')
    print(data.shape)
    data_length = data.shape[1]
    trial_num = data.shape[0]
    # 每一列的直方图
    for col in range(data_length):
        # 计算直方图
        hist, bins = np.histogram(data[:, col], bins=10, range=(0, 200),density=False)  # 最多10个bar统计，i.o.w. 每个时刻点的fr最多有10种状态，fr范围[0,500]spike/s
        # 计算条形的中心
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        # 绘制每个切片
        hist_normalized = hist / trial_num  #归一化为 [0, 1] 之间的概率 
        colors = ['blue', 'green', 'red', 'cyan']
        ax.bar(bin_centers, hist_normalized, zs=col*fr_bin, zdir='y', width=20, alpha=0.8,color=colors)

    ax.set_xlabel('Firing rate (spike/s)')
    ax.set_ylabel('Time (ms)')
    ax.set_zlabel('Probability')
    ax.set_title(f'Neuron{id}_{region}_{mode}_S.P.firing rate distribution')
    plt.savefig(fig_save_path+fr"/{mode}_trials/Neuron{id}_S.P.firing rate distribution.jpg",dpi=600)
    plt.clf()

def Stoch_P_one_neuron_distribuplot(data,num_cols):
    num_cols = 10   # 时刻数

    # 初始化存储概率密度的数组
    x = np.linspace(0, 500, 1000)  # 定义x轴范围
    density_values = np.zeros((num_cols, len(x)))

    # 计算每一列的概率密度
    for col in range(num_cols):
        kde = gaussian_kde(data[:, col])  # 计算核密度估计
        density_values[col] = kde(x)  # 评估密度函数

    # 准备绘图
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 创建每个切片
    y = np.arange(num_cols)  # 每一列的索引

    # 绘制切片
    for col in range(num_cols):
        ax.plot(x, y[col] * np.ones_like(x), density_values[col], alpha=0.7)

    # 设置坐标轴标签
    ax.set_xlabel('Value')
    ax.set_ylabel('Columns (Time Moments)')
    ax.set_zlabel('Probability Density')
    ax.set_title('3D Density Slices for Each Column')
    # 设置视角
    ax.view_init(elev=20, azim=30)
    plt.show()

def neuron_fr_session(neuron_id,marker_start,marker_end,fr_bin):
    spike_times = singleneuron_spiketrain(neuron_id)
    spike_times_trail = spike_times[(spike_times > marker_start) & (spike_times < marker_end)]
    spiketrain = neo.SpikeTrain(spike_times_trail,units='sec',t_start=marker_start, t_stop=marker_end)
    fr = BinnedSpikeTrain(spiketrain, bin_size=fr_bin*pq.ms,tolerance=None)
    neuron_fr_ses = fr.to_array().astype(int)[0]
    print(fr.to_array().astype(int)[0].shape)
    print(fr.to_array().astype(int).shape)
    neuron_fr_ses = neuron_fr_ses*1000/fr_bin  # 换算为spike/s的单位
    return neuron_fr_ses

def truncating(marker,trial,fr_bin,fr_session):
    trial_start = marker['time_interval_left_end'].iloc[trial]
    trail_end = marker['time_interval_right_end'].iloc[trial]
    # 一个stop trial里，截取3s为一个小trial(统计区间)
    trial_len = trail_end-trial_start
    remainder = trial_len % 3
    trial_len_trun = trial_len - remainder
    trial_len_trun_bin = int(trial_len_trun*1000/fr_bin)
    start_bin = int(trial_start*1000/fr_bin)
    small_trial_len = int(3*1000/fr_bin)
    for start in range(start_bin,start_bin+trial_len_trun_bin,small_trial_len):
        end = start + small_trial_len
        neruon_trial = fr_session[start:end]
    return neruon_trial

def runtrials(marker,fr_session,fr_bin):
    a=0
    for trial_num in np.arange(1,len(marker['time_interval_left_end']),2):
        neruon_trial = truncating(marker,trial_num,fr_bin,fr_session)
        if a == 0:
            neruon_trials = neruon_trial
            a=a+1
        else:
            neruon_trials = np.vstack((neruon_trials, neruon_trial))
    return neruon_trials

def stoptrials(marker,fr_session,fr_bin):
    a=0
    for trial_num in np.arange(0,len(marker['time_interval_left_end']),2):
        neruon_trial = truncating(marker,trial_num,fr_bin,fr_session)
        if a == 0:
            neruon_trials = neruon_trial
            a=a+1
        else:
            neruon_trials = np.vstack((neruon_trials, neruon_trial))
    
    return neruon_trials

def main(neurons,marker,region_name,mice_type,mode):
    fr_bin = 30 # unit ms  #对于存在delay的20ms的间隔可能都bin不到一个spike
    ## 取脑区全部的neuron id 
    for i in range(neurons.shape[1]):  
        name = neurons.columns.values[i]
        if name == region_name:
            region_neurons_id = np.array(neurons.iloc[:, i].dropna()).astype(int)
    fig = plt.figure(figsize=(12, 8))
    
    session_start = marker['time_interval_left_end'].iloc[0]
    session_end = marker['time_interval_right_end'].iloc[-1]
    # 运动和静止的阶段，分别人为切分成3s算一次轨道的实现（对应行为3s完成locomtion的一次循环是合理的）
    # 对于一个session，约100多个轨道进行统计，否则数目过少，不具备统计显著性
    # 对于一次轨道，应该切分的足够小的时间段，才可以得到具体的分布特征，
    # 假设符合泊松过程，这个足够小的时间段，就是对应的泊松分布里的t，这里取t=30ms，以近似足够小的时间段
    # 即一个3s（3000ms）的轨道，切分为30ms一个，对应100个切片
    for neuron_id in region_neurons_id:
        if mice_type == 'cond_ET':
            fr_session = neuron_fr_session(neuron_id,session_start,session_end,fr_bin)
            if mode == 'run':
                neruon_trials = runtrials(marker,fr_session,fr_bin)
                SP_neuron_fr_trial(neruon_trials,neuron_id,region_name,fr_bin,fig,mode='run')
            elif mode == 'stop':
                neruon_trials = stoptrials(marker,fr_session,fr_bin)
                SP_neuron_fr_trial(neruon_trials,neuron_id,region_name,fr_bin,fig,mode='stop')
        elif mice_type == 'littermate':
            fr_session = neuron_fr_session(neuron_id,session_start,session_end,fr_bin)
            if mode == 'run':
                neruon_trials = runtrials(marker,fr_session,fr_bin)
                SP_neuron_fr_trial(neruon_trials,neuron_id,region_name,fr_bin,fig,mode='run')
            elif mode == 'stop':
                neruon_trials = stoptrials(marker,fr_session,fr_bin)
                SP_neuron_fr_trial(neruon_trials,neuron_id,region_name,fr_bin,fig,mode='stop')
        '''
        elif mice_type == 'PV_Syt2':
            PV_Syt2_runtrials(marker,neuron_id,fr_bin)
            PV_Syt2_stoptrials(marker,neuron_id,fr_bin)
        ''' 
    
region_name = 'Lobules IV-V'
mode='run'
#mice_type = 'cond_ET'
#mice_type = 'PV_Syt2'
mice_type = 'littermate'
main(neurons,treadmill,region_name,mice_type,mode)