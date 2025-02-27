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
import numba
np.set_printoptions(threshold=np.inf)

np.set_printoptions(threshold=np.inf)
np.seterr(divide='ignore',invalid='ignore')

### path
mice_name = '20230602_Syt2_conditional_tremor_mice2_lateral'
main_path = r'E:\xinchao\sorted neuropixels data\useful_data\20230602_Syt2_conditional_tremor_mice2_lateral\data'
save_path = r'C:\Users\zyh20\Desktop\ET_data analysis\spectrum analysis\20230602_Syt2_conditional_tremor_mice2_lateral\Medial vestibular nucleus'

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

def popu_fr_session(neuron_ids,marker_start,marker_end,fr_bin):   #开始推杆，到推杆结束的一个trial的population spike counts
    for j in range(len(neuron_ids)): #第j个neuron
        spike_times = singleneuron_spiketrain(neuron_ids[j])
        spike_times_trail = spike_times[(spike_times > marker_start) & (spike_times < marker_end)]
        spiketrain = neo.SpikeTrain(spike_times_trail,units='sec',t_start=marker_start, t_stop=marker_end)
        fr = BinnedSpikeTrain(spiketrain, bin_size=fr_bin*pq.ms,tolerance=None)
        one_neruon = fr.to_array().astype(int)[0]
        if j == 0:
            neurons = one_neruon
        else:
            neurons = np.vstack((neurons, one_neruon))
    return neurons

def reduce_dimension(count,bin_size,n_components): # 默认: 0.1 感觉改bin_size影响不大，改firing rate的bin size影响较大
    #smooth data
    count = pd.DataFrame(count)
    rate = np.sqrt(count/bin_size)
    #对数据做均值  默认: window=50  min_periods=1  感觉改这些值影响不大，改firing的bin size影响较大
    rate = rate.rolling(window=50,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=2) 
    #reduce dimension
    
    pca = PCA(n_components)
    X_pca = pca.fit_transform(rate.values)   #对应的是Explained variance
    explained_variance_ratio = pca.explained_variance_ratio_   #每个主成分所解释的方差比例
    explained_variance_sum = np.cumsum(explained_variance_ratio)  #计算累积解释方差比例
    
    #X_isomap = Isomap(n_components = 3, n_neighbors = 21).fit_transform(rate.values)  #对应的是Residual variance
    #X_tsne = TSNE(n_components=3,random_state=21,perplexity=20).fit_transform(rate.values)  #t-SNE没有Explained variance，t-SNE 旨在保留局部结构而不是全局方差
    return X_pca

def SVD_matrix_spectrum(data):
    # 1. 进行奇异值分解
    U, S, Vt = np.linalg.svd(data)

    # 2. 绘制奇异值的分布
    plt.figure(figsize=(8, 5))
    plt.plot(S, marker='o', linestyle='-', color='b')
    plt.title('Singular Values from SVD')
    plt.xlabel('Index')
    plt.ylabel('Singular Value')
    plt.grid(True)
    plt.show()

    # 3. 计算最大奇异值对应的特征向量
    max_singular_value_index = np.argmax(S)
    max_singular_value_vector = U[:, max_singular_value_index]

    # 输出最大奇异值及其对应的特征向量
    print(f'Max Singular Value: {S[max_singular_value_index]}')
    print(f'Corresponding Feature Vector (Left Singular Vector): {max_singular_value_vector}')

def redu_eign(data):
    print(data.shape)
    data2pca=data.T
    redu_dim_data=reduce_dimension(data2pca,0.1,30)
    ## 这里的矩阵是人为确定过保证Neuron和time两个轴的维度相同，对这样截断的矩阵做谱分析，因此存在虚数
    print(redu_dim_data.shape)
    # 对非对称矩阵进行特征值分解
    eigenvalues, eigenvectors = np.linalg.eig(redu_dim_data)
    return eigenvalues

def sliding_spectrum(data,mode):  #mode='b' or 'r'
    print(data.shape)
    eigenvalues, eigenvectors = np.linalg.eig(data)
    # 绘制特征值在复平面上的位置
    plt.scatter(eigenvalues.real, eigenvalues.imag, c=mode, marker='o')
    #plt.pause(0.02)

def covmatr_spectrum(data):
    ## covarience matrix是对称阵，所以只有实数值的特征值
    ATA = np.dot(data.T, data)
    # 对 A^T A 进行特征值分解
    eigenvalues, eigenvectors = np.linalg.eig(ATA)

    # 绘制特征值的直方图
    plt.figure(figsize=(8, 6))
    plt.hist(eigenvalues, bins=30, color='blue', edgecolor='black', alpha=0.7)

    # 设置标题和标签
    plt.title('Histogram of Eigenvalues of $A^T A$')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Frequency')

    # 显示图像
    plt.grid(True)
    plt.show()

    # 存储特征值最大的特征向量
    max_index = np.argmax(eigenvalues)  # 找到最大特征值的索引
    max_eigenvalue = eigenvalues[max_index]  # 最大特征值
    max_eigenvector = eigenvectors[:, max_index]  # 对应的特征向量
    return max_eigenvector

def eigen_vector_included_angle(A,B):
    # 将矩阵展平为一维数组
    A_flat = A.flatten()
    B_flat = B.flatten()
    # 计算点积和模长
    dot_product = np.dot(A_flat, B_flat)
    norm_A = np.linalg.norm(A_flat)
    norm_B = np.linalg.norm(B_flat)
    # 计算余弦相似度
    similarity = dot_product / (norm_A * norm_B)
    print(similarity)

def ET_runtrials(marker,pfs,region_neurons_id,fr_bin):
    neuron_num = len(region_neurons_id)
    remainder = 6000 % neuron_num  # 6000 = 30000ms / 5ms  一个trial 6000个bin
    segmen = 6000 - remainder
    for trial in np.arange(1,len(marker['time_interval_left_end']),2):
        marker_start = marker['time_interval_left_end'].iloc[trial]
        bin_level_start = int(marker_start*1000/fr_bin)
        if marker['time_interval_right_end'].iloc[trial] - marker_start > 30:
            for start in range(0,segmen,neuron_num):
                start_bin = start + bin_level_start
                end_bin = start_bin + neuron_num
                data = pfs[:,start_bin:end_bin]
                sliding_spectrum(data,'r')

def ET_stoptrials(marker,pfs,region_neurons_id,fr_bin):
    neuron_num = len(region_neurons_id)
    remainder = 6000 % neuron_num  # 6000 = 30000ms / 5ms
    segmen = 6000 - remainder
    for trial in np.arange(0,len(marker['time_interval_left_end'])-2,2):
        marker_start = marker['time_interval_left_end'].iloc[trial]
        bin_level_start = int(marker_start*1000/fr_bin)
        if marker['time_interval_right_end'].iloc[trial] - marker_start > 30:
            for start in range(0,segmen,neuron_num):
                start_bin = start + bin_level_start
                end_bin = start_bin + neuron_num
                data = pfs[:,start_bin:end_bin]
                sliding_spectrum(data,'b')

def littermate_runtrials(pfs,region_neurons_id,fr_bin):  ## LITTERMATE 人为切分30s的trial
    neuron_num = len(region_neurons_id)
    remainder = 6000 % neuron_num  # 6000 = 30000ms / 5ms
    segmen = 6000 - remainder
    for marker_start in np.arange(105,495,30):
        bin_level_start = int(marker_start*1000/fr_bin)
        for start in range(0,segmen,neuron_num):
            start_bin = start + bin_level_start
            end_bin = start_bin + neuron_num
            data = pfs[:,start_bin:end_bin]
            sliding_spectrum(data,'r')
    for marker_start in np.arange(705,1095,30):
        bin_level_start = int(marker_start*1000/fr_bin)
        for start in range(0,segmen,neuron_num):
            start_bin = start + bin_level_start
            end_bin = start_bin + neuron_num
            data = pfs[:,start_bin:end_bin]
            sliding_spectrum(data,'r')

def littermate_stoptrials(pfs,region_neurons_id,fr_bin):  ## LITTERMATE 人为切分30s的trial
    neuron_num = len(region_neurons_id)
    remainder = 6000 % neuron_num  # 6000 = 30000ms / 5ms
    segmen = 6000 - remainder
    for marker_start in np.arange(0,90,30):
        bin_level_start = int(marker_start*1000/fr_bin)
        for start in range(0,segmen,neuron_num):
            start_bin = start + bin_level_start
            end_bin = start_bin + neuron_num
            data = pfs[:,start_bin:end_bin]
            sliding_spectrum(data,'b')
    for marker_start in np.arange(515,695,30):
        bin_level_start = int(marker_start*1000/fr_bin)
        for start in range(0,segmen,neuron_num):
            start_bin = start + bin_level_start
            end_bin = start_bin + neuron_num
            data = pfs[:,start_bin:end_bin]
            sliding_spectrum(data,'b')
    for marker_start in np.arange(1115,1295,30):
        bin_level_start = int(marker_start*1000/fr_bin)
        for start in range(0,segmen,neuron_num):
            start_bin = start + bin_level_start
            end_bin = start_bin + neuron_num
            data = pfs[:,start_bin:end_bin]
            sliding_spectrum(data,'b')

def main(neurons,marker,region_name,mice_type):
    fr_bin = 5  # unit: ms
    plt.figure()
    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    plt.title('Eigenvalues of Non-Symmetric Matrix on the Complex Plane')
    plt.xlabel('Real Part')
    plt.xlim(-5,5)
    plt.ylim(-5,5)
    plt.ylabel('Imaginary Part')
    plt.grid(True)
    theta = np.linspace(0, 2 * np.pi, 100)
    x = 1 * np.cos(theta)
    y = 1 * np.sin(theta)
    plt.plot(x, y)
    ## 取脑区全部的neuron id 
    for i in range(neurons.shape[1]):  
        name = neurons.columns.values[i]
        if name == region_name:
            region_neurons_id = np.array(neurons.iloc[:, i].dropna()).astype(int)

    session_start = marker['time_interval_left_end'].iloc[0]
    session_end = marker['time_interval_right_end'].iloc[-1]
    pfs = popu_fr_session(region_neurons_id,session_start,session_end,fr_bin)

    if mice_type == 'cond_ET':
        ET_runtrials(marker,pfs,region_neurons_id,fr_bin)
        ET_stoptrials(marker,pfs,region_neurons_id,fr_bin)
    elif mice_type == 'littermate':
        littermate_runtrials(pfs,region_neurons_id,fr_bin)
        littermate_stoptrials(pfs,region_neurons_id,fr_bin)
    '''
    elif mice_type == 'PV_Syt2':
        PV_Syt2_runtrials(marker,region_neurons_id,fr_bin)
        PV_Syt2_stoptrials(marker,region_neurons_id,fr_bin)
    '''
    #max_eigen1 = covmatr_spectrum(data[:,(pert_s-40):(pert_s)])
    #max_eigen2 = covmatr_spectrum(data[:,(pert_s):(pert_s+40)])
    #eigen_vector_included_angle(max_eigen1,max_eigen2)
    #covmatr_spectrum(data[:,(pert_s-40):(pert_s)])
    #covmatr_spectrum(data[:,(pert_s-40):(pert_e+150)])
    #SVD_matrix_spectrum(data[:,(pert_s):(pert_s+40)])
    plt.show()

region_name = 'Medial vestibular nucleus'
mice_type = 'cond_ET'
#mice_type = 'PV_Syt2'
#mice_type = 'littermate'
main(neurons,treadmill,region_name,mice_type)
