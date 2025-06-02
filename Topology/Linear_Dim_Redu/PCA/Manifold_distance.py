"""
# coding: utf-8
@author: Yuhao Zhang
last updated: 08/27/2024
data from: Xinchao Chen
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
np.set_printoptions(threshold=np.inf)

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
    ## smooth data
    data2pca = pd.DataFrame(data2pca)
    rate = np.sqrt(data2pca/0.1)
    #对数据做均值  默认: window=50  min_periods=1  感觉改这些值影响不大，改firing的bin size影响较大
    rate = rate.rolling(window=50,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=2) 
    ## reduce dimension
    #由于是计算距离，因此必须使用没有伸缩变换的PCA来计算
    pca = PCA(n_components=3)
    redu_dim_data = pca.fit_transform(rate.values) 
    explained_variance_ratio = pca.explained_variance_ratio_   #每个主成分所解释的方差比例
    explained_variance_sum = np.cumsum(explained_variance_ratio)  #计算累积解释方差比例

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