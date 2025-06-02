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