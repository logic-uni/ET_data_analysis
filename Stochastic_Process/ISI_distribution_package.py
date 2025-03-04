"""
# coding: utf-8
@author: Yuhao Zhang
last updated: 11/11/2024
data from: Xinchao Chen
"""
import neo
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantities as pq
from mpl_toolkits.axes_grid1 import make_axes_locatable
from elephant.spike_train_generation import homogeneous_poisson_process
from viziphant.statistics import plot_isi_histogram
np.set_printoptions(threshold=np.inf)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### path
mice_name = '20230602_Syt2_conditional_tremor_mice2_lateral'
main_path = r'E:\xinchao\sorted neuropixels data\useful_data\20230602_Syt2_conditional_tremor_mice2_lateral\data'
fig_save_path = r'C:\Users\zyh20\Desktop\ET_data analysis\ISI_distribution\20230602_Syt2_conditional_tremor_mice2_lateral'

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
    spike_times=np.zeros(len(y))
    for i in range(0,len(y)):
        z=y[i]
        spike_times[i]=times[z]/sample_rate
    return spike_times

def trails_division(marker_start,spike_times,duration):
    marker_end = marker_start + duration
    spike_times_trail = spike_times[(spike_times > marker_start) & (spike_times < marker_end)]
    spiketrain = neo.SpikeTrain(spike_times_trail,units='sec',t_start=marker_start, t_stop=marker_end)
    return spiketrain

def ISI_distri_single_trial(population,marker,region_name,trail_mode,duration):  # duration(s)
    neurons_np = population[region_name].to_numpy()
    region_neuron_id = neurons_np[~np.isnan(neurons_np)].astype(int)
    trails_marker = marker[marker['run_or_stop'] == trail_mode]
    ## 截取duration大于等于30s的
    # 初始化一个空数组来存储结果
    result = []
    # 遍历每一行
    for index, row in trails_marker.iterrows():
        # 计算第二列减去第一列的值
        diff = row['time_interval_right_end'] - row['time_interval_left_end']
        # 如果差值大于或等于30，存入第一列的值
        if diff >= 29:
            result.append(row['time_interval_left_end'])

    spike_times = singleneuron_spiketrain(region_neuron_id[0])
    print(region_neuron_id[0])
    spiketrain_trails = trails_division(result[0],spike_times,duration)
    plot_isi_histogram(spiketrain_trails, cutoff=250*pq.ms, histtype='bar')
    plt.show()

def ISI_distri_trails(population,marker,region_name,trail_mode,duration):  # duration(s)
    neurons_np = population[region_name].to_numpy()
    region_neuron_id = neurons_np[~np.isnan(neurons_np)].astype(int)
    trails_marker = marker[marker['run_or_stop'] == trail_mode]
    
    ## 对于运动和静止每段非常短 多个trail切换的 截取duration大于等于30s的
    # 初始化一个空数组来存储结果
    result = []
    # 遍历每一行
    for index, row in trails_marker.iterrows():
        # 计算第二列减去第一列的值
        diff = row['time_interval_right_end'] - row['time_interval_left_end']
        # 如果差值大于或等于30，存入第一列的值
        if diff >= 29:
            result.append(row['time_interval_left_end'])
    '''
    ## 对于littermate trail时间非常长的  duration取400
    result = [105,705,1305]
    '''
    if trail_mode == 1: trail_type = 'run_trials'
    else: trail_type = 'stop_trials'

    print(result)

    for id in region_neuron_id:
        spike_times = singleneuron_spiketrain(id)
        spiketrain_trails = [trails_division(marker_start,spike_times,duration) for marker_start in result]
        plot_isi_histogram(spiketrain_trails, cutoff=250*pq.ms, histtype='bar', legend=result)
        plt.savefig(fig_save_path+f"/{region_name}/{trail_type}/neuron_id_{id}_ISI_ditribution.png",dpi=600,bbox_inches = 'tight')
        plt.close()

def ISI_distri_population(population,marker,region_name,trail_mode,duration):  # duration(s)
    neurons_np = population[region_name].to_numpy()
    region_neuron_id = neurons_np[~np.isnan(neurons_np)].astype(int)
    
    ### Here
    ## Option 1 对于运动和静止多个trail切换的 每段非常短
    trails_marker = marker[marker['run_or_stop'] == trail_mode]  # 区分stop trial or run trial
    '''
    # 找到所有run trial length > 24 的 run trial 的开始时间  duration取24
    result = []
    # 遍历每一行
    for index, row in trails_marker.iterrows():
        # 计算第二列减去第一列的值
        diff = row['time_interval_right_end'] - row['time_interval_left_end']
        # 如果差值大于或等于24  存入第一列的值
        if diff >= 24:
            result.append(row['time_interval_left_end'])
    '''
    # 找到所有stop trial length > 70 的 stop trial 的开始时间  duration取70
    result = []
    # 遍历每一行
    for index, row in trails_marker.iterrows():
        # 计算第二列减去第一列的值
        diff = row['time_interval_right_end'] - row['time_interval_left_end']
        # 如果差值大于或等于70  存入第一列的值
        if diff >= 70:
            result.append(row['time_interval_left_end'])
    '''
    ## Option 2 对于littermate trail时间非常长的  
    result = [105,705]  # run trial  duration取400
    #result = [0,515,1115]  # stop trial  duration取100

    ## Option 3 对于PV syt2 
    result = [364.98]  # stop trial 364.98 - 555 duration取190
    #result = [0] # run trial 0 - 364.98 duration取364
    '''
    
    if trail_mode == 1: trail_type = 'run_trials'
    else: trail_type = 'stop_trials'

    popu_spiketrain_trial = []

    for id in region_neuron_id:
        spike_times = singleneuron_spiketrain(id)
        spiketrain_trails = [trails_division(marker_start,spike_times,duration) for marker_start in result]  #单个神经元多个trial的spike train list
        popu_spiketrain_trial.append(spiketrain_trails)  #每个神经元的 多个trial的spike train list 作为元素 拼接起来

    plot_isi_histogram(popu_spiketrain_trial, cutoff=250*pq.ms)  # 截断ISI分布大于0.25s的
    plt.title(f"{mice_name}_{region_name}_{trail_type}")
    plt.savefig(fig_save_path+f"/{region_name}/{trail_type}_population_ISI_ditribution.png",dpi=600,bbox_inches = 'tight')
    plt.close()
    
ISI_distri_population(neurons,treadmill,region_name='Medial vestibular nucleus',trail_mode=0,duration=70)  # trail_mode: run is 1, stop is 0; duration = trial duration
## 运行前修改region_name, trial mode和duration 这里的duration根据不同的鼠修改，见上面的三个Option