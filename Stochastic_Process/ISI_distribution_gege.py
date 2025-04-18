"""
# coding: utf-8
data from: Gegedong Yang
@author: Yuhao Zhang
last updated: 03/11/2025
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import expon
np.set_printoptions(threshold=np.inf)
np.seterr(divide='ignore',invalid='ignore')

# ------- NEED CHANGE -------
main_path = r'D:\20241227_NP2_r1_co2_session1_g0\20241227_NP2_r1_co2_session1_g0_imec0'


# ------- NO NEED CHANGE -------
### parameter
stat_fit = False             # True/False 是否拟合
fr_filter = 1         # 1  firing rate > 1
cutoff_distr = 250           # 250ms/None  cutoff_distr=0.25代表截断ISI分布大于0.25s的
histo_bin_num = 100          # 统计图bin的个数

### electrophysiology
sorting_path = main_path + '/kilosort4'
save_path = main_path + '/isi'  
sample_rate = 30000 #spikeGLX neuropixel sample rate
identities = np.load(sorting_path + '/spike_clusters.npy') # time series: unit id of each spike
times = np.load(sorting_path + '/spike_times.npy')  # time series: spike time of each spike
neurons = pd.read_csv(sorting_path + '/cluster_group.tsv', sep='\t')  
print(neurons)
elec_dura = times[-1] / sample_rate

print(elec_dura)

# ------- Main Program -------
# get single neuron spike train
def singleneuron_spiketimes(id):
    x = np.where(identities == id)
    y=x[0]
    spike_times=np.empty(len(y))
    for i in range(0,len(y)):
        z=y[i]
        spike_times[i]=times[z]/sample_rate
    return spike_times

def ISI_single_neuron_session(unit_id):
    spike_times = singleneuron_spiketimes(unit_id)
    intervals = np.array([])
    # 如果整个session的spike个数，小于elec_dura电生理时长（s），即每秒钟的spike个数小于1，则不进行统计
    if len(spike_times) > (fr_filter*elec_dura): 
        intervals = np.diff(spike_times)  # 计算时间间隔
        intervals = intervals * 1000   # 转为ms单位
        # 绘制时间间隔的直方图
        if cutoff_distr != None:
            intervals = intervals[(intervals > 0.000999999999) & (intervals <= cutoff_distr)]
            if len(intervals) != 0:  #截取区间可能导致没有interval
                plt.hist(intervals, bins=histo_bin_num, density=False,alpha=0.6)  #如果只看0.25s间隔以内细致的，画counts更适合
        else:
            plt.hist(intervals, bins=20, density=True, alpha=0.6)

    plt.xlabel('Inter-spike Interval (ms)')
    plt.ylabel('Counts')
    plt.title(f'unit id {unit_id}')
    plt.text(0.95, 0.95, f'firing filter > {fr_filter} spike/s\n\nhisto bin: {histo_bin_num}\n\ndistr cutoff > {cutoff_distr}\n\n', ha='right', va='top', transform=plt.gca().transAxes)
    plt.savefig(save_path+f"/neuron_{unit_id}.png",dpi=600,bbox_inches = 'tight')
    plt.clf()
        
def main(units):
    plt.figure(figsize=(10, 6))
    for index, row in units.iterrows():
        unit_id = row['cluster_id']
        ISI_single_neuron_session(unit_id)

main(neurons)