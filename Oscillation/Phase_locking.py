"""
# coding: utf-8
@author: Yuhao Zhang
last updated: 04/16/2025
data from: Xinchao Chen
"""

import pandas as pd
import numpy as np
import cupy as cp
from scipy.signal import butter, filtfilt, hilbert
import matplotlib.pyplot as plt
import os
from elephant.phase_analysis import phase_locking_value

mice_name = '20250310_VN_tremor'  # 20250310_VN_control 20250310_VN_harmaline  20250310_VN_tremor

# ------- NO NEED CHANGE -------
fs = 30000  # 30 kHz for NP2
lfp_data = np.load(f"/data1/zhangyuhao/xinchao_data/NP2/{mice_name}/LFP_npy/{mice_name}.npy")

sorting_path = f"/data1/zhangyuhao/xinchao_data/NP2/{mice_name}/Sorted/"
identities = np.load(sorting_path + '/spike_clusters.npy') # time series: unit id of each spike
times = np.load(sorting_path + '/spike_times.npy')  # time series: spike time of each spike
neurons = pd.read_csv(sorting_path + '/cluster_group.tsv', sep='\t')  
print(neurons)
print(f"LFP duration: {lfp_data.shape[1]/fs}")
print(f"AP duration: {times[-1] / fs}")
save_path = f"/home/zhangyuhao/Desktop/Result/ET/Spike_Phase/NP2/{mice_name}/"  

def singleneuron_spiketimes(id):
    x = np.where(identities == id)
    y=x[0]
    spike_times=np.empty(len(y))
    for i in range(0,len(y)):
        z=y[i]
        spike_times[i]=times[z]/fs
    return spike_times

def popu_spiketimes():
    for index, row in neurons.iterrows():
        unit_id = row['cluster_id']
        spike_times = singleneuron_spiketimes(unit_id)
        fr = len(spike_times)/(times[-1] / fs)
        if fr > 2:
            if 'spike_matrix' not in locals():
                spike_matrix = [spike_times]
            else:
                spike_matrix.append(spike_times)
            
popu_spike_times = popu_spiketimes()
plv = phase_locking_value(popu_spike_times, lfp_data)  # phases_i, phases_j(t, n) np.ndarray