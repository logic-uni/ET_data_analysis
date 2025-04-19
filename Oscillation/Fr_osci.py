"""
# coding: utf-8
@author: Yuhao Zhang
last updated: 04/18/2025
data from: Xinchao Chen
"""
import neo
import quantities as pq
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.pyplot import *
from ast import literal_eval
from elephant.conversion import BinnedSpikeTrain
import os
np.set_printoptions(threshold=np.inf)
np.seterr(divide='ignore',invalid='ignore')

fr_bin = 10  # unit: ms
fs = 1000 / fr_bin  # # fr sample rate = 1000ms / bin size
freq_low, freq_high = 25, 120
neuron_id = 14

### ------------------ Load Data-------------------

## --------- NP2 ----------
mice_name = '20250310_VN_tremor'
sorting_path = f"/data1/zhangyuhao/xinchao_data/NP2/{mice_name}/Sorted/"
save_path = f"/home/zhangyuhao/Desktop/Result/ET/Fr_FFT/NP2/{mice_name}/"
neurons = pd.read_csv(sorting_path + '/cluster_group.tsv', sep='\t')  
marker = pd.read_csv(f"/data1/zhangyuhao/xinchao_data/NP2/{mice_name}/Marker/static_motion_segement.csv")

'''
## --------- NP1 ----------
mice_name = '20230602_Syt2_conditional_tremor_mice2_lateral'
region_name = 'Superior vestibular nucleus'
# ------ Easysort ------  With QC
#  NEED CHANGE
mapping_file = 'unit_ch_dep_region_QC_isi_violations_ratio_pass_rate_60.17316017316017%.csv'
#  NO NEED CHANGE
sorting_path = f"/data1/zhangyuhao/xinchao_data/NP1/{mice_name}/Sorted/Easysort/results_KS2/sorter_output/"
save_path = f"/home/zhangyuhao/Desktop/Result/ET/Fr_FFT/NP1/Easysort/{mice_name}/"
neurons = pd.read_csv(f"/data1/zhangyuhao/xinchao_data/NP1/{mice_name}/Sorted/Easysort/mapping/{mapping_file}")  # different sorting have different nueron id
# ------ Xinchao_sort ------  Without QC
#  NO NEED CHANGE 
#sorting_path = f"/data1/zhangyuhao/xinchao_data/NP1/{mice_name}/Sorted/Xinchao_sort/"
#save_path = "/home/zhangyuhao/Desktop/Result/ET/Fr_FFT/NP1/Xinchao_sort/{mice_name}/"
#neurons = pd.read_csv(sorting_path + '/neuron_id_region_firingrate.csv')  # different sorting have different nueron id
#marker = pd.read_csv(f"/data1/zhangyuhao/xinchao_data/NP1/{mice_name}/Marker/treadmill_move_stop_velocity_segm_trial.csv",index_col=0)
marker = pd.read_csv(f"/data1/zhangyuhao/xinchao_data/NP1/{mice_name}/Marker/treadmill_move_stop_velocity.csv",index_col=0)
'''

# ---------- Load electrophysiology data ----------
sample_rate = 30000 #spikeGLX neuropixel sample rate
identities = np.load(sorting_path + '/spike_clusters.npy') # time series: unit id of each spike
times = np.load(sorting_path + '/spike_times.npy')  # time series: spike time of each spike
print("Test if electrophysiology duration is equal to treadmill duration ...")
print(f"Marker duration: {marker['time_interval_right_end'].iloc[-1]}")

## Change NP1 or NP2
print(f"Electrophysiology duration: {times[-1] / sample_rate}")     # NP2
#print(f"Electrophysiology duration: {(times[-1] / sample_rate)[0]}")  # NP1

### ------------------ Main Program -------------------
def singleneuron_spiketimes(id):
    x = np.where(identities == id)
    y=x[0]
    spike_times=np.zeros(len(y))
    for i in range(0,len(y)):
        z=y[i]
        spike_times[i]=times[z]/sample_rate
    return spike_times

def neuron_fr_trial(neuron_id,marker_start,marker_end,type):
    spike_times = singleneuron_spiketimes(neuron_id)
    spike_times_trail = spike_times[(spike_times > marker_start) & (spike_times < marker_end)]
    spiketrain = neo.SpikeTrain(spike_times_trail,units='sec',t_start=marker_start, t_stop=marker_end)
    fr = BinnedSpikeTrain(spiketrain, bin_size=fr_bin*pq.ms,tolerance=None)
    trial_neuron_fr = fr.to_array().astype(int)[0]
    print("hhhhhhh")
    print(trial_neuron_fr)
    window = np.hanning(len(trial_neuron_fr))  # 正确函数名为 hanning()
    windowed_data = trial_neuron_fr * window
    # Perform FFT on trial_neruon_fr
    fft_result = np.fft.fft(windowed_data)
    freqs = np.fft.fftfreq(len(windowed_data), 1/fs)  
    # Focus on the frequency range of interest
    freq_mask = (freqs >= freq_low) & (freqs <= freq_high)
    fft_filtered = fft_result[freq_mask]
    freqs_filtered = freqs[freq_mask]
    # Plot the FFT result
    plt.plot(freqs_filtered, np.abs(fft_filtered))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title(f'FFT of Neuron {neuron_id} (Trial {marker_start}-{marker_end})')
    plt.savefig(os.path.join(save_path, f'Neuron_id_{neuron_id}_trial_{marker_start}_{marker_end}_{type}.png'))
    plt.clf()

def enumerate_neurons(start,end,trial_type):
    result = neurons.groupby('region')['cluster_id'].apply(list).reset_index(name='cluster_ids')
    for index, row in result.iterrows():
        region = row['region']
        popu_ids = row['cluster_ids']
        if region == region_name:
            for j in range(len(popu_ids)): #第j个neuron
                neuron_fr_trial(popu_ids[j],start,end,trial_type)

def main():
    plt.figure(figsize=(10, 6))
    for index, row in marker.iterrows():
        start = row['time_interval_left_end']
        end = row['time_interval_right_end']
        if end - start < 2:
            continue
        status = row['run_or_stop']
        if status == 0:
            trial_type = 'static'
        else:
            trial_type = 'run'
        neuron_fr_trial(neuron_id,start,end,trial_type)
        #enumerate_neurons(start,end,trial_type)

main()
