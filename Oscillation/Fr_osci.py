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
np.set_printoptions(threshold=np.inf)
np.seterr(divide='ignore',invalid='ignore')

fr_bin = 10  # unit: ms
fs = 
freq_low, freq_high = 4, 16

### ------------------ Load Data-------------------

## --------- NP2 ----------
mice_name = '20250310_VN_tremor'
sorting_path = f"/data1/zhangyuhao/xinchao_data/NP2/{mice_name}/Sorted/"
save_path = f"/home/zhangyuhao/Desktop/Result/ET/Fr_FFT/NP1/Easysort/{mice_name}/"
sample_rate = 30000 #spikeGLX neuropixel sample rate
identities = np.load(sorting_path + '/spike_clusters.npy') # time series: unit id of each spike
times = np.load(sorting_path + '/spike_times.npy')  # time series: spike time of each spike
neurons = pd.read_csv(sorting_path + '/cluster_group.tsv', sep='\t')  
print(neurons)
elec_dura = times[-1] / sample_rate
marker = pd.read_csv(f"/data1/zhangyuhao/xinchao_data/NP2/{mice_name}/Marker/static_motion_segement.csv")
print(marker)
print("Test if electrophysiology duration is equal to treadmill duration ...")
print(f"Electrophysiology duration: {elec_dura}")
print(f"marker duration: {marker['time_interval_right_end'].iloc[-1]}")

## --------- NP1 ----------
mice_name = '20230602_Syt2_conditional_tremor_mice2_lateral'
region_name = 'Lobule III'
# Regeion List
# Lobule
#20230113_littermate    Lobules IV-V
#20230523_Syt2_conditional_tremor_mice1    Lobule III  Lobule II
#20230604_Syt2_conditional_tremor_mice2_medial   Lobule III
#20230602_Syt2_conditional_tremor_mice2_lateral  Lobule III
#20230623_Syt2_conditional_tremor_mice4  Lobule III  Lobule II
# DCN
#20230604_Syt2_conditional_tremor_mice2_medial  xinchaosort  Vestibulocerebellar nucleus
#20230113_littermate   Interposed nucleus
# ------ Easysort ------  With QC
#  NEED CHANGE
mapping_file = 'unit_ch_dep_region_QC_isi_violations_ratio_pass_rate_60.17316017316017%.csv'
#  NO NEED CHANGE
sorting_path = f"/data1/zhangyuhao/xinchao_data/NP1/{mice_name}/Sorted/Easysort/results_KS2/sorter_output/"
save_path = f"/home/zhangyuhao/Desktop/Result/ET/Fr_FFT/NP1/Easysort/{mice_name}/"
neurons = pd.read_csv(f"/data1/zhangyuhao/xinchao_data/NP1/{mice_name}/Sorted/Easysort/mapping/")  # different sorting have different nueron id
# ------ Xinchao_sort ------  Without QC
#  NO NEED CHANGE 
#sorting_path = f"/data1/zhangyuhao/xinchao_data/NP1/{mice_name}/Sorted/Xinchao_sort/"
#save_path = "/home/zhangyuhao/Desktop/Result/ET/Fr_FFT/NP1/Xinchao_sort/{mice_name}/"
#neurons = pd.read_csv(sorting_path + '/neuron_id_region_firingrate.csv')  # different sorting have different nueron id
#  ------ Common ------ electrophysiology & behavior
treadmill = pd.read_csv(f"/data1/zhangyuhao/xinchao_data/NP1/{mice_name}/Marker/treadmill_move_stop_velocity_segm_trial.csv",index_col=0)
treadmill_origin = pd.read_csv(f"/data1/zhangyuhao/xinchao_data/NP1/{mice_name}/Marker/treadmill_move_stop_velocity.csv",index_col=0)
sample_rate = 30000 #spikeGLX neuropixel sample rate
identities = np.load(sorting_path + '/spike_clusters.npy') # time series: unit id of each spike
times = np.load(sorting_path + '/spike_times.npy')  # time series: spike time of each spike
print(neurons)
print("Test if electrophysiology duration is equal to treadmill duration ...")
elec_dura = (times[-1]/sample_rate)[0]
treadmill_dura = treadmill_origin['time_interval_right_end'].iloc[-1]
print(f"Electrophysiology duration: {elec_dura}")
print(f"Treadmill duration: {treadmill_dura}")

### ------------------ Main Program -------------------

def singleneuron_spiketimes(id):
    x = np.where(identities == id)
    y=x[0]
    spike_times=np.zeros(len(y))
    for i in range(0,len(y)):
        z=y[i]
        spike_times[i]=times[z]/sample_rate
    return spike_times

def fr_trial(neuron_ids,marker_start,marker_end):
    for j in range(len(neuron_ids)): #第j个neuron
        spike_times = singleneuron_spiketimes(neuron_ids[j])
        spike_times_trail = spike_times[(spike_times > marker_start) & (spike_times < marker_end)]
        spiketrain = neo.SpikeTrain(spike_times_trail,units='sec',t_start=marker_start, t_stop=marker_end)
        fr = BinnedSpikeTrain(spiketrain, bin_size=fr_bin*pq.ms,tolerance=None)
        trial_neruon_fr = fr.to_array().astype(int)[0]
        
        # Perform FFT on trial_neruon_fr
        fft_result = np.fft.fft(trial_neruon_fr)
        freqs = np.fft.fftfreq(len(trial_neruon_fr), d=fr_bin / 1000.0)  # Convert bin size to seconds

        # Focus on the frequency range of interest
        freq_mask = (freqs >= freq_low) & (freqs <= freq_high)
        fft_filtered = fft_result[freq_mask]
        freqs_filtered = freqs[freq_mask]

        # Plot the FFT result
        plt.figure(figsize=(10, 6))
        plt.plot(freqs_filtered, np.abs(fft_filtered), label=f'Neuron {neuron_ids[j]}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.title(f'FFT of Neuron {neuron_ids[j]} (Trial {marker_start}-{marker_end})')
        plt.legend()
        plt.grid()
        plt.show()

def main():
    result = neurons.groupby('region')['cluster_id'].apply(list).reset_index(name='cluster_ids')
    print(result)
    for index, row in result.iterrows():
        region = row['region']
        popu_ids = row['cluster_ids']
        if region == region_name:
            for index, row in treadmill.iterrows():
                trial_start = row['time_interval_left_end']
                trial_end = row['time_interval_right_end']
                status = row['run_or_stop']
                if status == 0:
                    color = 'b'
                else:
                    color = 'r'
                data = fr_trial(popu_ids,trial_start,trial_end)

main()
