"""
# coding: utf-8
@author: Yuhao Zhang
last updated: 04/19/2025
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
import warnings
np.set_printoptions(threshold=np.inf)
np.seterr(divide='ignore',invalid='ignore')

fr_bin = 10  # unit: ms
fs = 1000 / fr_bin  # fr sample rate = 1000ms / bin size 如果bin是10ms，则采样率为100hz，根据香农采样定理，FFT时候会自动把最大频率确定在100hz以内
freq_low, freq_high = 1, 25
neuron_id = 1

### ------------------ Load Data-------------------

## --------- NP2 ----------
mice_name = '20250310_VN_control'
sorting_path = f"/data1/zhangyuhao/xinchao_data/NP2/{mice_name}/Sorted/"
neurons = pd.read_csv(f"/data1/zhangyuhao/xinchao_data/NP2/{mice_name}/filtered_quality_metrics.csv")  # QC neurons
#neurons = pd.read_csv(sorting_path + '/cluster_group.tsv', sep='\t')   # all neurons
marker = pd.read_csv(f"/data1/zhangyuhao/xinchao_data/NP2/{mice_name}/Marker/static_motion_segement.csv")
save_path = f"/home/zhangyuhao/Desktop/Result/ET/Spikecounts_FFT/NP2/{mice_name}/"

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

## -------- FFT ---------
def neuron_spikecounts(neuron_id,marker_start,marker_end):
    spike_times = singleneuron_spiketimes(neuron_id)
    spike_times_trail = spike_times[(spike_times > marker_start) & (spike_times < marker_end)]
    spiketrain = neo.SpikeTrain(spike_times_trail,units='sec',t_start=marker_start, t_stop=marker_end)
    fr = BinnedSpikeTrain(spiketrain, bin_size=fr_bin*pq.ms,tolerance=None)  # had been qualified that elephant can generate correct spike counts
    trial_neuron_fr = fr.to_array().astype(int)[0]
    return trial_neuron_fr

def FFT(data,content):
    window = np.hanning(len(data))
    windowed_data = data * window
    fft_result = np.fft.fft(windowed_data)
    freqs = np.fft.fftfreq(len(windowed_data), 1/fs)
    freq_mask = (freqs >= freq_low) & (freqs <= freq_high)
    fft_filtered = fft_result[freq_mask]
    freqs_filtered = freqs[freq_mask]
    plt.plot(freqs_filtered, np.abs(fft_filtered))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title(f'FFT {content}')
    plt.savefig(os.path.join(save_path, f'{content}.png'))
    plt.clf()

def popu_spikecounts(start,end):
    # NP2
    for index, row in neurons.iterrows():
        unit_id = row['cluster_id']
        spikecount = neuron_spikecounts(unit_id,start,end)
        if index == 0:
            all_spikecounts = np.array([spikecount])
        else:
            all_spikecounts = np.vstack((all_spikecounts, spikecount))
    '''
    # NP1
    result = neurons.groupby('region')['cluster_id'].apply(list).reset_index(name='cluster_ids')
    for index, row in result.iterrows():
        region = row['region']
        popu_ids = row['cluster_ids']
        if region == region_name:
            for j in range(len(popu_ids)): #第j个neuron
                spikecount = neuron_spikecounts(popu_ids[j],start,end)
                if j == 0:
                    all_spikecounts = np.array([spikecount])
                else:
                    all_spikecounts = np.vstack((all_spikecounts, spikecount))
    '''
    return all_spikecounts

def tiny_range_popusc(spec_start,spec_end,trial_type):
    start = spec_start
    end = spec_end
    while start < spec_end:
        end = start + 0.5
        if end > spec_end:
            end = spec_end
        popu_sc = popu_spikecounts(start, end)
        sum_sc = np.sum(popu_sc, axis=0)
        time = np.linspace(start, end, len(sum_sc))
        plt.plot(time, sum_sc)
        plt.title(f'Spike counts of summed neurons ({start}-{end}s)')
        plt.xlabel('Time (s)')
        plt.ylabel('Spike counts')
        plt.savefig(os.path.join(save_path, f'sum_all_units_{start}_{end}_{trial_type}.png'))
        plt.clf()
        start += 0.5

def spec_range_popusc(start,end,trial_type):
    popu_sc = popu_spikecounts(start, end)
    sum_sc = np.sum(popu_sc, axis=0)
    time = np.linspace(start, end, len(sum_sc))
    plt.plot(time, sum_sc)
    plt.title(f'Spike counts of summed neurons ({start}-{end}s)')
    plt.xlabel('Time (s)')
    plt.ylabel('Spike counts')
    plt.savefig(os.path.join(save_path, f'sum_all_units_{start}_{end}_{trial_type}.png'))
    plt.clf()
    FFT(sum_sc,f'sum_all_units_{start}_{end}_{trial_type}_FFT')

def enumarate_trials():
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
        #neuron_spikecounts(neuron_id,start,end)
        popu_sc = popu_spikecounts(start,end)
        sum_sc = np.sum(popu_sc, axis=0)
        time = np.linspace(start, end, len(sum_sc))
        plt.plot(time,sum_sc)
        plt.title('Spike counts of sumed neurons')
        plt.xlabel('time')
        plt.ylabel('Spike counts')
        plt.savefig(os.path.join(save_path, f'sum_all_units_{start}_{end}_{trial_type}.png'))
        plt.clf()
        FFT(sum_sc,f'sum_all_units_{start}_{end}_{trial_type}_FFT')

enumarate_trials()
#tiny_range_popusc(40,350,'locomotion')
#spec_range_popusc(40,350,'run')