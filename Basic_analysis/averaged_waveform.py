"""
# coding: utf-8
@author: Yuhao Zhang
last updated: 02/27/2025
data from: Xinchao Chen
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# waveform 88 time points, one time point = 1/30000 (s)，so the whole time scale of waveform = 88/30000 (s) = 2.93 (ms)

# ------- NEED CHANGE -------
mice_name = '20230602_Syt2_conditional_tremor_mice2_lateral'
QC_method = 'QC_ISI_violation'  # Without_QC/QC_ISI_violation
sorting_method = 'Easysort'  # Easysort/Previous_sorting
neurons_id = pd.read_csv(rf'C:\Users\zyh20\Desktop\Research\01_ET_data_analysis\Research\ISI_distribution\{sorting_method}\{QC_method}\{mice_name}\00_seperated_pattern_neuron.csv')
neurons_id = neurons_id['seperated_pattern_neuron'].to_numpy()
channels_id = [2]  # interested channels id
save_path = rf'C:\Users\zyh20\Desktop\Research\01_ET_data_analysis\Research\waveform\{sorting_method}\{QC_method}\{mice_name}'  

# ------- NO NEED CHANGE -------
main_path = rf'E:\xinchao\Data\useful_data\{mice_name}\Sorted\Easysort\waveforms\extensions\templates'
waveform = np.load(main_path+'/average.npy')  # Load average waveform, this is a 3D numpy array, shape: (251, 88, 384)
print(waveform.shape)

def channel_neurons(channel_id):
    chan_wav = waveform[:, :, channel_id]  # Extract each slice along the last dimension of the 3D numpy array
    for i in range(chan_wav.shape[0]):
        plt.plot(chan_wav[i, :]) # each column is a voltage time series in a channel of this neuron

    plt.title('Channel id {0}\n Channel-Neurons: Averaged Waveform through Whole Time from Different Neuron'.format(channel_id))
    plt.show()

def neuron_responsed_channel(neuron_id):
    neuron_wav = waveform[neuron_id, :, :]
    for i in range(neuron_wav.shape[1]):
        waveform_channel = neuron_wav[:, i]
        if not np.all(waveform_channel == 0):
            time_points = np.linspace(0, 2.93, 88)
            plt.plot(time_points, waveform_channel)  # each column is a voltage time series in a channel of this neuron
            plt.xlabel('time (ms)')
            plt.title(f'{neuron_id}_Averaged Waveform'.format(neuron_id))
            plt.savefig(save_path+f"/neuron_{neuron_id}_channel_{i}.png",dpi=600,bbox_inches = 'tight')
            plt.clf()

def neuron_channels_overlap(neuron_id):
    neuron_wav = waveform[neuron_id, :, :]
    for i in range(neuron_wav.shape[1]):
        time_points = np.linspace(0, 2.93, 88)
        plt.plot(time_points, neuron_wav[:, i])  # each column is a voltage time series in a channel of this neuron
        
    plt.xlabel('time (ms)')
    plt.title('Unit id {0}\n Neuron-Channels: Averaged Waveform through Whole Time in Different Channel'.format(neuron_id))
    plt.savefig(save_path+f"/neuron_{neuron_id}_channel_overlap.png",dpi=600,bbox_inches = 'tight')
    plt.clf()

def interested_neurons(neurons_id):
    plt.figure(figsize=(10, 6))
    for i in neurons_id:
        neuron_responsed_channel(i)
        neuron_channels_overlap(i)

def interested_channels(channels_id):
    for i in channels_id:
        channel_neurons(i)

interested_neurons(neurons_id)
#interested_channels(channels_id)