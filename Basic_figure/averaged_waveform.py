"""
# coding: utf-8
@author: Yuhao Zhang
last updated: 02/27/2025
data from: Xinchao Chen
"""
import numpy as np
import matplotlib.pyplot as plt

# ------- NEED CHANGE -------
mice_name = '20230604_Syt2_conditional_tremor_mice2_medial'
neurons_id = [50]  # interested neurons id
channels_id = [2]  # interested channels id

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

def neuron_channels(neuron_id):
    neuron_wav = waveform[neuron_id, :, :]
    for i in range(neuron_wav.shape[1]):
        plt.plot(neuron_wav[:, i]) # each column is a voltage time series in a channel of this neuron

    plt.title('Unit id {0}\n Neuron-Channels: Averaged Waveform through Whole Time in Different Channel'.format(neuron_id))
    plt.show()

def interested_neurons(neurons_id):
    for i in neurons_id:
        neuron_channels(i)

def interested_channels(channels_id):
    for i in channels_id:
        channel_neurons(i)

interested_neurons(neurons_id)
interested_channels(channels_id)