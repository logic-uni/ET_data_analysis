"""
# coding: utf-8
@author: Yuhao Zhang
last updated: 04/16/2025
data from: Xinchao Chen
"""

import pandas as pd
import numpy as np
import cupy as cp
from scipy.signal import butter, sosfiltfilt, hilbert
import matplotlib.pyplot as plt
import os

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

# truncated focused time interval
#trunc_left = 529
#trunc_right = 692
#lfp_trunc = lfp_data[:, trunc_left*fs:trunc_right*fs]
# -------------------------------
# Parameters
# -------------------------------
fs = 30000
lowcut = 0.8
highcut = 40.0

# -------------------------------
# Band-pass filter (CPU)
# -------------------------------
def cpu_bandpass_filter(signal, fs, lowcut, highcut, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    if high >= 1.0 or low <= 0 or high <= low:
        raise ValueError("Invalid filter frequency range.")
    
    sos = butter(order, [low, high], btype='band', output='sos')
    
    # Replace NaN just in case
    signal = np.nan_to_num(signal)
    
    # Stability check
    if len(signal) < (order * 3):
        raise ValueError("Signal too short for filtering.")
    
    return sosfiltfilt(sos, signal)

# -------------------------------
# Extract spike phases (CPU filtering & hilbert, GPU for indexing)
# -------------------------------
def extract_spike_phases(lfp, spike_times, fs, ch):
    # Filter and Hilbert on CPU
    filtered = cpu_bandpass_filter(lfp, fs, lowcut, highcut)
    print("Any NaN in filtered LFP:", np.isnan(filtered).any())
    if np.any(np.isnan(filtered)):
        print(f"Channel {ch+1}: Filtered signal contains NaN.")
    if np.all(filtered == 0):
        print(f"Channel {ch+1}: Filtered signal is all zeros.")
    analytic = hilbert(filtered)
    phase = np.angle(analytic)
    corrected_phase = (phase + np.pi / 2) % (2 * np.pi)

    # Extract phases at spike times
    spike_indices = np.searchsorted(np.arange(len(phase)) / fs, spike_times)
    spike_indices = np.clip(spike_indices, 0, len(phase) - 1)
    return corrected_phase[spike_indices]

# -------------------------------
# Polarity Index (GPU)
# -------------------------------
def compute_polarity_index_gpu(phases):
    if len(phases) == 0:
        return 0.0
    phases_gpu = cp.asarray(phases)
    vectors = cp.exp(1j * phases_gpu)
    return float(cp.abs(cp.sum(vectors)) / len(phases))

# -------------------------------
# Save Polar Histogram (Matplotlib)
# -------------------------------
def save_polar_hist(phases, channel_idx, polarity_index, unit_id):
    if len(phases) == 0 or np.isnan(phases).any():
        print(f"Channel {channel_idx+1}: Skipped (no valid phases).")
        return
    os.makedirs(save_path, exist_ok=True)
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(4, 4))
    ax.hist(phases, bins=30, density=True)
    ax.set_title(f'Channel {channel_idx+1}\nPI = {polarity_index:.2f}', fontsize=10)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'Neuron_id_{unit_id}_channel_{channel_idx+1}.png'))
    plt.close(fig)

# -------------------------------
# Main processing loop
# -------------------------------
def process_all_channels(lfp_data, spike_times, fs, unit_id):
    n_channels = lfp_data.shape[0]
    polarity_indices = []

    for ch in range(n_channels):
        print(f"Processing channel {ch+1}/{n_channels}...")
        try:
            spike_phases = extract_spike_phases(lfp_data[ch], spike_times, fs, ch)
            if len(spike_phases) == 0 or np.isnan(spike_phases).any():
                polarity_indices.append(np.nan)
                continue
            pi = compute_polarity_index_gpu(spike_phases)
            polarity_indices.append(pi)
            save_polar_hist(spike_phases, ch, pi, unit_id)
        except Exception as e:
            print(f"Channel {ch+1}: Error -> {e}")
            polarity_indices.append(np.nan)

    return polarity_indices

def singleneuron_spiketimes(id):
    x = np.where(identities == id)
    y=x[0]
    spike_times=np.empty(len(y))
    for i in range(0,len(y)):
        z=y[i]
        spike_times[i]=times[z]/fs
    return spike_times

def main():
    for index, row in neurons.iterrows():
        unit_id = row['cluster_id']
        spike_times = singleneuron_spiketimes(unit_id)
        fr = len(spike_times)/(times[-1] / fs)
        if fr > 2:  # filter fr
            spike_times_trunc = spike_times[(spike_times >= trunc_left) & (spike_times < trunc_right)]
            print(f"LFP duration: {lfp_trunc.shape[1]/fs} seconds")
            if len(spike_times_trunc) != 0:  # filter not firing in this interval
                print(f"Spike times range: {spike_times_trunc.min()} to {spike_times_trunc.max()} seconds")
                print(f"Neuron id:{unit_id}, computing spike phase...")
                polarity_indices = process_all_channels(lfp_trunc, spike_times_trunc, fs, unit_id)
                print("Done! Polarity Indices:", polarity_indices)

def speci_neuron_phase():
    spike_times = singleneuron_spiketimes(116)
    polarity_indices = process_all_channels(lfp_data, spike_times, fs, 116)
    print("Done! Polarity Indices:", polarity_indices)

#main()
speci_neuron_phase()