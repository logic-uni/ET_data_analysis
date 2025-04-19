"""
# coding: utf-8
@author: Yuhao Zhang
last updated: 04/06/2025
data from: Xinchao Chen
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from mpl_toolkits.axes_grid1 import make_axes_locatable
np.set_printoptions(threshold=np.inf)

# --- NEED CHANGE ---
region_name = 'Lobule III'

#20230113_littermate  Simple lobule  Lobules IV-V  Interposed nucleus  Superior vestibular nucleus   Spinal vestibular nucleus
#20230523_Syt2_conditional_tremor_mice1    Lobule III  Lobule II
#20230604_Syt2_conditional_tremor_mice2_medial   Lobule III  Vestibulocerebellar nucleus  Medial vestibular nucleus  Spinal vestibular nucleus
#20230602_Syt2_conditional_tremor_mice2_lateral  Lobule III  Lobules IV-V  Superior vestibular nucleus  Medial vestibular nucleus
#20230623_Syt2_conditional_tremor_mice4  Lobule III  Lobule II  Medial vestibular nucleus

## Easysort  NEED CHANGE
mice_name = '20230602_Syt2_conditional_tremor_mice2_lateral'
mapping_file = 'unit_ch_dep_region_QC_isi_violations_ratio_pass_rate_60.17316017316017%.csv'
QC_method = 'QC_ISI_violation'  # Without_QC/QC_ISI_violation/etc
# Easysort   NO NEED CHANGE
sorting_path = rf'E:\xinchao\Data\useful_data\NP1\{mice_name}\Sorted\Easysort\results_KS2\sorter_output'
save_path = rf'C:\Users\zyh20\Desktop\Research\01_ET_data_analysis\Research\Raster_plot\NP1\Easysort\{QC_method}\{mice_name}\selected_neuron'
neurons = pd.read_csv(rf'E:\xinchao\Data\useful_data\NP1\{mice_name}\Sorted\Easysort\mapping\{mapping_file}')  # different sorting have different nueron id

# Xinchao_sort  NEED CHANGE
#mice_name = '20230602_Syt2_conditional_tremor_mice2_lateral'
# Xinchao_sort  NO NEED CHANGE
#sorting_path = rf'E:\xinchao\Data\useful_data\NP1\{mice_name}\Sorted\Xinchao_sort'
#save_path = rf'C:\Users\zyh20\Desktop\Research\01_ET_data_analysis\Research\spectrum_analysis\NP1\Xinchao_sort\{mice_name}'  
#neurons = pd.read_csv(sorting_path + '/neuron_id_region_firingrate.csv')  # different sorting have different nueron id

# --------- Main Program ----------
treadmill = pd.read_csv(rf'E:\xinchao\Data\useful_data\NP1\{mice_name}\Marker\treadmill_move_stop_velocity_segm_trial.csv',index_col=0)
treadmill_origin = pd.read_csv(rf'E:\xinchao\Data\useful_data\NP1\{mice_name}\Marker\treadmill_move_stop_velocity.csv',index_col=0)
# electrophysiology
sample_rate = 30000 #spikeGLX neuropixel sample rate
identities = np.load(sorting_path + '/spike_clusters.npy') # time series: unit id of each spike
times = np.load(sorting_path + '/spike_times.npy')  # time series: spike time of each spike
print(neurons)
print("Test if electrophysiology duration is equal to treadmill duration ...")
elec_dura = (times[-1]/sample_rate)[0]
treadmill_dura = treadmill_origin['time_interval_right_end'].iloc[-1]
print(f"Electrophysiology duration: {elec_dura}")
print(f"Treadmill duration: {treadmill_dura}")

#### spike train & firing rates
# get single neuron spike train
def singleneuron_spiketimes(id):
    x = np.where(identities == id)
    y=x[0]
    spike_times=np.zeros(len(y))
    for i in range(0,len(y)):
        z=y[i]
        spike_times[i]=times[z]/sample_rate
    return spike_times

# plot single neuron spike train
def raster_plot_singleneuron(spike_times,region,unit_id,trial_type):
    y=np.empty(len(spike_times))      
    plt.plot(spike_times,y, '|', color='gray') 
    plt.title('neuron 15') 
    plt.xlabel("time") 
    plt.xlim(0,t)
    plt.savefig(save_path+f"/{region}_neuron_{unit_id}_{trial_type}.png",dpi=600,bbox_inches = 'tight')
    plt.clf()

def RP_neurons_trial(spike_times, trial_num, trial_type,popu_ids):
    nu_num = len(spike_times)
    colors = plt.cm.gist_ncar(np.linspace(0, 1, nu_num))
    for i in range(nu_num):
        plt.plot(spike_times[i], np.repeat(i, len(spike_times[i])), '|', color=colors[i], label=f'Neuron {popu_ids[i]}') 
    plt.title(f'Spike Train - Trial {trial_num}') 
    plt.xlabel("Time (s)")
    plt.ylabel("Neurons")
    plt.legend(
        loc='upper right',
        fontsize='small',
        title='Neuron IDs',
        ncol=2,  # 分多列显示
        framealpha=0.7  # 增加透明度防止遮挡
    )
    plt.savefig(save_path+f"/{region_name}_trial{trial_num}_{trial_type}.png", dpi=600, bbox_inches='tight')
    plt.clf()

# PETH: peri-event time histogram  事件周围时间直方图
def PETH_singleneuron(firing_rate,duration):
    plt.rcParams['xtick.direction'] = 'in'#将x轴的刻度线方向设置向内
    plt.rcParams['ytick.direction'] = 'in'#将y轴的刻度线方向设置向内
    plt.plot(firing_rate)
    plt.axvspan(0, duration, color='gray', alpha=0.1)
    plt.xlim((-1, 50))
    plt.xticks(np.arange(-1,50,1))
    plt.ylabel('Firing rate (spikes/s)')
    plt.xlabel('Time (s)')
    plt.title('Neuron %d, Push rod'%id)
    plt.show()
    
def PETH_heatmap_1(data): #未调试
    mean_histograms = data.mean(dim="stimulus_presentation_id")
    print(mean_histograms)
    # plot
    fig, ax = plt.subplots(figsize=(8, 8))
    c = ax.pcolormesh(
        mean_histograms["time_relative_to_stimulus_onset"], 
        np.arange(mean_histograms["unit_id"].size),
        mean_histograms, 
        vmin=0,
        vmax=1
    )
    plt.colorbar(c) 
    ax.set_ylabel("unit", fontsize=24)
    ax.set_xlabel("time relative to movement onset (s)", fontsize=24)
    ax.set_title("PSTH for units", fontsize=24)
    plt.show()

def PETH_heatmap_2(data,id):  #已调试
    data_minus_mean=data-np.mean(data)
    # plot
    fig, ax = plt.subplots(figsize=(12, 12))
    div = make_axes_locatable(ax)
    cbar_axis = div.append_axes("right", 0.2, pad=0.05)
    img = ax.imshow(
        data_minus_mean, 
        extent=(-2,30,0,len(data)),  #前两个值是x轴的时间范围，后两个值是y轴的值
        interpolation='none',
        aspect='auto',
        vmin=2, 
        vmax=20 #热图深浅范围
    )
    plt.colorbar(img, cax=cbar_axis)

    cbar_axis.set_ylabel('Spike counts', fontsize=20)
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.set_ylabel('Trials', fontsize=20)
    reltime = np.arange(-2, 30, 1)
    ax.set_xticks(np.arange(-2, 30, 1))
    ax.set_xticklabels([f'{mp:1.3f}' for mp in reltime[::1]], rotation=45)
    ax.set_xlabel('Time(s) Move: 0s', fontsize=20)
    ax.set_title('Spike counts Neuron id %d'%id, fontsize=20)
    plt.show()

def PETH_heatmap_shorttime(data,id):
    t0=-0.3
    t1=0.5
    #data_logtrans=np.log2(data+1)
    #data_minus_mean=data_logtrans-2.2
    data_minus_mean=data-np.median(data)
    print(data_minus_mean)
    # plot
    fig, ax = plt.subplots(figsize=(12, 12))
    div = make_axes_locatable(ax)
    cbar_axis = div.append_axes("right", 0.2, pad=0.05)
    img = ax.imshow(
        data_minus_mean, 
        extent=(t0,t1,0,len(data)),  #前两个值是x轴的时间范围，后两个值是y轴的值
        interpolation='none',
        aspect='auto',
        vmin=0, 
        vmax=3 #热图深浅范围
    )
    plt.colorbar(img, cax=cbar_axis)

    cbar_axis.set_ylabel('spike count', fontsize=20)
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.set_ylabel('Trials', fontsize=20)
    reltime = np.arange(t0,t1, 0.05)
    ax.set_xticks(np.arange(t0,t1, 0.05))
    ax.set_xticklabels([f'{mp:1.3f}' for mp in reltime[::1]], rotation=45,fontsize=12)
    ax.set_xlabel('Time(s) Move: 0s', fontsize=20)
    ax.set_title('Spike Counts Neuron id %d'%id, fontsize=20)
    plt.show()

def popu_sptime_trial(neuron_ids,start,end):
    popu_sptime = []
    for j in range(len(neuron_ids)): #第j个neuron
        spike_times = singleneuron_spiketimes(neuron_ids[j])
        spike_times_trail = spike_times[(spike_times > start) & (spike_times < end)]
        align_nu_times = spike_times_trail - start
        popu_sptime.append(align_nu_times)
    return popu_sptime

def main():
    plt.figure(figsize=(55, 15))
    result = neurons.groupby('region')['cluster_id'].apply(list).reset_index(name='cluster_ids')
    print(result)
    for index, row in result.iterrows():
        region = row['region']
        popu_ids = row['cluster_ids']
        if region == region_name and len(popu_ids) > 3:
            # enumarate trials
            for index, row in treadmill.iterrows():
                start = row['time_interval_left_end']
                end = row['time_interval_right_end']
                if row['run_or_stop'] == 0:
                    stage = 'static'
                else:
                    stage = 'running'
                # for selected neurons, replace popu_ids with specific neurons id
                popu_ids = np.array([291, 292, 293, 294, 297, 299, 300, 302, 303, 304, 305])
                spike_times_trail = popu_sptime_trial(popu_ids,start,start+15)
                RP_neurons_trial(spike_times_trail,index,stage,popu_ids)

main()