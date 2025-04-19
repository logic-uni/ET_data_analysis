"""
# coding: utf-8
@author: Yuhao Zhang
last updated: 04/05/2025
data from: Xinchao Chen
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
np.set_printoptions(threshold=np.inf)

mice_name = '20250310_VN_tremor'
segment = 15 # unit s

# --- NO NEED CHANGE ---
main_path = f"/data1/zhangyuhao/xinchao_data/NP2/{mice_name}/"
save_path = f"/home/zhangyuhao/Desktop/Result/ET/RasterPlot/{mice_name}/special_isi_neurons/"  # all_neurons  selected_neurons  special_isi_neurons

# --------- Main Program ----------
# behavior
loadcell_sp_rate = 10593.2 # loadcell sample rate
loadcell = np.load(main_path + '/Marker/motion_marker.npy')
loadcell = loadcell[0]
# electrophysiology
ephys_sp_rate = 30000 #spikeGLX neuropixel sample rate
identities = np.load(main_path + '/Sorted/spike_clusters.npy') # time series: unit id of each spike
times = np.load(main_path + '/Sorted/spike_times.npy')  # time series: spike time of each spike
neurons = pd.read_csv(main_path + '/Sorted/cluster_group.tsv', sep='\t')  
print(neurons)
print("Test if electrophysiology duration is equal to treadmill duration ...")
elec_length = times[-1] / ephys_sp_rate
behav_length = loadcell.shape[0] / loadcell_sp_rate
print(f"Electrophysiology duration: {elec_length}")
print(f"Loadcell duration: {behav_length}")

#popu_ids = neurons['cluster_id'].to_numpy()
popu_ids = np.array([4,5])

#### spike train & firing rates
# get single neuron spike train
def singleneuron_spiketimes(id):
    x = np.where(identities == id)
    y=x[0]
    spike_times=np.zeros(len(y))
    for i in range(0,len(y)):
        z=y[i]
        spike_times[i]=times[z]/ephys_sp_rate
    return spike_times

# plot single neuron spike train
def Rasterplot_singleneuron(spike_times,region,unit_id,trial_type):
    y=np.empty(len(spike_times))      
    plt.plot(spike_times,y, '|', color='gray') 
    plt.title('neuron 15') 
    plt.xlabel("time") 
    plt.xlim(0,t)
    plt.savefig(save_path+f"/{region}_neuron_{unit_id}_{trial_type}.png",dpi=600,bbox_inches = 'tight')
    plt.clf()

def Rasterp_popu_behav(spike_times, start_time, end_time):
    # 创建画布和子图 gridspec 控制子图高度比例
    # if all neurons, set figsize=(35, 25) 'height_ratios': [5, 1]}
    # if selected neurons, set figsize=(35,10) 'height_ratios': [3, 1]}
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(35,10))  
    nu_num = len(spike_times)
    for i in range(0, nu_num):
        ax1.plot(spike_times[i], np.repeat(i, len(spike_times[i])), '|', color='gray')
    ax1.set_yticks(np.arange(0,nu_num,1))
    ax1.set_xticks(np.arange(0,segment+1,5))
    ax1.set_title(f'Spike Train {start_time} - {end_time}') 
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Neurons")

    start_sample = int(start_time * loadcell_sp_rate)
    end_sample = int(end_time * loadcell_sp_rate)
    time_axis = np.linspace(0, segment, end_sample - start_sample)  # 创建实际时间坐标
    ax2.plot(time_axis,loadcell[int(start_time*loadcell_sp_rate):int(end_time*loadcell_sp_rate)])  # 没动就是平的，动了才会变化，向上向下分别代表压力传感器的方向
    ax2.set_yticks([])
    ax2.set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(f"{save_path}/{start_time}-{end_time}.png", dpi=300,bbox_inches='tight')
    plt.close()

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
    trunc = int(elec_length - elec_length % segment)
    for start in range(0, trunc, segment):
        end = start + segment
        spike_times = popu_sptime_trial(popu_ids,start,end)  # start,end unit s
        Rasterp_popu_behav(spike_times,start,end)

main()