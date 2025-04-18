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
from sklearn.decomposition import PCA
from elephant.conversion import BinnedSpikeTrain
np.set_printoptions(threshold=np.inf)
np.seterr(divide='ignore',invalid='ignore')

# Lobule
#20230113_littermate    Lobules IV-V
#20230523_Syt2_conditional_tremor_mice1    Lobule III  Lobule II
#20230604_Syt2_conditional_tremor_mice2_medial   Lobule III
#20230602_Syt2_conditional_tremor_mice2_lateral  Lobule III
#20230623_Syt2_conditional_tremor_mice4  Lobule III  Lobule II
#  DCN
#20230604_Syt2_conditional_tremor_mice2_medial  xinchaosort  Vestibulocerebellar nucleus
#20230113_littermate   Interposed nucleus

region_name = 'Lobule III'
fr_bin = 1  # unit: ms
avoid_spikemore1 = True # 避免1ms的bin里有多个spike,对于1ms内多个spike的，强行置1

## ------ Easysort ------
# --- NEED CHANGE ---
mice_name = '20230602_Syt2_conditional_tremor_mice2_lateral'
mapping_file = 'unit_ch_dep_region_QC_isi_violations_ratio_pass_rate_60.17316017316017%.csv'
QC_method = 'Without_QC'  # Without_QC/QC_ISI_violation/etc
# --- NO NEED CHANGE ---
sorting_path = rf'E:\xinchao\Data\useful_data\NP1\{mice_name}\Sorted\Easysort\results_KS2\sorter_output'
save_path = rf'C:\Users\zyh20\Desktop\Research\01_ET_data_analysis\Research\spectrum_analysis\NP1\Easysort\{QC_method}\{mice_name}'  
neurons = pd.read_csv(rf'E:\xinchao\Data\useful_data\NP1\{mice_name}\Sorted\Easysort\mapping\{mapping_file}')  # different sorting have different nueron id

# ------ Xinchao_sort ------
# --- NEED CHANGE ---
#mice_name = '20230602_Syt2_conditional_tremor_mice2_lateral'
# --- NO NEED CHANGE ---
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

# get single neuron spike train
def singleneuron_spiketimes(id):
    x = np.where(identities == id)
    y=x[0]
    spike_times=np.zeros(len(y))
    for i in range(0,len(y)):
        z=y[i]
        spike_times[i]=times[z]/sample_rate
    return spike_times

def popu_sptrain_trial(neuron_ids,marker_start,marker_end):
    for j in range(len(neuron_ids)): #第j个neuron
        spike_times = singleneuron_spiketimes(neuron_ids[j])
        spike_times_trail = spike_times[(spike_times > marker_start) & (spike_times < marker_end)]
        spiketrain = neo.SpikeTrain(spike_times_trail,units='sec',t_start=marker_start, t_stop=marker_end)
        fr = BinnedSpikeTrain(spiketrain, bin_size=fr_bin*pq.ms,tolerance=None)
        if avoid_spikemore1 == False:
            one_neruon = fr.to_array().astype(int)[0]
        else:
            fr_binar = fr.binarize()  # 对于可能在1ms内出现两个spike的情况，强制置为该bin下即1ms只能有一个spike
            one_neruon = fr_binar.to_array().astype(int)[0]
        
        if j == 0:
            neurons = one_neruon
        else:
            neurons = np.vstack((neurons, one_neruon))
    return neurons

def spectrum_analysis(region,popu_ids):
    neuron_num = len(popu_ids)
    squmat_interval = neuron_num  #单位ms
    trial_spiketr_len = 30000/fr_bin
    truncated_trial = int(trial_spiketr_len - (trial_spiketr_len % squmat_interval))   # 一个trial切分成多个小正方形矩阵，余下的不要
    for index, row in treadmill.iterrows():
        trial_start = row['time_interval_left_end']
        trial_end = row['time_interval_right_end']
        status = row['run_or_stop']
        if status == 0:
            color = 'b'
        else:
            color = 'r'
        data = popu_sptrain_trial(popu_ids,trial_start,trial_end)
        print(data.shape)
        for squleft in range(0,squmat_interval,truncated_trial):
            square_data = data[:,squleft:squleft+squmat_interval]
            print(square_data.shape)
            eigenvalues, eigenvectors = np.linalg.eig(square_data)
            print(np.abs(eigenvalues))
            plt.scatter(eigenvalues.real, eigenvalues.imag, c=color, marker='o')
    
    plt.title(f'spectrum_analysis_{region}')
    plt.text(0.05, 0.95, f'Each Truncated Time Length: {squmat_interval} ms', 
                    ha='left', va='top', transform=plt.gca() .transAxes, fontsize=10)
    plt.savefig(save_path+f"/{region}.png",dpi=600,bbox_inches = 'tight')
    plt.clf()

def main():
    # 谱分析画布固定内容
    plt.figure(figsize=(10, 10))
    plt.xlabel('Real')
    plt.xlim(-5,5)
    plt.ylim(-5,5)
    plt.ylabel('Imaginary')
    plt.grid(True)
    plt.gca().set_aspect('equal')
    theta = np.linspace(0, 2 * np.pi, 100)
    x = 1 * np.cos(theta)
    y = 1 * np.sin(theta)
    plt.plot(x, y,c='black')

    result = neurons.groupby('region')['cluster_id'].apply(list).reset_index(name='cluster_ids')
    print(result)
    for index, row in result.iterrows():
        region = row['region']
        popu_ids = row['cluster_ids']
        if region == region_name:
            spectrum_analysis(region,popu_ids)

    #max_eigen1 = covmatr_spectrum(data[:,(pert_s-40):(pert_s)])
    #max_eigen2 = covmatr_spectrum(data[:,(pert_s):(pert_s+40)])
    #eigen_vector_included_angle(max_eigen1,max_eigen2)
    #covmatr_spectrum(data[:,(pert_s-40):(pert_s)])
    #covmatr_spectrum(data[:,(pert_s-40):(pert_e+150)])
    #SVD_matrix_spectrum(data[:,(pert_s):(pert_s+40)])

main()
