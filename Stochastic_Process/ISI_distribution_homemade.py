"""
# coding: utf-8
@author: Yuhao Zhang
last updated: 02/27/2025
data from: Xinchao Chen
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import expon
np.set_printoptions(threshold=np.inf)
np.seterr(divide='ignore',invalid='ignore')

# ------- NEED CHANGE -------
mice_name = '20230113_littermate'
mapping_file = 'unit_ch_dep_region_QC_isi_violations_ratio_pass_rate_55.30864197530864%.csv'
sorting_method = 'Easysort'  # Easysort/Previous_sorting
mice_type = 'cond_ET'        # PV_Syt2/littermate/cond_ET
stat_fit = True              # True/False
fr_filter = False             # False/1

# ------- NO NEED CHANGE -------
### path
main_path = rf'E:\xinchao\Data\useful_data\{mice_name}\Sorted\Easysort'
sorting_path = main_path + '/results_KS2/sorter_output'
save_path = rf'C:\Users\zyh20\Desktop\Research\01_ET_data_analysis\Research\ISI_distribution\{sorting_method}\{mice_name}'  
treadmill = pd.read_csv(rf'E:\xinchao\Data\useful_data\{mice_name}\Marker\treadmill_move_stop_velocity.csv',index_col=0)

### electrophysiology
sample_rate = 30000 #spikeGLX neuropixel sample rate
identities = np.load(sorting_path + '/spike_clusters.npy') # time series: unit id of each spike
times = np.load(sorting_path + '/spike_times.npy')  # time series: spike time of each spike
neurons = pd.read_csv(main_path + f'/mapping/{mapping_file}')
print(neurons)
print("Test if electrophysiology duration is equal to treadmill duration ...")
elec_dura = (times[-1]/sample_rate)[0]
treadmill_dura = treadmill['time_interval_right_end'].iloc[-1]
print(f"Electrophysiology duration: {elec_dura}")
print(f"Treadmill duration: {treadmill_dura}")

### ISI parameters
fr_filter = 1 # 滤掉fr小于1spike/s
fr_fil = 30/fr_filter
cutoff_distr = 0.25 # cutoff_distr=0.25代表截断ISI分布大于0.25s的
histo_bin_num = 100

# ------- Main Program -------
# get single neuron spike train
def singleneuron_spiketrain(id):
    x = np.where(identities == id)
    y=x[0]
    spike_times=np.empty(len(y))
    for i in range(0,len(y)):
        z=y[i]
        spike_times[i]=times[z]/sample_rate
    return spike_times
'''
plt.savefig(save_path+f"/{region}_neuron_{id}_runtrials.png",dpi=600,bbox_inches = 'tight')
plt.clf()

if plot_mode == 'each_n_t':
plt.savefig(save_path+f"/each_neuron_trials/Neuron{id}_Trial{trial_num}.png",dpi=600,bbox_inches = 'tight')
plt.clf()
'''
def plot(spike_times,start,end):
    spike_times_trail = spike_times[(spike_times > start) & (spike_times < end)]
    intervals = np.array([])
    if len(spike_times_trail) > fr_fil: 
        intervals = np.diff(spike_times_trail)  # 计算时间间隔
        # 绘制时间间隔的直方图
        if cutoff_distr != None:
            intervals = intervals[intervals <= cutoff_distr]
            intervals = intervals[intervals > 9.99999999e-04]
            if len(intervals) != 0:  #截断后可能导致没有interval
                plt.hist(intervals, bins=histo_bin_num, density=False,alpha=0.6,color=color)  #如果只看0.25s间隔以内细致的，画counts更适合
        else:
            plt.hist(intervals, bins=20, density=True, alpha=0.6,color=color)
    return intervals

def ET_trials(spike_times,neuron_intervals,trial_type):
    marker = treadmill
    stop_trials = marker[marker['run_or_stop'] == 0]
    run_trials = marker[marker['run_or_stop'] == 1]
    # stoptrials
    for trial in range(len(stop_trials)):
        marker_start = stop_trials['time_interval_left_end'].iloc[trial]
        marker_end = stop_trials['time_interval_right_end'].iloc[trial]
        if marker_end - marker_start > 30:
            trunc_end = marker_start + 30
            intervals_trial = plot(spike_times,marker_start,trunc_end)
        neuron_intervals.extend(intervals_trial)
    plt.savefig(save_path+f"/each_neuron_trials/Neuron{id}_Trial{trial}.png",dpi=600,bbox_inches = 'tight')
    # runtrials
    for trial in range(len(run_trials)):
        marker_start = run_trials['time_interval_left_end'].iloc[trial]
        marker_end = run_trials['time_interval_right_end'].iloc[trial]
        if marker_end - marker_start > 30:
            trunc_end = marker_start + 30
            intervals_trial = plot(spike_times,marker_start,trunc_end)
        neuron_intervals.extend(intervals_trial)
    plt.savefig(save_path+f"/each_neuron_trials/Neuron{id}_Trial{trial}.png",dpi=600,bbox_inches = 'tight')

    return neuron_intervals

def littermate_trials(spike_times,neuron_intervals,color,id):
    marker = treadmill
    stop_trials = marker[marker['run_or_stop'] == 0]
    run_trials = marker[marker['run_or_stop'] == 1]
    # stoptrials
    for trial in range(len(stop_trials)):
        marker_start = run_trials['time_interval_left_end'].iloc[trial]
        marker_end = run_trials['time_interval_right_end'].iloc[trial]
        intervals_trial = plot(spike_times,marker_start,marker_end)
        neuron_intervals.extend(intervals_trial)
    plt.savefig(save_path+f"/each_neuron_trials/Neuron{id}_Trial{trial}.png",dpi=600,bbox_inches = 'tight')
    # stoptrials
    for trial in range(len(run_trials)):
        marker_start = run_trials['time_interval_left_end'].iloc[trial]
        marker_end = run_trials['time_interval_right_end'].iloc[trial]
        intervals_trial = plot(spike_times,marker_start,marker_end)
        neuron_intervals.extend(intervals_trial)
    plt.savefig(save_path+f"/each_neuron_trials/Neuron{id}_Trial{trial}.png",dpi=600,bbox_inches = 'tight')
    return neuron_intervals

def PVSyt2_trials(neuron_a_id,neuron_b_id,marker):
    neuron_intervals = 0
    return neuron_intervals

def stat_fit_plot(neuron_intervals,region_name):
    lambdas = []
        ## 画完当前neuron所有trial的hist之后，对当前neuron整体的ISI拟合指数分布
    if len(neuron_intervals) != 0 and cutoff_distr == None:
        params = expon.fit(neuron_intervals, floc=0)  # 固定起始点为0
        lambda_fit = 1 / params[1]  # 拟合的lambda值
        # 绘制拟合的指数分布曲线
        x = np.linspace(0, max(neuron_intervals), 100)
        plt.plot(x, expon.pdf(x, *params), 'b-', lw=0.2)
        lambdas.append(lambda_fit)  # 保存lambda值 each lambda represent a neurons firing strength in different run trials
        
    plt.xlabel('Inter-spike Interval (s)')
    plt.ylabel('Probability Density')
    plt.title(f'ISI Distribution_{region_name}')
    # 画完所有的ISI分布后 计算平均 lambda 值
    lambda_mean = np.mean(lambdas)
    neuron_infig_num = len(lambdas)
    # 在图的右上角显示平均的lambda值
    plt.text(0.95, 0.95, f'{mice_name}\n\nfiring filter: 1 spike/s\n\nneurons num after filter: {neuron_infig_num}\n\nMean λ: {lambda_mean:.2f}\n\nhisto bin: 20\n\n{plot_mode} trials', ha='right', va='top', transform=plt.gca().transAxes)
    # 插入子图表示lambda的分布
    left, bottom, width, height = 0.6,0.25,0.25,0.25
    plt.axes([left,bottom,width,height])
    plt.hist(lambdas, bins=50, density=False, alpha=0.6)
    plt.title(f'lambda distribution')
    plt.xlabel('lambda value')
    plt.ylabel('Prob.')

def ISI_single_neuron_trials(region_name,unit_id):
    spike_times = singleneuron_spiketrain(unit_id)
    neuron_intervals = []  # 用于存储该neuron的所有时间间隔
    # neuron ID - runtrials / neuron ID - stoptrials 
    if mice_type == 'cond_ET':
        neuron_intervals = ET_trials(spike_times,neuron_intervals,unit_id)
    elif mice_type == 'littermate':
        neuron_intervals = littermate_trials(spike_times,neuron_intervals,unit_id)
    elif mice_type == 'PV_Syt2':
        neuron_intervals = PVSyt2_trials(spike_times,neuron_intervals)

def ISI_population_trials(region_name,popu_id):
    plt.figure(figsize=(10, 6))
    neuron_colors = plt.cm.hsv(np.linspace(0, 1, len(region_neuron)))
    for i in range(len(region_neuron)):
        spike_times = singleneuron_spiketrain(region_neuron[i])
        color = neuron_colors[i]  #一个neuron使用同一个color
        neuron_intervals = []  # 用于存储该neuron的所有时间间隔
        neuron_id = region_neuron[i]
        # neuron ID - runtrials / neuron ID - stoptrials 
        if mice_type == 'cond_ET':
            neuron_intervals = ET_runtrials_ISI(spike_times,neuron_intervals,color,neuron_id)  ##需要加颜色修改，这样对于单个neuron不同trial颜色都不同
            neuron_intervals = ET_stoptrials_ISI(spike_times,neuron_intervals,color,neuron_id)
        elif mice_type == 'littermate':
            neuron_intervals = littermate_runtrials_ISI(spike_times,neuron_intervals,color,neuron_id)
            neuron_intervals = littermate_stoptrials_ISI(spike_times,neuron_intervals,color,neuron_id)
        elif mice_type == 'PV_Syt2':
            neuron_intervals = PV_Syt2(spike_times,neuron_intervals)
            neuron_intervals = PV_Syt2(spike_times,neuron_intervals)
    
    # neurons_runtrials
    plt.xlim(-0.01,0.26) #add this when needed
    plt.xlabel('Inter-spike Interval (s)')
    plt.ylabel('Counts')
    plt.title(f'ISI Distribution_{region_name}')
    plt.text(0.95, 0.95, f'{mice_name}\n\nfiring filter > {fr_filter} spike/s\n\nhisto bin: {histo_bin_num}\n\ndistr cutoff > {cutoff_distr}\n\n{trial_type} trials', ha='right', va='top', transform=plt.gca().transAxes)
    plt.savefig(save_path+f"/{trial_type}_trials_{region_name}_cutoff_{cutoff_distr}.png",dpi=600,bbox_inches = 'tight')
    
    if stat_fit == True:  stat_fit_plot(neuron_intervals,region_name)
        
def main(units):
    # single neuron - trials
    for index, row in units.iterrows():
        unit_id = row['cluster_id']
        region_name = row['region']
        ISI_single_neuron_trials(region_name,unit_id)
    # population - trials
    region_neurons = units.groupby('region')['cluster_id'].apply(list).to_dict()
    for region_name, popu_id in region_neurons.items():
        ISI_population_trials(region_name,popu_id)

#main(neurons)