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
mice_name = '20230604_Syt2_conditional_tremor_mice2_medial'
mapping_file = 'unit_ch_dep_region_QC_isi_violations_ratio_pass_rate_72.90836653386454%.csv'
QC_method = 'QC_ISI_violation'  # Without_QC/QC_ISI_violation

# ------- NO NEED CHANGE -------
### parameter
sorting_method = 'Easysort'  # Easysort/Previous_sorting
stat_fit = False             # True/False 是否拟合
fr_filter_trial = 30         # 30/1  30s的一个trial内至少要有30个spike，否则不具有统计意义/30s的一个trial内至少要有2个spike，否则不具有统计意义
cutoff_distr = 250           # 250ms/None  cutoff_distr=0.25代表截断ISI分布大于0.25s的
histo_bin_num = 100          # 统计图bin的个数
fr_filter = 1  

### path
main_path = rf'E:\xinchao\Data\useful_data\NP1\{mice_name}\Sorted\Easysort'
sorting_path = main_path + '/results_KS2/sorter_output'
save_path = rf'C:\Users\zyh20\Desktop\Research\01_ET_data_analysis\Research\ISI_distribution\NP1\{sorting_method}\{QC_method}\{mice_name}'  
save_path_whole_time = save_path + '/whole_time'
treadmill = pd.read_csv(rf'E:\xinchao\Data\useful_data\NP1\{mice_name}\Marker\treadmill_move_stop_velocity_segm_trial.csv',index_col=0)
treadmill_origin = pd.read_csv(rf'E:\xinchao\Data\useful_data\NP1\{mice_name}\Marker\treadmill_move_stop_velocity.csv',index_col=0)
### electrophysiology
sample_rate = 30000 #spikeGLX neuropixel sample rate
identities = np.load(sorting_path + '/spike_clusters.npy') # time series: unit id of each spike
times = np.load(sorting_path + '/spike_times.npy')  # time series: spike time of each spike
neurons = pd.read_csv(main_path + f'/mapping/{mapping_file}')
print(neurons)
print("Test if electrophysiology duration is equal to treadmill duration ...")
elec_dura = (times[-1]/sample_rate)[0]
treadmill_dura = treadmill_origin['time_interval_right_end'].iloc[-1]
print(f"Electrophysiology duration: {elec_dura}")
print(f"Treadmill duration: {treadmill_dura}")

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

def plot(spike_times,start,end,color):
    spike_times_trail = spike_times[(spike_times > start) & (spike_times < end)]
    intervals = np.array([])
    #  30s的一个trial内如果没有30个spike，则不进行统计
    if len(spike_times_trail) > fr_filter_trial: 
        intervals = np.diff(spike_times_trail)  # 计算时间间隔
        intervals = intervals * 1000
        # 绘制时间间隔的直方图
        if cutoff_distr != None:
            intervals = intervals[(intervals > 0.000999999999) & (intervals <= cutoff_distr)]
            if len(intervals) != 0:  #截取区间可能导致没有interval
                plt.hist(intervals, bins=histo_bin_num, density=False,alpha=0.6,color=color)  #如果只看0.25s间隔以内细致的，画counts更适合
        else:
            plt.hist(intervals, bins=20, density=True, alpha=0.6,color=color)
    return intervals

def ISI_single_neuron_trials(region,unit_id,trial_type):
    spike_times = singleneuron_spiketrain(unit_id)
    neuron_intervals = []  # 用于存储该neuron的所有时间间隔
    if trial_type == 'stoptrials':
        trials = treadmill[treadmill['run_or_stop'] == 0]
    elif trial_type == 'runtrials':
        trials = treadmill[treadmill['run_or_stop'] == 1]

    trials_colors = plt.cm.hsv(np.linspace(0, 1, len(trials)))  # each color is a trial
    for trial in range(len(trials)):
        color = trials_colors[trial]
        marker_start = trials['time_interval_left_end'].iloc[trial]
        marker_end = trials['time_interval_right_end'].iloc[trial]
        intervals_trial = plot(spike_times,marker_start,marker_end,color)
        neuron_intervals.extend(intervals_trial)

    plt.xlabel('Inter-spike Interval (ms)')
    plt.ylabel('Counts')
    plt.title(f'ISI Distribution_{region}')
    plt.text(0.95, 0.95, f'Each color is a trial\n\none trial is 30s\n\n{mice_name}\n\nfiring filter > {fr_filter_trial/30} spike/s\n\nhisto bin: {histo_bin_num}\n\ndistr cutoff > {cutoff_distr}\n\n{trial_type}', ha='right', va='top', transform=plt.gca().transAxes)
    plt.savefig(save_path+f"/{region}_neuron_{unit_id}_{trial_type}.png",dpi=600,bbox_inches = 'tight')
    plt.clf()
    return neuron_intervals

def ISI_population_trials(region,popu_id,trial_type):
    neuron_colors = plt.cm.hsv(np.linspace(0, 1, len(popu_id)))
    popu_intervals = []  # 用于存储该neuron的所有时间间隔
    for i in range(len(popu_id)):
        color = neuron_colors[i]  #一个neuron使用同一个color
        unit_id = popu_id[i]
        spike_times = singleneuron_spiketrain(unit_id)
        if trial_type == 'stoptrials':
            trials = treadmill[treadmill['run_or_stop'] == 0]
        elif trial_type == 'runtrials':
            trials = treadmill[treadmill['run_or_stop'] == 1]

        for trial in range(len(trials)):
            marker_start = trials['time_interval_left_end'].iloc[trial]
            marker_end = trials['time_interval_right_end'].iloc[trial]
            intervals_trial = plot(spike_times,marker_start,marker_end,color)
            popu_intervals.extend(intervals_trial)

    plt.xlabel('Inter-spike Interval (s)')
    plt.ylabel('Counts')
    plt.title(f'ISI Distribution_{region}')
    plt.text(0.95, 0.95, f'Each color is a neuron\n\none trial is 30s\n\n{mice_name}\n\nfiring filter > {fr_filter_trial/30} spike/s\n\nhisto bin: {histo_bin_num}\n\ndistr cutoff > {cutoff_distr}\n\n{trial_type}', ha='right', va='top', transform=plt.gca().transAxes)
    plt.savefig(save_path+f"/neurons_{region}_{trial_type}.png",dpi=600,bbox_inches = 'tight')
    plt.clf()
    return popu_intervals

def stat_fit_plot(neuron_intervals,region,trial_type):
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
    plt.title(f'ISI Distribution_{region}')
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
    plt.savefig(save_path+f"/neurons_{region}_{trial_type}.png",dpi=600,bbox_inches = 'tight')

def ISI_single_neuron_session(region,unit_id):
    spike_times = singleneuron_spiketrain(unit_id)
    intervals = np.array([])
    # 如果整个session的spike个数，小于elec_dura电生理时长（s），即每秒钟的spike个数小于1，则不进行统计
    if len(spike_times) > (fr_filter*elec_dura): 
        intervals = np.diff(spike_times)  # 计算时间间隔
        intervals = intervals * 1000   # 转为ms单位
        # 绘制时间间隔的直方图
        if cutoff_distr != None:
            intervals = intervals[(intervals > 0.000999999999) & (intervals <= cutoff_distr)]
            if len(intervals) != 0:  #截取区间可能导致没有interval
                plt.hist(intervals, bins=histo_bin_num, density=False,alpha=0.6)  #如果只看0.25s间隔以内细致的，画counts更适合
        else:
            plt.hist(intervals, bins=20, density=True, alpha=0.6)

    plt.xlabel('Inter-spike Interval (ms)')
    plt.ylabel('Counts')
    plt.title(f'unit id {unit_id}')
    plt.text(0.95, 0.95, f'firing filter > {fr_filter} spike/s\n\nhisto bin: {histo_bin_num}\n\ndistr cutoff > {cutoff_distr}\n\n', ha='right', va='top', transform=plt.gca().transAxes)
    plt.savefig(save_path_whole_time+f"/{region}_neuron_{unit_id}.png",dpi=600,bbox_inches = 'tight')
    plt.clf()
        
def main(units):
    plt.figure(figsize=(10, 6))
    # single neuron whole time
    for index, row in units.iterrows():
        unit_id = row['cluster_id']
        region_name = row['region']
        ISI_single_neuron_session(region_name,unit_id)
    '''
    # single neuron - trials, each color is a trial
    for index, row in units.iterrows():
        unit_id = row['cluster_id']
        region_name = row['region']

        intervals = ISI_single_neuron_trials(region_name,unit_id,'stoptrials')
        if stat_fit == True:
            stat_fit_plot(intervals,region_name,unit_id,'stoptrials')

        intervals = ISI_single_neuron_trials(region_name,unit_id,'runtrials')
        if stat_fit == True:
            stat_fit_plot(intervals,region_name,unit_id,'runtrials')
    # region - population - trials, each color is a neuron
    region_neurons = units.groupby('region')['cluster_id'].apply(list).to_dict()
    for region_name, popu_id in region_neurons.items():
        intervals = ISI_population_trials(region_name,popu_id,'stoptrials')
        if stat_fit == True:
            stat_fit_plot(intervals,region_name)

        intervals = ISI_population_trials(region_name,popu_id,'runtrials')
        if stat_fit == True:
            stat_fit_plot(intervals,region_name)
    '''

main(neurons)