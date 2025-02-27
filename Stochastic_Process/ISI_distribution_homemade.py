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
### Basic information
mice_name = '20230602_Syt2_conditional_tremor_mice2_lateral'
region_name = 'Lobule III'
QC_method = 'WithoutQC'  # WithoutQC/ISI_violation/Amplitude_cutoff
Sorting_method = 'Easysort'
trial_type = 'stop'      #'stop' 'run'  run trials or stop trials
plot_mode = 'overlap'    #'each_n_t' 'overlap'
mice_type = 'cond_ET'    #'PV_Syt2'  'littermate' 'cond_ET'

# ------- NO NEED CHANGE -------
### path
sorting_path = rf'E:\xinchao\Data\useful_data\{mice_name}\Sorted\Easysort\results_KS2\sorter_output'
save_path = rf'C:\Users\zyh20\Desktop\Research\01_ET_data_analysis\Research\ISI_distribution\{Sorting_method}\{mice_name}\{region_name}\{QC_method}'  
treadmill = pd.read_csv(rf'E:\xinchao\Data\useful_data\{mice_name}\Marker\treadmill_move_stop_velocity.csv',index_col=0)
print(treadmill)

### electrophysiology
sample_rate=30000 #spikeGLX neuropixel sample rate
identities = np.load(sorting_path+'/spike_clusters.npy') #存储neuron的编号id,对应phy中的第一列id
times = np.load(sorting_path+'/spike_times.npy')  #
channel = np.load(sorting_path+'/channel_positions.npy')
neurons = pd.read_csv(sorting_path+'/region_neuron_id.csv', low_memory = False,index_col=0)#防止弹出警告
print(neurons)
print("检查treadmill总时长和电生理总时长是否一致")
print("电生理总时长")
print((times[-1]/sample_rate)[0])
print("跑步机总时长") 
print(treadmill['time_interval_right_end'].iloc[-1])
neuron_num = neurons.count().transpose().values

### ISI parameters
fr_filter = 1 # 滤掉fr小于1spike/s
fr_fil = 30/fr_filter
cutoff_distr = 0.25 # cutoff_distr=0.25代表截断ISI分布大于0.25s的, cutoff_distr=None代表不截断
histo_bin_num = 100

# get single neuron spike train
def singleneuron_spiketrain(id):
    x = np.where(identities == id)
    y=x[0]
    spike_times=np.empty(len(y))
    for i in range(0,len(y)):
        z=y[i]
        spike_times[i]=times[z]/sample_rate
    return spike_times

def ET_runtrials_ISI(spike_times,neuron_intervals,color,id):
    marker = treadmill
    for trial in np.arange(1,len(marker['time_interval_left_end']),2):
        marker_start = marker['time_interval_left_end'].iloc[trial]
        if marker['time_interval_right_end'].iloc[trial] - marker_start > 30:
            marker_end = marker_start+30
            spike_times_trail = spike_times[(spike_times > marker_start) & (spike_times < marker_end)]
            if len(spike_times_trail) > fr_fil: 
                intervals = np.diff(spike_times_trail)  # 计算时间间隔
                # 绘制时间间隔的直方图
                if cutoff_distr != None:
                    intervals = intervals[intervals <= cutoff_distr]
                    intervals = intervals[intervals > 9.99999999e-04]
                    if len(intervals) != 0:  #截断后可能导致没有interval
                        plt.hist(intervals, bins=histo_bin_num, density=False,alpha=0.6,color=color)
                        if plot_mode == 'each_n_t':
                            plt.savefig(save_path+f"/each_neuron_trials/Neuron{id}_Trial{trial}.png",dpi=600,bbox_inches = 'tight')
                            plt.clf()
                else:
                    plt.hist(intervals, bins=20, density=True, alpha=0.6,color=color)
                neuron_intervals.extend(intervals) 
    return neuron_intervals

def ET_stoptrials_ISI(spike_times,neuron_intervals,color,id):
    marker = treadmill
    for trial in np.arange(0,len(marker['time_interval_left_end']),2):
        marker_start = marker['time_interval_left_end'].iloc[trial]
        if marker['time_interval_right_end'].iloc[trial] - marker_start > 30:
            marker_end = marker_start+30
            spike_times_trail = spike_times[(spike_times > marker_start) & (spike_times < marker_end)]
            if len(spike_times_trail) > fr_fil: 
                intervals = np.diff(spike_times_trail)  # 计算时间间隔
                # 绘制时间间隔的直方图
                if cutoff_distr != None:
                    intervals = intervals[intervals <= cutoff_distr]
                    intervals = intervals[intervals > 9.99999999e-04]
                    if len(intervals) != 0:  #截断后可能导致没有interval
                        plt.hist(intervals, bins=histo_bin_num, density=False, alpha=0.6,color=color)
                        if plot_mode == 'each_n_t':
                            plt.savefig(save_path+f"/each_neuron_trials/Neuron{id}_Trial{trial}.png",dpi=600,bbox_inches = 'tight')
                            plt.clf()
                else:
                    plt.hist(intervals, bins=20, density=True, alpha=0.6,color=color)
                neuron_intervals.extend(intervals) 
    return neuron_intervals

def littermate_ISI_subfunction(marker_start,spike_times,color,id,trial_num):
    marker_end = marker_start + 30
    spike_times_trail = spike_times[(spike_times > marker_start) & (spike_times < marker_end)]
    intervals = np.array([])
    if len(spike_times_trail) > fr_fil:
        intervals = np.diff(spike_times_trail)  # 计算时间间隔
        # 绘制时间间隔的直方图
        if cutoff_distr != None:
            intervals = intervals[intervals <= cutoff_distr]
            intervals = intervals[intervals > 9.99999999e-04]
            if len(intervals) != 0:  #截断后可能导致没有interval
                plt.hist(intervals, bins=histo_bin_num, density=False,alpha=0.6,color=color)  #如果只看0.25s间隔以内细致的，画counts更适合
                if plot_mode == 'each_n_t':
                    plt.savefig(save_path+f"/each_neuron_trials/Neuron{id}_Trial{trial_num}.png",dpi=600,bbox_inches = 'tight')
                    plt.clf()
        else:
            plt.hist(intervals, bins=20, density=True,alpha=0.6,color=color)
    return intervals

def littermate_runtrials_ISI(spike_times,neuron_intervals,color,id):  ## LITTERMATE 人为切分30s的trial
    trial_num = 0
    for marker_start in np.arange(105,495,30):
        trial_num = trial_num+1
        intervals_trial = littermate_ISI_subfunction(marker_start,spike_times,color,id,trial_num)
        neuron_intervals.extend(intervals_trial)
    for marker_start in np.arange(705,1095,30):
        trial_num = trial_num+1
        intervals_trial = littermate_ISI_subfunction(marker_start,spike_times,color,id,trial_num)
        neuron_intervals.extend(intervals_trial)
    return neuron_intervals

def littermate_stoptrials_ISI(spike_times,neuron_intervals,color,id):  ## LITTERMATE 人为切分30s的trial
    trial_num = 0
    for marker_start in np.arange(0,90,30):
        trial_num = trial_num+1
        intervals_trial = littermate_ISI_subfunction(marker_start,spike_times,color,id,trial_num)
        neuron_intervals.extend(intervals_trial)
    for marker_start in np.arange(515,695,30):
        trial_num = trial_num+1
        intervals_trial = littermate_ISI_subfunction(marker_start,spike_times,color,id,trial_num)
        neuron_intervals.extend(intervals_trial)
    for marker_start in np.arange(1115,1295,30):
        trial_num = trial_num+1
        intervals_trial = littermate_ISI_subfunction(marker_start,spike_times,color,id,trial_num)
        neuron_intervals.extend(intervals_trial)
    return neuron_intervals

def ISI(region_neuron):
    lambdas = []
    plt.figure(figsize=(10, 6))
    neuron_colors = plt.cm.hsv(np.linspace(0, 1, len(region_neuron)))
    for i in range(len(region_neuron)):
        spike_times = singleneuron_spiketrain(region_neuron[i])
        color = neuron_colors[i]  #一个neuron使用同一个color
        neuron_intervals = []  # 用于存储该neuron的所有时间间隔
        neuron_id = region_neuron[i]
        # 每个neuron多个trial分别画histogram
        if trial_type == 'run':
            if mice_type == 'cond_ET':
                neuron_intervals = ET_runtrials_ISI(spike_times,neuron_intervals,color,neuron_id)  ##需要加颜色修改，这样对于单个neuron不同trial颜色都不同
            elif mice_type == 'littermate':
                neuron_intervals = littermate_runtrials_ISI(spike_times,neuron_intervals,color,neuron_id)
            elif mice_type == 'PV_Syt2':
                neuron_intervals = PV_Syt2(spike_times,neuron_intervals)
        elif trial_type == 'stop':
            if mice_type == 'cond_ET':
                neuron_intervals = ET_stoptrials_ISI(spike_times,neuron_intervals,color,neuron_id)
            elif mice_type == 'littermate':
                neuron_intervals = littermate_stoptrials_ISI(spike_times,neuron_intervals,color,neuron_id)
            elif mice_type == 'PV_Syt2':
                neuron_intervals = PV_Syt2(spike_times,neuron_intervals)

        ## 画完当前neuron所有trial的hist之后，对当前neuron整体的ISI拟合指数分布
        if len(neuron_intervals) != 0 and cutoff_distr == None:
            params = expon.fit(neuron_intervals, floc=0)  # 固定起始点为0
            lambda_fit = 1 / params[1]  # 拟合的lambda值
            # 绘制拟合的指数分布曲线
            x = np.linspace(0, max(neuron_intervals), 100)
            plt.plot(x, expon.pdf(x, *params), 'b-', lw=0.2)
            lambdas.append(lambda_fit)  # 保存lambda值 each lambda represent a neurons firing strength in different run trials
        print(neuron_intervals)

    if cutoff_distr == None:
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
    elif plot_mode != 'each_n_t':
        plt.xlim(-0.01,0.26) #add this when needed
        plt.xlabel('Inter-spike Interval (s)')
        plt.ylabel('Counts')
        plt.title(f'ISI Distribution_{region_name}')
        plt.text(0.95, 0.95, f'{mice_name}\n\nfiring filter > {fr_filter} spike/s\n\nhisto bin: {histo_bin_num}\n\ndistr cutoff > {cutoff_distr}\n\n{trial_type} trials', ha='right', va='top', transform=plt.gca().transAxes)
        plt.savefig(save_path+f"/{trial_type}_trials_{region_name}_cutoff_{cutoff_distr}.png",dpi=600,bbox_inches = 'tight')
    
def PV_Syt2(neuron_a_id,neuron_b_id,marker):
    neuron_intervals = 0
    return neuron_intervals
        
def main():
    ## 取脑区全部的neuron id 
    for i in range(neurons.shape[1]):  
        name = neurons.columns.values[i]
        if name == region_name:
            region_neurons_id = np.array(neurons.iloc[:, i].dropna()).astype(int)
            # 计算ISI分布
            ISI(region_neurons_id)

main()