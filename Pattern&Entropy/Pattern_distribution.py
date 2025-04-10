"""
# coding: utf-8
@author: Yuhao Zhang
last updated: 04/07/2025
data from: Xinchao Chen
"""
import neo
import quantities as pq
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
from math import log
import warnings
import random
from elephant.conversion import BinnedSpikeTrain
from elephant.spike_train_generation import StationaryPoissonProcess  # use this to test wherther the code is correct
np.set_printoptions(threshold=np.inf)

# Lobule
#20230113_littermate    Lobules IV-V
#20230523_Syt2_conditional_tremor_mice1    Lobule III  Lobule II
#20230604_Syt2_conditional_tremor_mice2_medial   Lobule III
#20230602_Syt2_conditional_tremor_mice2_lateral  Lobule III
#20230623_Syt2_conditional_tremor_mice4  Lobule III  Lobule II

region_name = 'Lobules IV-V'
## this parameter is for the elephant package
fr_bin = 1  # unit: ms
avoid_spikemore1 = True # 避免1ms的bin里有多个spike,对于1ms内多个spike的，强行置1

## ------ Easysort ------
# --- NEED CHANGE ---
mice_name = '20230113_littermate'
mapping_file = 'unit_ch_dep_region_QC_isi_violations_ratio_pass_rate_55.30864197530864%.csv'
QC_method = 'QC_ISI_violation'  # Without_QC/QC_ISI_violation/etc
# --- NO NEED CHANGE ---
sorting_path = rf'E:\xinchao\Data\useful_data\NP1\{mice_name}\Sorted\Easysort\results_KS2\sorter_output'
save_path = rf'C:\Users\zyh20\Desktop\Research\01_ET_data_analysis\Research\Firing_Pattern\NP1\Easysort\{QC_method}\{mice_name}'  
neurons = pd.read_csv(rf'E:\xinchao\Data\useful_data\NP1\{mice_name}\Sorted\Easysort\mapping\{mapping_file}')  # different sorting have different nueron id

# ------ Xinchao_sort ------
# --- NEED CHANGE ---
#mice_name = '20230602_Syt2_conditional_tremor_mice2_lateral'
# --- NO NEED CHANGE ---
#sorting_path = rf'E:\xinchao\Data\useful_data\NP1\{mice_name}\Sorted\Xinchao_sort'
#save_path = rf'C:\Users\zyh20\Desktop\Research\01_ET_data_analysis\Research\spectrum_analysis\NP1\Xinchao_sort\{mice_name}'  
#neurons = pd.read_csv(sorting_path + '/neuron_id_region_firingrate.csv')  # different sorting have different nueron id

# --------- Load data ----------
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

# --------- spike train ----------
# get single neuron spike train
def singleneuron_spiketrain(id):
    x = np.where(identities == id)
    y=x[0]
    spike_times=np.zeros(len(y))
    for i in range(0,len(y)):
        z=y[i]
        spike_times[i]=times[z]/sample_rate
    return spike_times

#split spike times into trials, now is the +-0.5s of start pushing the rod
def Trials_spiketrain(spike_times,marker):
    for i in range(len(marker)):
        Trials_spiketrain=np.array([])
        
        for j in range(len(spike_times)):
            if marker[i,0]<spike_times[j] and spike_times[j]<marker[i,1]:
                Trials_spiketrain=np.append(Trials_spiketrain,spike_times[j])
        if Trials_spiketrain.size != 0:
            for k in range(1,len(Trials_spiketrain)):
                Trials_spiketrain[k]=Trials_spiketrain[k]-Trials_spiketrain[0]
            Trials_spiketrain[0]=0
        y=np.full((len(Trials_spiketrain),1),i)      
        plt.plot(Trials_spiketrain,y, '|', color='gray') 

    plt.title('neuron') 
    plt.xlabel("time") 
    plt.xlim(0,1)
    plt.ylim(-2,len(marker)+5)
    plt.show()

# spike counts
def build_time_window_domain(bin_edges, offsets, callback=None):
    callback = (lambda x: x) if callback is None else callback
    domain = np.tile(bin_edges[None, :], (len(offsets), 1))
    domain += offsets[:, None]
    return callback(domain)

def build_spike_histogram(time_domain,
                          spike_times,
                          dtype=None,
                          binarize=False):

    time_domain = np.array(time_domain)

    tiled_data = np.zeros(
        (time_domain.shape[0], time_domain.shape[1] - 1),
        dtype=(np.uint8 if binarize else np.uint16) if dtype is None else dtype
    )

    starts = time_domain[:, :-1]
    ends = time_domain[:, 1:]

    data = np.array(spike_times)

    start_positions = np.searchsorted(data, starts.flat)
    end_positions = np.searchsorted(data, ends.flat, side="right")
    counts = (end_positions - start_positions)

    tiled_data[:, :].flat = counts > 0 if binarize else counts

    return tiled_data

def spike_counts(
    spike_times,
    bin_edges,
    movement_start_time,
    binarize=False,
    dtype=None,
    large_bin_size_threshold=0.001,
    time_domain_callback=None
):

    #build time domain
    bin_edges = np.array(bin_edges)
    domain = build_time_window_domain(
        bin_edges,
        movement_start_time,
        callback=time_domain_callback)

    out_of_order = np.where(np.diff(domain, axis=1) < 0)
    if len(out_of_order[0]) > 0:
        out_of_order_time_bins = \
            [(row, col) for row, col in zip(out_of_order)]
        raise ValueError("The time domain specified contains out-of-order "
                            f"bin edges at indices: {out_of_order_time_bins}")

    ends = domain[:, -1]
    starts = domain[:, 0]
    time_diffs = starts[1:] - ends[:-1]
    overlapping = np.where(time_diffs < 0)[0]

    if len(overlapping) > 0:
        # Ignoring intervals that overlaps multiple time bins because
        # trying to figure that out would take O(n)
        overlapping = [(s, s + 1) for s in overlapping]
        warnings.warn("You've specified some overlapping time intervals "
                        f"between neighboring rows: {overlapping}, "
                        "with a maximum overlap of"
                        f" {np.abs(np.min(time_diffs))} seconds.")
        
    #build_spike_histogram
    tiled_data = build_spike_histogram(
        domain,
        spike_times,
        dtype=dtype,
        binarize=binarize
    )
    return tiled_data

def binary_spiketrain(id,marker):  #each trial
    # bin
    bin_width = 0.0007
    duration = 5  #一个trial的时间，或你关注的时间段的长度
    pre_time = -0.1
    post_time = duration
    bins = np.arange(pre_time, post_time+bin_width, bin_width)   

    histograms=spike_counts(
        singleneuron_spiketrain(id),
        bin_edges=bins,
        movement_start_time=marker,
        )
    print(histograms)

    return histograms

def firingrate_time(id,marker,duration,bin_width):
    # bin
    pre_time = 0
    post_time = duration
    bins = np.arange(pre_time, post_time+bin_width,bin_width)  # bin_width默认 0.14
    # histograms
    histograms=spike_counts(
        singleneuron_spiketrain(id),
        bin_edges=bins,
        movement_start_time=marker,
        )
    return histograms

def population_spikecounts(neuron_id,marker_start,marker_end,trial_dura,bin): 
    #trial_dura is trivial duration, finally it will be merged
    #bin is the bin size of spike times, which is very important, now is 0.005s
    #这里由于allen的spike counts函数是针对视觉的，因此对trial做了划分，必要trialmarker作为参数，因此这里分假trial，再合并
    #Artificial_time_division是把整个session人为划分为一个个时间段trial
    #bin是对firing rate的滑窗大小，单位s
    one_neruon = np.array([])
    marker=np.array(range(int(marker_start),int(marker_end)-int(marker_end)%trial_dura,trial_dura))
    #get a 2D matrix with neurons, trials(trials contain times), trials and times are in the same dimension
    for j in range(len(neuron_id)): #第j个neuron
        #每个neuron的trials水平append
        for i in range(len(marker)):
            if i == 0:
                one_neruon = firingrate_time(neuron_id[j],marker,trial_dura,bin)[0]
            else:
                trail = firingrate_time(neuron_id[j],marker,trial_dura,bin)[i]
                one_neruon = np.append(one_neruon, trail)
        if j == 0:
            popu_spike = one_neruon
        else:
            popu_spike = np.vstack((popu_spike, one_neruon))
    '''
    print(popu_spike)
    print(popu_spike.shape)
    '''
    time_len=(int(marker_end)-int(marker_end)%trial_dura)/bin
    return popu_spike,time_len

def popu_sptrain_trial(neuron_ids,marker_start,marker_end):
    for j in range(len(neuron_ids)): #第j个neuron
        spike_times = singleneuron_spiketrain(neuron_ids[j])
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

# --------- pattern ----------
def pattern(data,neuronid,state,index):
    for j in range(len(data)):
        id = neuronid[j]
        neuron_sptrain = data[j]
        result_dic = {}
        
        # 步骤1：原始数据统计（包含全零）
        for i in range(0, len(neuron_sptrain) - len(neuron_sptrain) % 8, 8):
            segment = neuron_sptrain[i:i+8]
            pattern = ''.join(str(bit) for bit in segment)
            result_dic[pattern] = result_dic.get(pattern, 0) + 1

        # 步骤2：计算概率（包含全零样本）
        total_with_zero = max(sum(result_dic.values()), 1)  # 包含全零的总样本数
        prob_dict_with_zero = {k: v/total_with_zero for k, v in result_dic.items()}

        # 步骤3：过滤全零模式（仅影响绘图数据）
        filtered_prob = {k: v for k, v in prob_dict_with_zero.items() if k != '00000000'}

        # 步骤4：定义复合排序键（与之前相同）
        def sort_key(item):
            pattern = item[0]
            ones_pos = tuple(i for i, bit in enumerate(pattern) if bit == '1')
            return (len(ones_pos), ones_pos)
        
        # 执行排序（仅对非零模式）
        sorted_patterns = sorted(filtered_prob.items(), key=sort_key)
        
        # 准备绘图数据
        x = [item[0] for item in sorted_patterns]
        y = [item[1] for item in sorted_patterns]
        
        # 可视化设置
        plt.bar(x, y, color='steelblue', edgecolor='white')
        plt.xticks(x, rotation=90, fontsize=7, ha='center', fontfamily='monospace')
        plt.yticks(fontsize=12)
        plt.ylabel("Probability (Excluding Zero Pattern)", fontsize=14)
        plt.title(f'{region_name}_neuron{id} Pattern Distribution (包含全零计算概率)', fontsize=16)
        
        # 添加统计信息注释
        zero_prob = prob_dict_with_zero.get('00000000', 0)
        plt.text(0.95, 0.95, 
                f'Zero Pattern Prob: {zero_prob:.4f}\nTotal Patterns: {total_with_zero}',
                transform=plt.gca().transAxes,
                ha='right', va='top',
                fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8))
        plt.savefig(f"{save_path}/{region_name}_neuron{id}_trial{index}_{state}.png",dpi=600,bbox_inches='tight',pad_inches=0.1)
        plt.clf()

def dict2csv(dic, filename):
    file = open(filename, 'w', encoding='utf-8', newline='')
    csv_writer = csv.DictWriter(file, fieldnames=list(dic.keys()))
    csv_writer.writeheader()
    for i in range(len(dic[list(dic.keys())[0]])):   # 将字典逐行写入csv
        dic1 = {key: dic[key][i] for key in dic.keys()}
        csv_writer.writerow(dic1)
    file.close()

def main():
    plt.figure(figsize=(24, 10))  # 适当增加宽度保证标签显示
    result = neurons.groupby('region')['cluster_id'].apply(list).reset_index(name='cluster_ids')
    print(result)
    for index, row in result.iterrows(): # select region neurons
        region = row['region']
        popu_ids = row['cluster_ids']
        if region == region_name:
            for index, row in treadmill_origin.iterrows():  # enumerate trials
                status = row['run_or_stop']
                if status == 1:
                    run_time = row['time_interval_left_end']
                    # before running 0.5s
                    data = popu_sptrain_trial(popu_ids,run_time,run_time+0.5)
                    pattern(data,popu_ids,'-0.5s',index)
                    # after running 0.5s
                    data = popu_sptrain_trial(popu_ids,run_time-0.5,run_time)
                    pattern(data,popu_ids,'0.5s',index)

main()

'''
# information entropy
h=0
for i in p:
    h = h - p[i]*log(p[i],2)
print('Shannon Entropy=%f'%h)

#save to csv
my_list = [[key, value] for key, value in p_del0.items()]
with open('C:/Users/zyh20/Desktop/csv/0to5ET.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(my_list)
'''
'''
spiketrain = StationaryPoissonProcess(rate=50*pq.Hz,t_stop=10000*pq.ms,t_start=0*pq.ms,refractory_period=3*pq.ms).generate_spiketrain()

for i in range(0,len(neuron_sptrain)-len(neuron_sptrain)%8,8):  # delete end bits that can't be divide by 8
    a = np.array(neuron_sptrain[i:i+8])   # slice into 8 bit,  1 byte(1字节)(1B) = 8 bit(8比特)(8位二进制)；1KB = 1024B; 1MB = 1024KB; 1GB = 1024MB
    str1 = ''.join(str(z) for z in a)     # array to str
    # 如果 '1' 的数量大于 1，进行拆分
    if str1.count('1') > 1:
        result = []
        for i, bit in enumerate(str1):
            if bit == '1':
                # 生成仅含有一个 '1' 的序列
                new_seq = ['0'] * len(str1)
                new_seq[i] = '1'
                result.append(''.join(new_seq))
        for split in result:   
            if split not in result_dic:   # use dic to statistic, key = str, value = number of times
                result_dic[split]=1
            else:
                result_dic[split]+=1
    else:
        if str1 not in result_dic:     # use dic to statistic, key = str, value = number of times
            result_dic[str1]=1
        else:
            result_dic[str1]+=1
    

#compute probability
total=sum(result_dic.values())
p={k: v / total for k, v in result_dic.items()}
del result_dic['00000000']
total_del0=sum(result_dic.values())
p_del0={k: v / total_del0 for k, v in result_dic.items()}
#plot
x=list(p_del0.keys())
y=list(p_del0.values())



for i in range(0,len(neuron_sptrain)-len(neuron_sptrain)%8,8):  # delete end bits that can't be divide by 8
    # slice into 8 bit,  1 byte(1字节)(1B) = 8 bit(8比特)(8位二进制)；1KB = 1024B; 1MB = 1024KB; 1GB = 1024MB
    a = np.array(neuron_sptrain[i:i+8])                
    str1 = ''.join(str(z) for z in a)         # array to str
    if str1 not in result_dic:                # use dic to statistic, key = str, value = number of times
        result_dic[str1]=1
    else:
        result_dic[str1]+=1

#filtered_dict = {key: value for key, value in result_dic.items() if key.count('1') == 1 and key.count('0') == 7}
total=sum(result_dic.values())
filtered_dict = result_dic     # if you want to delete patttens that are all 0, please uncomment this line
del filtered_dict['00000000']  # if you want to delete patttens that are all 0, please uncomment this line
p={k: v / total for k, v in filtered_dict.items()}  #compute probability
# 提取并排序仅包含一个1的键值对
filtered_sorted_items = sorted(
    {key: value for key, value in p.items() if key.count('1') == 1 and key.count('0') == len(key) - 1}.items()
)
# 提取其他不满足条件的键值对
remaining_items = {key: value for key, value in p.items() if key.count('1') != 1 or key.count('0') != len(key) - 1}
# 合并两个字典，将排序后的键值对放在前面
sorted_dict = dict(filtered_sorted_items)
sorted_dict.update(remaining_items)
#plot
x=list(sorted_dict.keys())
y=list(sorted_dict.values())
plt.bar(x, y)
#plt.title(f'{mice}_{region_name}_{id}_Encoding pattern distribution', fontsize=16)
plt.xticks(x, rotation=90, fontsize=10)
plt.yticks(fontsize=16)
plt.ylabel("Probability of pattern", fontsize=16)
plt.savefig(fig_save_path+f"/{mice}_{region_name}_{id}_Encoding pattern distribution.png",dpi=600,bbox_inches = 'tight')
plt.clf()
'''