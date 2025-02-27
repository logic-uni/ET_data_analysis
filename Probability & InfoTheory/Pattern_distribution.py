"""
# coding: utf-8
@author: Yuhao Zhang
last updated: 08/14/2024
data from: Xinchao Chen
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from math import log
import warnings
import random
import quantities as pq  # noqa
from elephant.spike_train_generation import StationaryPoissonProcess
np.set_printoptions(threshold=np.inf)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
fig_save_path = r'C:\Users\zyh20\Desktop\ET_data analysis\firing pattern'
mice = '20230602-condictional tremor2-wai'
### marker
treadmill_marker_path = r'E:\chaoge\sorted neuropixels data\20230602-condictional tremor2-wai\20230523_Syt2_449_3_Day60_g0'
treadmill = pd.read_csv(treadmill_marker_path+'/treadmill_move_stop_velocity.csv',index_col=0)
print(treadmill)

### electrophysiology
sample_rate=30000 #spikeGLX neuropixel sample rate
file_directory=r'E:\chaoge\sorted neuropixels data\20230602-condictional tremor2-wai\20230523_Syt2_449_3_Day60_g0\20230523_Syt2_449_3_Day60_g0_imec0'
identities = np.load(file_directory+'/spike_clusters.npy') #存储neuron的编号id,对应phy中的第一列id
times = np.load(file_directory+'/spike_times.npy')  #
channel = np.load(file_directory+'/channel_positions.npy')
neurons = pd.read_csv(file_directory+'/region_neuron_id.csv', low_memory = False,index_col=0)#防止弹出警告
print(neurons)
print("检查treadmill总时长和电生理总时长是否一致")
print("电生理总时长")
print((times[-1]/sample_rate)[0])
print("跑步机总时长") 
print(treadmill['time_interval_right_end'].iloc[-1])
neuron_num = neurons.count().transpose().values

#### spike train & firing rates

# get single neuron spike train
def singleneuron_spiketrain(id):
    x = np.where(identities == id)
    y=x[0]
    spike_times=np.zeros(len(y))
    print()
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

def population_spikecounts(neuron_id,marker_start,marker_end,Artificial_time_division,bin):  
    #这里由于allen的spike counts函数是针对视觉的，因此对trial做了划分，必要trialmarker作为参数，因此这里分假trial，再合并
    #Artificial_time_division是把整个session人为划分为一个个时间段trial
    #bin是对firing rate的滑窗大小，单位s
    one_neruon = np.array([])
    marker=np.array(range(int(marker_start),int(marker_end)-int(marker_end)%Artificial_time_division,Artificial_time_division))
    #get a 2D matrix with neurons, trials(trials contain times), trials and times are in the same dimension
    for j in range(len(neuron_id)): #第j个neuron
        #每个neuron的trials水平append
        for i in range(len(marker)):
            if i == 0:
                one_neruon = firingrate_time(neuron_id[j],marker,Artificial_time_division,bin)[0]
            else:
                trail = firingrate_time(neuron_id[j],marker,Artificial_time_division,bin)[i]
                one_neruon = np.append(one_neruon, trail)
        if j == 0:
            popu_spike = one_neruon
        else:
            popu_spike = np.vstack((popu_spike, one_neruon))
    '''
    print(popu_spike)
    print(popu_spike.shape)
    '''
    time_len=(int(marker_end)-int(marker_end)%Artificial_time_division)/bin
    return popu_spike,time_len

def pattern_entropy_single(neuron_sptrain):
    # about bin 1 bit = 1 msec 
    # Statistics pattern all neurons
    result_dic={}
    for i in range(0,len(neuron_sptrain)-len(neuron_sptrain)%8,8):  # delete end bits that can't be divide by 8
        a = np.array(neuron_sptrain[i:i+8])                # slice into 8 bit,  1 byte(1字节)(1B) = 8 bit(8比特)(8位二进制)；1KB = 1024B; 1MB = 1024KB; 1GB = 1024MB
        str1 = ''.join(str(z) for z in a)         # array to str
        if str1 not in result_dic:                # use dic to statistic, key = str, value = number of times
            result_dic[str1]=1
        else:
            result_dic[str1]+=1

    for key in result_dic.keys():
        if key.count('1') != 1:
            del result_dic[key]

    #compute probability
    total=sum(result_dic.values())
    p={k: v / total for k, v in result_dic.items()}
    #plot
    x=list(p.keys())
    y=list(p.values())

    plt.bar(x, y)
    plt.title(f'sudo_Encoding pattern distribution', fontsize=16)
    plt.xticks(x, rotation=90, fontsize=10)
    plt.yticks(fontsize=16)
    plt.ylabel("Probability of pattern", fontsize=16)
    plt.savefig(fig_save_path+f"/sudo_Encoding pattern distribution.png",dpi=600,bbox_inches = 'tight')
    plt.clf()

def pattern_entropy_contain0(data,neuronid,region_name):
    # about bin 1 bit = 1 msec 
    # Statistics pattern all neurons
    plt.figure(figsize=(20, 10))  # 宽度为15，高度为5
    for j in range(0,len(data)):
        id = neuronid[j]
        neuron_sptrain=data[j]  # get a neuron
        result_dic={}
        for i in range(0,len(neuron_sptrain)-len(neuron_sptrain)%8,8):  # delete end bits that can't be divide by 8
            a = np.array(neuron_sptrain[i:i+8])                # slice into 8 bit,  1 byte(1字节)(1B) = 8 bit(8比特)(8位二进制)；1KB = 1024B; 1MB = 1024KB; 1GB = 1024MB
            str1 = ''.join(str(z) for z in a)         # array to str
            if str1 not in result_dic:                # use dic to statistic, key = str, value = number of times
                result_dic[str1]=1
            else:
                result_dic[str1]+=1

        total=sum(result_dic.values())
        p={k: v / total for k, v in result_dic.items()}  #compute probability

        #plot
        x=list(p.keys())
        y=list(p.values())

        plt.bar(x, y)
        #plt.title(f'{mice}_{region_name}_{id}_Encoding pattern distribution', fontsize=16)
        plt.xticks(x, rotation=90, fontsize=10)
        plt.yticks(fontsize=16)
        plt.ylabel("Probability of pattern", fontsize=16)
        plt.savefig(fig_save_path+f"/{mice}_{region_name}_{id}_Encoding pattern distribution.png",dpi=600,bbox_inches = 'tight')
        plt.clf()

def pattern_entropy(data,neuronid,region_name):
    # about bin 1 bit = 1 msec 
    # Statistics pattern all neurons
    plt.figure(figsize=(20, 10))  # 宽度为15，高度为5
    for j in range(0,len(data)):
        id = neuronid[j]
        neuron_sptrain=data[j]  # get a neuron
        result_dic={}
        for i in range(0,len(neuron_sptrain)-len(neuron_sptrain)%8,8):  # delete end bits that can't be divide by 8
            a = np.array(neuron_sptrain[i:i+8])                # slice into 8 bit,  1 byte(1字节)(1B) = 8 bit(8比特)(8位二进制)；1KB = 1024B; 1MB = 1024KB; 1GB = 1024MB
            str1 = ''.join(str(z) for z in a)         # array to str
            if str1 not in result_dic:                # use dic to statistic, key = str, value = number of times
                result_dic[str1]=1
            else:
                result_dic[str1]+=1

        #filtered_dict = {key: value for key, value in result_dic.items() if key.count('1') == 1 and key.count('0') == 7}
        total=sum(result_dic.values())
        filtered_dict = result_dic
        del filtered_dict['00000000']
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

def dict2csv(dic, filename):
    file = open(filename, 'w', encoding='utf-8', newline='')
    csv_writer = csv.DictWriter(file, fieldnames=list(dic.keys()))
    csv_writer.writeheader()
    for i in range(len(dic[list(dic.keys())[0]])):   # 将字典逐行写入csv
        dic1 = {key: dic[key][i] for key in dic.keys()}
        csv_writer.writerow(dic1)
    file.close()

def main_function(population,marker):
    '''
    spiketrain = StationaryPoissonProcess(rate=50*pq.Hz,t_stop=10000*pq.ms,t_start=0*pq.ms,refractory_period=3*pq.ms).generate_spiketrain()
    print(spiketrain)
    pattern_entropy_single(spiketrain)
    '''
    for i in range(population.shape[1]):  #遍历所有的脑区
        region_name = population.columns.values[i]
        if region_name == 'Superior vestibular nucleus':
            neuron_id = np.array(population.iloc[:, i].dropna()).astype(int)  #提取其中一个脑区的neuron id
            marker_start = marker['time_interval_left_end'].iloc[12]
            marker_end = marker['time_interval_right_end'].iloc[12]
            print(marker_start)
            print(marker_end)
            data,time_len = population_spikecounts(neuron_id,marker_start,marker_start+40,5,0.005)
            #print(data[-1].shape)

            pattern_entropy(data,neuron_id,region_name)
    

main_function(neurons,treadmill)

'''
for i in range(0,len(neuron_sptrain)-len(neuron_sptrain)%8,8):  # delete end bits that can't be divide by 8
    a = np.array(neuron_sptrain[i:i+8])                # slice into 8 bit,  1 byte(1字节)(1B) = 8 bit(8比特)(8位二进制)；1KB = 1024B; 1MB = 1024KB; 1GB = 1024MB
    str1 = ''.join(str(z) for z in a)         # array to str
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
            if split not in result_dic:                # use dic to statistic, key = str, value = number of times
                result_dic[split]=1
            else:
                result_dic[split]+=1
    else:
        if str1 not in result_dic:                # use dic to statistic, key = str, value = number of times
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

'''