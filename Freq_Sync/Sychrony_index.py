"""
# coding: utf-8
@author: Yuhao Zhang
last updated: 10/04/2024
data from: Xinchao Chen
"""
import neo
import quantities as pq
import matplotlib.pyplot as plt
from elephant.conversion import BinnedSpikeTrain
from viziphant.spike_train_correlation import plot_cross_correlation_histogram
from elephant.spike_train_correlation import cross_correlation_histogram  # noqa
from elephant.spike_train_synchrony import spike_contrast
from elephant import statistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import os
import glob
import pandas as pd
from scipy.signal import correlate, correlation_lags
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)
np.seterr(divide='ignore',invalid='ignore')

method='1'
mice = '20230523-condictional tremor1'
syn_save_path = r'C:\Users\zyh20\Desktop\ET_data analysis\Synchrony\scipy_package_ccg_synchrony'
### marker
treadmill_marker_path = r'E:\chaoge\sorted neuropixels data\20230523-condictional tremor1\20230523\raw\20230523_Syt2_449_1_Day50_g0'
treadmill = pd.read_csv(treadmill_marker_path+'/treadmill_move_stop_velocity.csv',index_col=0)
print(treadmill)

### electrophysiology
sample_rate=30000 #spikeGLX neuropixel sample rate
file_directory=r'E:\chaoge\sorted neuropixels data\20230523-condictional tremor1\working\sorted'
identities = np.load(file_directory+'/spike_clusters.npy') #存储neuron的编号id,对应phy中的第一列id
times = np.load(file_directory+'/spike_times.npy')  #
channel = np.load(file_directory+'/channel_positions.npy')
neurons = pd.read_csv(file_directory+'/region_neuron_id.csv', low_memory = False,index_col=0)#防止弹出警告
#neurons = neurons.drop(['Cerebellum', 'arbor vitae'], axis=1)
print(neurons)
print("检查treadmill总时长和电生理总时长是否一致")
print("电生理总时长")
print((times[-1]/sample_rate)[0])
max_time=np.ceil((times[-1]/sample_rate)[0])
print("跑步机总时长") 
print(treadmill['time_interval_right_end'].iloc[-1])

def load_syn_index_csv():
    # 定义存储所有文件数据的列表
    all_files = []

    # 获取文件夹中所有CSV文件的文件名列表
    file_list = glob.glob(os.path.join(syn_save_path, '*.csv'))

    # 循环读取每个CSV文件，并将其追加到all_files列表中
    for file in file_list:
        df = pd.read_csv(file)
        file_name = os.path.splitext(os.path.basename(file))[0]# 获取文件名，并去掉路径和后缀
        df['mice Name'] = file_name      # 添加文件名作为第三列
        all_files.append(df)# 将处理后的数据框追加到列表

    # 将所有文件合并为一个数据框
    merged_df = pd.concat(all_files, ignore_index=True)

    # 打印合并后的数据框
    print(merged_df)
    return merged_df

# get single neuron spike train
def singleneuron_spiketrain(id):
    x = np.where(identities == id)
    y=x[0]
    spike_times=np.empty(len(y))
    for i in range(0,len(y)):
        z=y[i]
        spike_times[i]=times[z]/sample_rate
    return spike_times

def synchrony_method_01(spiketrain1,spiketrain2,i):
    #plt.figure()
    binned_spiketrain1 = BinnedSpikeTrain(spiketrain1, bin_size=0.5*pq.ms,tolerance=None)
    binned_spiketrain2 = BinnedSpikeTrain(spiketrain2, bin_size=0.5*pq.ms,tolerance=None)
    fr_spiketrain1 = statistics.mean_firing_rate(spiketrain1)
    fr_spiketrain2 = statistics.mean_firing_rate(spiketrain2)
    norm_factor = np.sqrt(fr_spiketrain1*fr_spiketrain2)
    # scipy包计算cross correlation
    array1 = binned_spiketrain1.to_array().astype(int)[0]
    array2 = binned_spiketrain2.to_array().astype(int)[0]
    # 计算相关性
    corr = correlate(array1, array2, mode='full')
    corr = corr/norm_factor
    # 计算相关性的时间滞后范围
    lags = correlation_lags(len(array1), len(array2), mode='full')
    timelagzero = int((len(corr)-1)/2)  #是timelag为0时的indice
    '''
    # 画图
    plt.figure(figsize=(10, 6))
    plt.plot(lags, corr)
    plt.title('Cross Correlation between two arrays')
    plt.xlabel('Time Lag')
    plt.ylabel('Correlation')
    plt.grid(True)
    plt.xlim(-0.1, 0.1)  # 限制时间滞后范围在-10到10
    plt.show()
    
    #使用elephant包计算cross correlation
    cch, lags = cross_correlation_histogram(binned_spiketrain1,binned_spiketrain2,window=[-10, 10])
    cch_v = np.array([])
    for signal in cch:
        cch_value = signal.magnitude
        cch_norm = cch_value / norm_factor# normalize cch with mean firing rate each neuron
        cch_v = np.append(cch_v,cch_norm)
    #print(cch_norm_nump)

    #plt.plot(lags,cch_norm)
    plot_cross_correlation_histogram(cch)
    plt.show()
    #plt.savefig(syn_save_path+f"/fr_norm_{i}_cros_correlogram.png",dpi=600,bbox_inches = 'tight')
    #plt.clf
    '''
    #synchrony_index = (corr[9] + corr[10] + corr[11]) / 3  #-10 ms to 10 ms内的cross-correlogram值平均
    synchrony_index = float((corr[timelagzero] + corr[timelagzero-1] + corr[timelagzero+1]) / 3)    
    return synchrony_index

def synchrony(neurons,marker):
    total_syn = 0
    k=0
    paircount=0
    syn_reg_name = np.array([])
    syn_reg_index = np.array([])
    for i in range(neurons.shape[1]):  #遍历所有的脑区
        region_name = neurons.columns.values[i]
        neuron_id = np.array(neurons.iloc[:, i].dropna()).astype(int)  #提取其中一个脑区的neuron id
        combinations = list(itertools.combinations(neuron_id, 2))
        for comb in combinations:
            neuron_a_id = comb[0]
            neuron_b_id = comb[1]
            spike_times1 = singleneuron_spiketrain(neuron_a_id)
            spike_times2 = singleneuron_spiketrain(neuron_b_id)
            synchrony_index = float(0)
            for j in np.arange(1,len(marker['time_interval_left_end']),2):
                marker_start = marker['time_interval_left_end'].iloc[j]
                marker_end = marker_start+30
                spike_times1_trail = spike_times1[(spike_times1 > marker_start) & (spike_times1 < marker_end)]
                spike_times2_trail = spike_times2[(spike_times2 > marker_start) & (spike_times2 < marker_end)]
                spiketrain1 = neo.SpikeTrain(spike_times1_trail,units='sec',t_start=marker_start, t_stop=marker_end)
                spiketrain2 = neo.SpikeTrain(spike_times2_trail,units='sec',t_start=marker_start, t_stop=marker_end)
                temp = synchrony_method_01(spiketrain1,spiketrain2,k)  # synchrony_method_01  Calculates the synchrony of spike trains, according to (Han KS et al. Ephaptic. Neuron. 2018).
                if np.isnan(temp) == True:  #如果该neuron pair在该trail下的synindex为0，也应该计算进来，因为是所有trail的平均衡量同步程度
                    temp = 0
                synchrony_index = synchrony_index + temp
                k=k+1  #k是用来画CCG图的
            synchrony_index = synchrony_index / ((len(marker['time_interval_left_end'])+1)/2)  # 分母为trail个数，如果synindex为0也算入
            total_syn = total_syn + synchrony_index
            
            # synchrony_method_02  Calculates the synchrony of spike trains,according to (Ciba et al., 2018).
            #synchrony_index = round(spike_contrast([spiketrain1, spiketrain2]),3)  
            # synchrony_method_03  Calculates the synchrony of spike trains,according to (Grün et al., 2007)
            #__init__(spiketrains, sampling_rate, bin_size=None, binary=True, spread=0, tolerance=1e-08)
        
        region_syn_index = total_syn/len(combinations)  #是对所有neuron配对然后计算synindex，因此如果synindex为0，也应该计算进来
        print(region_syn_index)
        syn_reg_name = np.append(syn_reg_name, region_name)
        syn_reg_index = np.append(syn_reg_index, region_syn_index)

    data = {'region': syn_reg_name, 'synchrony index': syn_reg_index}
    df = pd.DataFrame(data)
    df.to_csv(syn_save_path+f"/{mice}.csv", index=False)

    plt.figure()
    plt.bar(syn_reg_name, syn_reg_index)
    plt.title(f'{mice}')
    plt.xticks(rotation=90)
    plt.xlabel('Region')
    plt.ylabel('Synchrony index')
    plt.savefig(syn_save_path+f"/{mice}_Synchrony index of different region_{method}.png",dpi=600,bbox_inches = 'tight')
    

def compare_syn_index():
    syn_index = load_syn_index_csv()
    #region_name = ['Superior vestibular nucleus','Spinal vestibular nucleus','Medial vestibular nucleus']
    region_name = ['Simple lobule','Lobules IV-V','Lobule III','Lobule II']
    for region in region_name:
        plt.figure()
        filtered_rows = syn_index[syn_index.iloc[:, 0] == region]
        syn_values_mice = filtered_rows.iloc[:, 2].tolist() 
        syn_values = filtered_rows.iloc[:, 1].tolist()
        plt.title(region)
        plt.bar(syn_values_mice,syn_values)
        plt.xticks(rotation=90) 
        plt.savefig(syn_save_path+f"/{region}_Synchrony index of different mice.png",dpi=600,bbox_inches = 'tight')
    
synchrony(neurons,treadmill)
#compare_syn_index()