"""
# coding: utf-8
data from: Xinchao Chen
@author: Yuhao Zhang
last updated: 04/17/2025
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import expon
from mpl_toolkits.mplot3d import Axes3D
np.set_printoptions(threshold=np.inf)
np.seterr(divide='ignore',invalid='ignore')

# ------- Load data -------
'''
# ------- NP1 -------
mice_name = '20230604_Syt2_conditional_tremor_mice2_medial'
sorting_method = 'Xinchao_sort'  
sorting_path = f"/data1/zhangyuhao/xinchao_data/NP1/{mice_name}/Sorted/{sorting_method}/"
treadmill = pd.read_csv(f"/data1/zhangyuhao/xinchao_data/NP1/{mice_name}/Marker/treadmill_move_stop_velocity_segm_trial.csv",index_col=0)
treadmill_origin = pd.read_csv(f"/data1/zhangyuhao/xinchao_data/NP1/{mice_name}/Marker/treadmill_move_stop_velocity.csv",index_col=0)
### electrophysiology
sample_rate = 30000 #spikeGLX neuropixel sample rate
identities = np.load(sorting_path + '/spike_clusters.npy') # time series: unit id of each spike
times = np.load(sorting_path + '/spike_times.npy')  # time series: spike time of each spike
neurons = pd.read_csv(sorting_path + f'/neuron_id_region_firingrate.csv')
print(neurons)
print("Test if electrophysiology duration is equal to treadmill duration ...")
elec_dura = (times[-1]/sample_rate)[0]
treadmill_dura = treadmill_origin['time_interval_right_end'].iloc[-1]
print(f"Electrophysiology duration: {elec_dura}")
print(f"Treadmill duration: {treadmill_dura}")
'''
# ------- NP2 -------
mice_name = '20250310_VN_harmaline' # 20250310_VN_harmaline
sorting_path = f"/data1/zhangyuhao/xinchao_data/NP2/{mice_name}/Sorted/"
### electrophysiology
sample_rate = 30000 #spikeGLX neuropixel sample rate
identities = np.load(sorting_path + '/spike_clusters.npy') # time series: unit id of each spike
times = np.load(sorting_path + '/spike_times.npy')  # time series: spike time of each spike
neurons = pd.read_csv(sorting_path + '/cluster_group.tsv', sep='\t')  
print(neurons)
elec_dura = times[-1] / sample_rate
print(elec_dura)

save_path = f"/home/zhangyuhao/Desktop/Result/ET/ISI/NP2/{mice_name}/"

# ------- Main Program -------
### parameter
fr_filter = 8                # 1  firing rate > 1
cutoff_distr = 80           # 250ms/None  cutoff_distr=0.25代表截断ISI分布大于0.25s的
histo_bin_num = 100          # 统计图bin的个数

def singleneuron_spiketimes(id):
    x = np.where(identities == id)
    y = x[0]
    spike_times = np.empty(len(y))
    for i in range(0, len(y)):
        z = y[i]
        spike_times[i] = times[z] / sample_rate
    return spike_times

def collect_isi_data(units, exclude_top=0):
    all_histograms = []
    valid_unit_ids = []
    spike_counts = []
    
    for index, row in units.iterrows():
        unit_id = row['cluster_id']  # NP1 is neuron_id, NP2 is cluster_id
        spike_times = singleneuron_spiketimes(unit_id)
        spike_count = len(spike_times)  # 直接获取原始发放次数
        
        # 保持原有过滤条件
        if spike_count <= (fr_filter * elec_dura):
            continue
        
        intervals = np.diff(spike_times) * 1000
        
        if cutoff_distr is not None:
            intervals = intervals[(intervals > 0.001) & (intervals <= cutoff_distr)]
            if len(intervals) == 0:
                continue
        
        # 计算直方图
        if cutoff_distr is not None:
            bins = np.linspace(0.001, cutoff_distr, histo_bin_num)
        else:
            bins = 20
        
        counts, bin_edges = np.histogram(intervals, bins=bins)
        
        # 记录数据
        spike_counts.append(spike_count)
        all_histograms.append(counts)
        valid_unit_ids.append(unit_id)
    
    # 按发放次数降序排序
    sorted_data = sorted(zip(spike_counts, valid_unit_ids, all_histograms), 
                        key=lambda x: x[0], 
                        reverse=True)
    
    # 过滤前N个高发放神经元
    filtered_data = sorted_data[exclude_top:]  # 排除前10个
    
    # 解包过滤后的数据
    _, valid_unit_ids, all_histograms = zip(*filtered_data) if len(filtered_data) > 0 else ([], [], [])
    
    return valid_unit_ids, all_histograms, bin_edges

def plot_3d_isi(units):
    # Collect data from all neurons
    unit_ids, histograms, bin_edges = collect_isi_data(units)
    if not histograms:
        print("No valid units with sufficient spikes.")
        return
    # 创建3D画布
    fig = plt.figure(figsize=(36, 24))
    ax = fig.add_subplot(111, projection='3d')
    
    # 优化参数
    BAR_WIDTH = 0.7      # 柱宽
    BAR_DEPTH = 1.2      # 柱深 
    ALPHA = 0.7          # 透明度
    LABEL_GAP = 10        # 标签间隔
    
    # 生成坐标
    y_centers = (bin_edges[:-1] + bin_edges[1:])/2
    dy = np.diff(bin_edges)[0]

    # 绘制3D柱状图
    for i, (unit_id, counts) in enumerate(zip(unit_ids, histograms)):
        xs = i * np.ones_like(y_centers)
        ax.bar3d(xs,
                y_centers - dy/2,
                np.zeros_like(counts),
                BAR_WIDTH,
                BAR_DEPTH,
                counts,
                alpha=ALPHA,
                edgecolor=None)

    # 轴标签优化
    ax.set_xlabel('neuron sort with fr (unit: number)', labelpad=15, fontsize=12)
    ax.set_ylabel('ISI (ms)', labelpad=20, fontsize=12) 
    ax.set_zlabel('Count', labelpad=15, fontsize=12)
    
    # X轴标签间隔显示
    ax.set_xticks(np.arange(0, len(unit_ids), LABEL_GAP))
    ax.set_xticklabels(unit_ids[::LABEL_GAP], 
                      rotation=90,         # 垂直显示
                      fontsize=8,
                      ha='center')         # 居中显示
    
    # 视角调整：正对ISI轴
    ax.view_init(elev=30,  # 俯仰角
                azim=45)   # 方位角设为0度（正对Y轴）
    
    # 添加网格线增强立体感
    ax.xaxis._axinfo["grid"].update({"linewidth":0.5, "color":"#DDDDDD"})
    ax.yaxis._axinfo["grid"].update({"linewidth":0.5, "color":"#DDDDDD"})
    ax.zaxis._axinfo["grid"].update({"linewidth":0.5, "color":"#DDDDDD"})

    # 图形装饰
    plt.title('3D ISI Distribution\nNeuron Firing Patterns', 
             pad=25, 
             fontsize=14)
    plt.savefig(save_path + "/3d_isi_distribution.png", 
               dpi=600, 
               bbox_inches='tight')
    plt.close()

plot_3d_isi(neurons)