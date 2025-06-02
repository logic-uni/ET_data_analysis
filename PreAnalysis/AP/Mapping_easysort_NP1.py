"""
# coding: utf-8
@author: Yuhao Zhang
last updated: 03/03/2025
Running this code will get three results:
1. A 3D plot of the DiI dot, brain surface, brain bottom, tip DiI, tip projection, record start, and record end.
2. A 2D plot of the recording depth, region, and unit id.
3. A csv file "unit_ch_dep_region.csv", mapping unit id, channel id, LFP channel id (in spikeGLX), depth, region.
"""
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ------- NEED CHANGE -------
mice_name = '20230623_Syt2_conditional_tremor_mice4'
QC_metric = None  # None/'isi_violations_ratio'/'amplitude_cutoff'/'presence_ratio'

# ------- NO NEED CHANGE -------
# load neuron id - channel id - depth
# Mind: depth 0 is "channel 0" dpeth in sorting data
sorting_path = rf'E:\xinchao\Data\useful_data\{mice_name}\Sorted\Easysort'
neuron_channel = pd.read_csv(sorting_path+'/peak_channels.csv')
depth_channel = pd.read_csv(sorting_path+'/probe_info.csv')  #neuropixel channel 0 位于最下方，depth也是基于最下方 channel 0 开始向上计算
# load qc_metrics
quality_metrics = pd.read_csv(sorting_path+r'\waveforms\extensions\quality_metrics\metrics.csv')
# load depth - region
registration_file = scipy.io.loadmat(rf'C:\Users\zyh20\Desktop\Research\01_ET_data_analysis\Research\registration\{mice_name}\Process\slices\coordinate.mat')
# load spike times
identities = np.load(sorting_path+'/results_KS2/sorter_output/spike_clusters.npy')  # unit id of each spike
times = np.load(sorting_path+'/results_KS2/sorter_output/spike_times.npy')  # spike time of each spike
sample_rate = 30000  # spikeGLX neuropixel sample rate

def threeDTr_ch_starend_dep():
    #由于registration得到的是一个neuropixel所在直线穿过的全部脑区和深度的信息，并给出了荧光点
    #基于首尾荧光点，计算出电极插入的深度，电极最上方的channel所在深度，电极Tip所在深度
    depth = registration_file['depth']
    DiI = registration_file['DiI']
    firstend = registration_file['firstend']  # 脑表面和脑底部坐标
    # 创建一个新的三维坐标系
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # DiI坐标
    x, y, z = DiI[:, 0], DiI[:, 1], DiI[:, 2]
    # 脑表面坐标
    x1 = firstend[0,0]
    y1 = firstend[0,1]
    z1 = firstend[0,2]
    # 脑底部坐标
    x2 = firstend[1,0]
    y2 = firstend[1,1]
    z2 = firstend[1,2]
    # 绘制点
    ax.scatter(x, y, z,color='blue',label="DiI")
    ax.scatter(x1, y1, z1,color='red',label="brain_surface")
    ax.scatter(x2, y2, z2,color='green',label="brain_bottom")
    ax.plot([x1, x2], [y1, y2], [z1, z2], color='blue', linewidth=2)
    # 设置图形的标题和坐标轴标签
    ax.set(xlabel='X', ylabel='Y', zlabel='Z')
    #Tip点到直线的投影
    dii_endpoint = DiI[-1]
    ax.scatter(dii_endpoint[0], dii_endpoint[1], dii_endpoint[2],color='black',label="tip_DiI")
    surface = np.array([float(x1), float(y1),float(z1)])
    bottom = np.array([float(x2),float(y2),float(z2)])
    # 计算点到线段的投影
    ab = surface - bottom
    unit_v = ab / np.linalg.norm(ab)
    ap = dii_endpoint - bottom
    projection = bottom + np.dot(ap, unit_v) * unit_v
    print("tip到拟合直线的投影点坐标:", projection)
    ax.scatter(projection[0], projection[1], projection[2],color='yellow',label="tip_project")
    #电极插入的深度
    depth = np.sqrt(np.sum((projection - surface)**2))*10 # 计算两点之间的距离
    print("电极深度:", depth)
    ax.set_title('neuropixel depth : {} um'.format(depth))
    
    # 计算结束点的坐标
    end_point = projection + (17.5 / np.linalg.norm(ab)) * ab
    # 计算开始点的坐标
    start_point = end_point + (384 / np.linalg.norm(ab)) * ab
    surface = [x1, y1, z1]
    v1 = start_point-surface
    v2 = end_point-surface
    dot = np.dot(v1, v2)
    if dot<0:
        start_point = surface

    ax.scatter(end_point[0], end_point[1], end_point[2],color='brown',label="record_end")
    ax.scatter(start_point[0], start_point[1], start_point[2],color='pink',label="record_start")
    print("记录开始点的坐标:", start_point)
    print("记录结束点的坐标:", end_point)
    record_start = depth - 175
    record_end = record_start - 3840
    if record_end < 0:
        record_end=0
    print("记录开始点深度:",record_start)  
    print("记录结束点深度:",record_end)
    ax.legend()
    plt.show()

    return record_start, record_end  #record_start指电极最上方的channel所在深度，record_end指电极Tip所在深度

def map_unit_ch_depth(unit_ch,dep_ch):
    # Modify the 'peak_channel' column to remove words before the final number
    unit_ch['peak_channel'] = unit_ch['peak_channel'].str.extract(r'(\d+)$')
    # Keep only the 'id' and 'y' columns in depth_channel
    dep_ch = dep_ch[['id', 'y']]
    unit_ch = unit_ch.rename(columns={'unit_id': 'cluster_id'})
    dep_ch = dep_ch.rename(columns={'id': 'ch'})
    dep_ch = dep_ch.rename(columns={'y': 'depth'})
    unit_ch = unit_ch.rename(columns={'peak_channel': 'ch'})
    unit_ch['ch'] = unit_ch['ch'].astype(int)
    # Merge neuron_channel with depth_channel to add the depth column
    unit_ch_depth = unit_ch.merge(dep_ch, on='ch', how='left')
    # Add a column of spikeGLX_LFP_channel in csv, for when exported LFP file of different brain region from spikeGLX
    # SpikeGLX exported LFP channel number = AP channel number + 384, but Actual LFP channel number = AP channel number
    unit_ch_depth['LFP_SpikeGLX_channel'] = unit_ch_depth['ch'] + 384
    # compute each neuron firing rate
    for id in unit_ch_depth['cluster_id']:
        count = np.sum(identities == id)
        duration = (times[-1] / sample_rate)[0]  # 单位：秒
        fr = count / duration  # 单位：个数/秒
        unit_ch_depth.loc[unit_ch_depth['cluster_id'] == id, 'fr'] = fr
    
    # Quality Control
        # isi_violations < 0.5
        # amplitude_cutoff < 0.1
        # presence_ratio > 0.9
    if QC_metric is not None:
        if QC_metric == 'isi_violations_ratio':
            filtered_neurons = quality_metrics[quality_metrics[QC_metric] < 0.5]
        elif QC_metric == 'amplitude_cutoff':
            filtered_neurons = quality_metrics[quality_metrics[QC_metric] < 0.1]
        elif QC_metric == 'presence_ratio':
            filtered_neurons = quality_metrics[quality_metrics[QC_metric] > 0.9]
        filtered_neuron_id = filtered_neurons.iloc[:, 0]
        # Convert filtered_neuron_id to numpy array
        filtered_neuron_id_np = filtered_neuron_id.to_numpy()
        # Filter neuron_channel_depth by cluster_id using filtered_neuron_id_np
        unit_ch_depth = unit_ch_depth[unit_ch_depth['cluster_id'].isin(filtered_neuron_id_np)]

    return unit_ch_depth

def plt_tr_region_dep_unit(recording_depth,region_depth_real,region,region_name):
    # ------ 2D Plot -------
    ## In the figure, recording depth means depth saved in the phy file, real depth means the real depth in the brain.
    fig, ax = plt.subplots(figsize = (15, 15))
    ax.vlines(3, 0, 3840, linestyles='solid', colors='blue')  #画竖直线
    m=0
    for i in range(len(recording_depth)):
        ax.plot(3, recording_depth[i], marker='_', color='red', markersize=8)
        ax.text(4,recording_depth[i],recording_depth[i],fontsize=8)
        ax.text(4.5,recording_depth[i],region_depth_real[m],fontsize=8)
        m=m+1
    for k in range(len(region)):
        ax.text(3.3,(recording_depth[k]+recording_depth[k+1])/2,region_name[k],fontsize=8)

    ax.set_xlim(2.5, 6.5)
    ax.set_ylim(-3840*0.05, 3840*1.05,100)
    p=3840*1.03
    ax.text(2.96,p,'electrode',fontsize=8,fontdict={'style':'italic', 'weight':'bold'})
    ax.text(3.3,p,'region',fontsize=8,fontdict={'style':'italic', 'weight':'bold'})
    ax.text(4,p,'recording_depth',fontsize=8,fontdict={'style':'italic', 'weight':'bold'})
    ax.text(4.5,p,'real_depth',fontsize=8,fontdict={'style':'italic', 'weight':'bold'})
    plt.show()

def map_dep_region(record_start, record_end):
    region = np.array(registration_file['region'])
    depth = registration_file['depth']

    #从registration_file里截取有channel记录的部分
    rec_interval_suf_zero = depth[(depth>record_end) & (depth<record_start)] 
    rec_interval_suf_zero = np.insert(rec_interval_suf_zero, 0, record_end) #在数组开头插入最上方channel所在深度
    rec_interval_suf_zero = np.append(rec_interval_suf_zero, record_start) #在数组末尾插入tip所在深度
    if len(rec_interval_suf_zero) % 2 == 0:
        rec_interval_suf_zero = rec_interval_suf_zero.reshape(-1, 2) #判断数组长度是否为偶数
    else:
        print("数组长度不是偶数，无法每两个组成一组。")
    
    #截取记录部分的脑区
    start_position = np.where(depth>rec_interval_suf_zero[0,0])[0][0] #找到最上方channel在深度数组中的位置
    region = region[start_position:len(rec_interval_suf_zero) + start_position]  
    region_name = np.array([])
    for x in region:
        region_name = np.append(region_name,x[0])

    rec_interval_top_ch_zero = rec_interval_suf_zero - rec_interval_suf_zero[0,0]    #以最上方channel为零点对齐，即所有点都减去最上方channel的深度（以脑表面为零点）
    rec_interval_tip_zero = rec_interval_top_ch_zero[-1][1] - rec_interval_top_ch_zero    # 计算以tip为零点的depth，对应图中的recording_depth
    print(rec_interval_suf_zero)   # filtered_array是分段的数组，比如[   963.     1473.    ]，[ 1473.     1994.    ], ...
    print(rec_interval_top_ch_zero)     # region_depth_norm是分段的数组，比如[   0.     510.    ]，[ 510.     1031.    ], ...
    print(rec_interval_tip_zero)
    print(region_name)
    # Merge rec_depth_interval, rec_dep_norm_tip, region_name into a DataFrame
    dep_region = pd.DataFrame({
        'suf_zero_start': rec_interval_suf_zero[:, 0],
        'suf_zero_end': rec_interval_suf_zero[:, 1],
        'tip_zero_end': rec_interval_tip_zero[:, 0],
        'tip_zero_start': rec_interval_tip_zero[:, 1],
        'region_name': region_name
    })
    print(dep_region)
    dep_region.to_csv(sorting_path+'/mapping/dep_region.csv', index=False)

    # 画二维图
    region_depth_real = np.delete(rec_interval_suf_zero.flatten(), np.arange(1, len(rec_interval_suf_zero.flatten())-1, 2))  # 摊平为一维数组中的各个区间点
    recording_depth = np.delete(rec_interval_tip_zero.flatten(), np.arange(1, len(rec_interval_suf_zero.flatten())-1, 2))  # 摊平为一维数组中的各个区间点
    plt_tr_region_dep_unit(recording_depth, region_depth_real, region, region_name)

    return dep_region

def merge(unit_ch_dep,dep_region): 
    unit_ch_dep_region = unit_ch_dep
    unit_ch_dep_region['region'] = None  # Add a column in unit_ch_depth called "region"

    for index, row in unit_ch_dep_region.iterrows():
        depth = row['depth']
        for _, region_row in dep_region.iterrows():
            if region_row['tip_zero_start'] <= depth <= region_row['tip_zero_end']:
                unit_ch_dep_region.at[index, 'region'] = region_row['region_name']
                break
    
    neuron_num = neuron_channel.shape[0]
    neuron_num_QC = unit_ch_dep_region.shape[0]
    pass_rate = 100 * neuron_num_QC / neuron_num
    unit_ch_dep_region.to_csv(sorting_path+f'/mapping/unit_ch_dep_region_QC_{QC_metric}_pass_rate_{pass_rate}%.csv', index=False)


# Mapping unit id - channel id - depth
unit_ch_depth = map_unit_ch_depth(neuron_channel,depth_channel) 
print(unit_ch_depth)
# Mapping depth - region
record_start, record_end = threeDTr_ch_starend_dep()
depth_region = map_dep_region(record_start, record_end)
# Merge the two mapping results, through depth
merge(unit_ch_depth,depth_region)