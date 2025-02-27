"""
# coding: utf-8
@author: Yuhao Zhang
last updated: 05/28/2024
data from: Xinchao Chen
"""
import torch
import numpy as np
import pandas as pd
import pynapple as nap
import pynacollada as pyna
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns 
import csv
from scipy.stats import pareto
from itertools import count
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
from sklearn.datasets import load_iris,load_digits
from sklearn.decomposition import PCA
from matplotlib.colors import hsv_to_rgb
from mpl_toolkits.axes_grid1 import make_axes_locatable
from math import log
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
import seaborn as sns
import scipy.io as sio
np.set_printoptions(threshold=np.inf)
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### path
main_path = r'E:\xinchao\sorted neuropixels data\useful_data\20230113_littermate\data'
csv_save_path = r'C:\Users\zyh20\Desktop\ET_data analysis\firingrate_distribution\all_time_no_filter\littermate.csv'

### marker
treadmill = pd.read_csv(main_path+'/marker/treadmill_move_stop_velocity.csv',index_col=0)
print(treadmill)

### electrophysiology
sample_rate=30000 #spikeGLX neuropixel sample rate
identities = np.load(main_path+'/spike_train/spike_clusters.npy') #存储neuron的编号id,对应phy中的第一列id
times = np.load(main_path+'/spike_train/spike_times.npy')  #
channel = np.load(main_path+'/spike_train/channel_positions.npy')
region_neruonid = pd.read_csv(main_path+'/spike_train/region_neuron_id.csv', low_memory = False,index_col=0)#防止弹出警告
neuron_info = main_path+'/spike_train/cluster_info.tsv'

print(region_neruonid)
print("检查treadmill总时长和电生理总时长是否一致")
print("电生理总时长")
print((times[-1]/sample_rate)[0])
print("跑步机总时长") 
print(treadmill['time_interval_right_end'].iloc[-1])
neuron_num = region_neruonid.count().transpose().values

def getregion_fr_csv():
    #get regions name
    regions=region_neruonid.columns.values.tolist()
    print(regions)

    #get neuron_id with firing_rate
    neurons_info = pd.read_csv(
        neuron_info,
        sep='\t',
        header=0,
        index_col='cluster_id'
    )
    neurons_info.sort_values(by="depth" , inplace=True, ascending=True)  # df index 为 neuron id, 后面包括各列，有一列的名称为fr

    neuron_id=np.array([],dtype=int)
    regions_fr = pd.DataFrame(columns=['region', 'firing_rate'])

    for region in regions:
        #提取该region的neuron id
        neuron_id=np.array(region_neruonid.loc[:,region])
        neuron_id=neuron_id[~np.isnan(neuron_id)].astype(int)  #去除NAN值
        #提取这些neuron的firing rate
        subregion_fr=np.array([])
        for id in neuron_id:
            subregion_fr = np.append(subregion_fr, float(neurons_info.loc[id, 'fr']))  #index 为 Neuron id
        #存入dataframe
        for fr in subregion_fr:
            regions_fr.loc[len(regions_fr)] = [region, fr]

    regions_fr.to_csv(csv_save_path,sep=',',index=False,header=True)
    return regions_fr

def plot_onemice_region_fr(df):
    df['firing_rate'] = pd.to_numeric(df['firing_rate'], errors='coerce')

    # 查看转换后的数据
    print(df.head())
    print(df.dtypes)
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='firing_rate', hue='region', multiple='stack', bins=50, kde=False)
    plt.title('Firing Rate Distribution by Region')
    plt.xlabel('Firing Rate')
    plt.ylabel('Count')
    plt.show()
    plt.figure(figsize=(10, 6))

    # 对每个region单独进行拟合和绘图
    for region in df['region'].unique():
        values = df[df['region'] == region]['firing_rate']
        
        # 拟合帕累托分布
        shape, loc, scale = pareto.fit(values, floc=0)
        
        # 绘制直方图
        sns.histplot(values, bins=50, kde=False, label=f'{region} Data', stat='density', alpha=0.5)
        
        # 绘制拟合曲线
        x = np.linspace(min(values), max(values), 100)
        pdf_fitted = pareto.pdf(x, shape, loc, scale)
        plt.plot(x, pdf_fitted, label=f'{region} Pareto Fit')

    plt.title('Firing Rate Distribution with Pareto Fit by Region')
    plt.xlabel('Firing Rate')
    plt.ylabel('Density')
    plt.legend()
    plt.show()


regions_fr = getregion_fr_csv()
#plot_onemice_region_fr(regions_fr)