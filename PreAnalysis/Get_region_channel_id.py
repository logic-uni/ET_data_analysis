"""
@Author: Yuhao Zhang
Last updated : 02/19/2025
Data from: Xinchao Chen
"""

import numpy as np
import pandas as pd

file_directory=r'E:\chaoge\sorted neuropixels data\20230623-condictional tremor4\202300622_Syt2_512_2_Day18_P79_g0\202300622_Syt2_512_2_Day18_P79_g0_imec0'
neurons = pd.read_csv(file_directory+'/region_neuron_id.csv', low_memory = False,index_col=0)#防止弹出警告
cluster_channel = pd.read_csv(file_directory+'/cluster_info.tsv', sep='\t')
outputpath = file_directory+'/region_channel.csv'
region_name = neurons.columns.to_numpy()
dict = {}

for i in range(neurons.shape[1]):  #遍历所有的脑区
    neuron_id = np.array(neurons.iloc[:, i].dropna()).astype(int)
    # 筛选DataFrame中的行，其中第一列包含在一维数组中的值
    filtered_df = cluster_channel[cluster_channel['cluster_id'].isin(neuron_id)]
    channel = filtered_df['ch'].values
    channel = np.unique(channel)
    column_name = str(region_name[i])
    dict[column_name] = channel+384 

df = pd.DataFrame.from_dict(dict, orient='index').transpose()
df.to_csv(outputpath)