"""
# coding: utf-8
@author: Yuhao Zhang
last updated: 03/03/2025
# this code is used to map neuron id - channel id - depth - firing rate - region
this code is for without registration, so artifical divied according to the depth of the channel
"""
import numpy as np
import pandas as pd

# ------- NEED CHANGE -------
data_path = '/data2/zhangyuhao/xinchao_data/test/headtremor/Mice_1411_1/20250108_headtremor_Mice_1411_1_CBN_VN_shank1_2_freely_moving'

# ------- NO NEED CHANGE -------
neu_ch_dep_fr = pd.read_csv(data_path + '/Sorted/kilosort4/cluster_info.tsv', sep='\t')
QC_neuron = pd.read_csv(data_path + '/filtered_quality_metrics.csv')

df_new = neu_ch_dep_fr[['cluster_id', 'ch', 'depth', 'fr']].copy()
df_new = df_new.sort_values('depth').reset_index(drop=True)

min_depth = df_new['depth'].min()
max_depth = df_new['depth'].max()
'''
## 二等分
step = (max_depth - min_depth) / 2.0  # 计算每个区间的长度
boundary1 = min_depth + step

df_new['region'] = np.select(
    [
        df_new['depth'] <= boundary1,
        df_new['depth'] > boundary1
    ],
    [
        'VN',
        'DCN',
    ]
)
'''
## 三等分
step = (max_depth - min_depth) / 3.0  # 计算每个区间的长度
boundary1 = min_depth + step
boundary2 = min_depth + 2 * step

df_new['region'] = np.select(
    [
        df_new['depth'] <= boundary1,
        (df_new['depth'] > boundary1) & (df_new['depth'] <= boundary2),
        df_new['depth'] > boundary2
    ],
    [
        'VN',
        'DCN',
        'CbX'
    ]
)

print(df_new)
print("\n各区域数量统计:")
print(df_new['region'].value_counts())
df_new.to_csv(data_path+'/Sorted/kilosort4/mapping_artifi.csv', index=False)

qc_neurons = QC_neuron.iloc[:, 0].tolist()
result_df = df_new[df_new['cluster_id'].isin(qc_neurons)]
print(result_df)
print("\n各区域数量统计:")
print(result_df['region'].value_counts())
result_df.to_csv(data_path+'/Sorted/kilosort4/mapping_artifi_QC.csv', index=False)