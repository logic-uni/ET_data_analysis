"""
# coding: utf-8
@author: Yuhao Zhang
last updated: 02/19/2025
data from: Xinchao Chen
"""
import torch
import numpy as np
import pandas as pd
import csv
import numpy as np
np.set_printoptions(threshold=np.inf)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# path
file_directory=r'E:\chaoge\sorted neuropixels data\20230623-condictional tremor4\202300622_Syt2_512_2_Day18_P79_g0\202300622_Syt2_512_2_Day18_P79_g0_imec0'
region_neuron_id = pd.read_csv(file_directory+'/region_neuron_id.csv', low_memory = False)#防止弹出警告
neuron_info = file_directory+'/cluster_info.tsv'
outputpath = file_directory+'/neuron_id_region_firingrate.csv'

#get regions with neuron_id
region_neruonid = pd.DataFrame(region_neuron_id)
del region_neruonid[region_neruonid.columns[0]]
print(region_neruonid)

#get regions name
regions=region_neruonid.columns.values.tolist()
print(regions)

#get neuron_id with firing_rate
df = pd.read_csv(
    neuron_info,
    sep='\t',
    header=0,
    index_col='cluster_id'
)
df.sort_values(by="depth" , inplace=True, ascending=True)
print(df)

neuron_id=np.array([],dtype=int)
neurons = pd.DataFrame(columns=['neuron_id', 'region', 'firing_rate'])
for region in regions:
    #提取该region的neuron id
    neuron_id=np.array(region_neruonid.loc[:,region])
    neuron_id=neuron_id[~np.isnan(neuron_id)]  #去除NAN值
    #提取这些neuron的firing rate
    region_fir=np.array([])
    for id in neuron_id:
        region_fir = np.append(region_fir, float(df.loc[id, 'fr']))
    #存入dataframe
    for b in range(len(region_fir)):
        neurons.loc[len(neurons)] = [int(neuron_id[b]),region, region_fir[b]]

print(neurons)

neurons.to_csv(outputpath,sep=',',index=True,header=True)
