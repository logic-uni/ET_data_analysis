"""
# coding: utf-8
@author: Yuhao Zhang
last updated: 05/29/2024
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
import os
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

path = r'C:\Users\zyh20\Desktop\fr_dataminning\region_firingrate'
outputpath='C:/Users/zyh20/Desktop/neurons_mice.csv'
mice_region_fr_path='C:/Users/zyh20/Desktop/mice_region_fr.csv'

#读取path文件夹下所有的csv文件，即各个鼠的neuron region_firing rate csv
files = os.listdir(path)
files_csv = [f for f in files if f[-4:] == '.csv']
'''
#单个region合并所有mice
focus_region='Superior vestibular nucleus'
neurons_mice = pd.DataFrame(columns=['mice', 'firing_rate'])
for file in files_csv:
    neurons = pd.read_csv(os.path.join(path, file))
    del neurons[neurons.columns[0]]
    neurons.set_index('region', inplace=True)
    if focus_region in neurons.index.values:
        focus_region_fr=neurons.loc[focus_region, 'firing_rate']
        for fr in focus_region_fr:
            neurons_mice.loc[len(neurons_mice)] = [file, fr]

neurons_mice['mice'] = neurons_mice['mice'].str.replace ('.csv', '', regex= True )

print(neurons_mice)
neurons_mice.to_csv(outputpath,sep=',',index=True,header=True)
sns.displot(neurons_mice, x="firing_rate", hue="mice", kind="kde", fill=True)
'''


#所有region合并所有mice

mice_region_fr=pd.DataFrame()
for file in files_csv:
    mice_region = pd.read_csv(os.path.join(path, file))
    del mice_region[mice_region.columns[0]]
    # 创建一个字典来存储每个名字对应的值
    grouped = mice_region.groupby('region')['firing_rate'].apply(list).to_dict()

    # 找到最长的组
    max_len = max(len(values) for values in grouped.values())

    # 创建一个空的DataFrame用于存储结果
    new_df = pd.DataFrame(index=range(max_len))

    # 将数据填充到新的DataFrame中
    for name, values in grouped.items():
        # 如果当前组的长度小于最大长度，则用NaN填充
        if len(values) < max_len:
            values.extend([np.nan] * (max_len - len(values)))
        new_df[name] = values
    
    #添加一列以标明mice
    new_df.insert(0, 'mice', file)
    new_df['mice'] = new_df['mice'].str.replace ('.csv', '', regex= True )
    print(new_df)
    
    mice_region_fr = pd.concat([mice_region_fr, new_df], ignore_index=True)
    '''
    g = sns.PairGrid(new_df, diag_sharey=False)
    g.map_upper(sns.scatterplot, s=15)
    g.map_lower(sns.kdeplot)
    g.map_diag(sns.kdeplot, lw=2)
    plt.show()
    
    print(new_df)
    new_df.plot.hist(alpha=0.5,bins=100,density=True)
    plt.show()
    # alpha 代表重叠的透明度，如果是 0.9，表示 90% 的颜色为 'b'
    # bins 代表一组数据有多少根柱子，即把数据分成多少份
    '''
    

#mice_region_fr.to_csv(mice_region_fr_path,sep=',',index=True,header=True)

#筛选感兴趣的脑区
#miceregion_compare=mice_region_fr[['mice','Lobule III','Lobules IV-V','Superior vestibular nucleus','Medial vestibular nucleus','Parvicellular reticular nucleus']]
mice_region_fr = mice_region_fr[mice_region_fr['mice'] != '20230113-litermate']

sns.pairplot(data=mice_region_fr)
sns.set_theme(rc={'figure.figsize':(5,5)})
#g.fig.set_figheight(5)
#g.fig.set_figwidth(5)
plt.savefig('C:/Users/zyh20/Desktop/0054.png', dpi=600, bbox_inches='tight')

'''
g = sns.PairGrid(mice_region_fr, diag_sharey=False)
g.map_upper(sns.scatterplot, s=15)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot, lw=2)
'''


