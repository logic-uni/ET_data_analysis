"""
# coding: utf-8
@author: Yuhao Zhang
last updated: 06/01/2024
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
from scipy.signal import hilbert
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
import warnings
import scipy.io as sio

path = r'E:\chaoge\sorted neuropixels data\20230113-litermate'
marker = pd.read_csv(path+'/20230112_0338_3_litermate_female_recording_Neuropixels.txt',sep = '|',dtype = 'str')
outputpath=path+'/treadmill_move_stop.csv'
marker['Time'] = marker['Time'].str.replace(' sec', '')   #删除第一列的sec字符
marker=marker.astype(float)   #设置所有marker类型为float

#Treadmill二值化
marker['0'] = marker['0'].apply(lambda x: 1 if x > 2 else 0)

#提取开始运动的时间，掐头，没有第一个
treadmill_up=marker.loc[marker['0'].diff() == 1]      #提取电机PWM所有上升沿
move_df=treadmill_up[treadmill_up['Time'].diff() > 0.5]  #运转时上升沿时刻点之间差值小于0.5，差值大于0.5的点是下一次运转的开始时间
#加上第一个上升沿
first_row = treadmill_up.iloc[[0]]
move_df = pd.concat([first_row, move_df], ignore_index=True)

#提取运动结束的时间，去尾，没有最后一个
treadmill_down=marker.loc[marker['Treadmill_marker'].diff() == -1]   #提取电机PWM所有下降沿
rows_to_extract = treadmill_down.index[(treadmill_down['Time'].shift(-1) - treadmill_down['Time']) > 0.5].tolist()  #运转时下降沿时刻点之间差值小于0.5，差值大于0.5的点的上一个点是运转的结束时间
stop_df = treadmill_down.loc[rows_to_extract]
#加上最后一个下降沿
last_row = treadmill_down.iloc[[-1]]
stop_df = pd.concat([stop_df, last_row], ignore_index=True)

move=np.array(move_df['Time'])
stop=np.array(stop_df['Time'])
move_stop=np.insert(stop,np.arange(len(move)),move)

np.savetxt(outputpath, move_stop, delimiter=",")