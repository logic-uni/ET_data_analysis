"""
# coding: utf-8
@author: Yuhao Zhang
last updated: 06/20/2024
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
import numpy as np
import scipy.stats as stats
import warnings
import scipy.io as sio
import scipy.stats as stats
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
from scipy.signal import hilbert, butter, filtfilt
from scipy.fftpack import fft,fftfreq,rfft,irfft,ifft
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d
from itertools import combinations

path = r'E:\chaoge\sorted neuropixels data\20230523-condictional tremor1\20230523\raw\20230523_Syt2_449_1_Day50_g0'
marker = pd.read_csv(path+'/20230523_Syt2_449_1_Day50_g0_t0.exported.nidq.csv', encoding='utf-8',header=None)

print(len(marker))
time=np.arange(0, len(marker))
time=time/10593.22041272369  #每秒钟采集10,593.22041272369次  sample rate of marker of spikeGLX
print(time[-1])

marker.insert(loc=0, column='Time', value=time)
marker.columns = ['Time', 'Treadmill_marker']
outputpath=path+'/treadmill_move_stop_velocity.csv'
print(marker)
#Treadmill二值化
marker['Treadmill_marker'] = marker['Treadmill_marker'].apply(lambda x: 1 if x > 2 else 0)

def treadmill_move_stop_time(marker):
    #提取开始运动的时间，掐头，没有第一个
    treadmill_up=marker.loc[marker['Treadmill_marker'].diff() == 1]      #提取电机PWM所有上升沿
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
    move_stop = np.insert(stop,np.arange(len(move)),move)
    move_stop = np.append(move_stop, time[-1])

    move_stop=np.insert(move_stop,0,0)
    move_stop_interval=np.array([[move_stop[i], move_stop[i+1]] for i in range(len(move_stop) - 1)])
    print(move_stop_interval)
    output = pd.DataFrame({
        'time_interval_left_end': [list(row)[0] for row in move_stop_interval],
        'time_interval_right_end': [list(row)[1] for row in move_stop_interval],
        'run_or_stop': [i % 2 for i in range(len(move_stop_interval))],  # 添加一列，0表示跑步机静止状态，1表示跑步机运动状态
        'velocity_recording':np.zeros(len(move_stop_interval)),
        'velocity_level': np.full(len(move_stop_interval), np.nan),
        'velocity_theoretical':np.zeros(len(move_stop_interval))
    })
    stop=move_stop_interval[::2]
    run=move_stop_interval[1::2] 

    print("跑步机运动时间区间：")
    print(run)
    print("跑步机静止时间区间：")
    print(stop)
    return run,stop,output

def calculate_duty_cycle_from_waveform(waveform):
    """
    计算给定波形的占空比

    :param waveform: 一个表示波形的列表（仅包含0和1）
    :return: 占空比（百分比）
    """
    high_time = sum(waveform)  # 高电平时间为所有1的个数
    period = len(waveform)  # 总周期时间为波形的总长度

    duty_cycle = (high_time / period) * 100
    print(f"占空比: {duty_cycle}%")
    return duty_cycle

def classify_value(value):
    if value < 3.8:
        return 0
    elif 3.8 <= value < 4.8:
        return 1
    elif 4.8 <= value < 6.1:
        return 2
    else:
        return 3

run=treadmill_move_stop_time(marker)[0]
output=treadmill_move_stop_time(marker)[2]

#跑步机速度
#motor 6 cm/s / v
#Experimental: based on PWM Duty cycle, V(cm/s) = Duty cycle(%) * 5(V) * 6(cm/s / v) 100%是5V
velocity_value = np.array([])
velocity_theor = np.array([])
for interval in run:

    # 筛选出第一列的值在区间内的行
    filtered_df = marker[(marker['Time'] >= interval[0]) & (marker['Time'] <= interval[1])]
    # 取出这些行中第二列的值
    PWM_series = filtered_df['Treadmill_marker'].tolist()
    duty_cycle=calculate_duty_cycle_from_waveform(PWM_series)
    v=duty_cycle*5*6/100
    print(v)
    velocity_value=np.append(velocity_value, v)

print(velocity_value)
plt.plot(velocity_value)
plt.ylim(0,10)
plt.show()

#Theoretical: based on code
PWMVoltage = np.array([0.6,0.75,0.9,1.05])
for vtheor in PWMVoltage:
    velocity_theor=np.append(velocity_theor, vtheor*6)
print(velocity_theor)

for i in range(len(velocity_value)):
    output.loc[i*2+1, 'velocity_recording'] = velocity_value[i]
    output.loc[i*2+1, 'velocity_level'] = classify_value(velocity_value[i])
    output.loc[i*2+1, 'velocity_theoretical'] = velocity_theor[classify_value(velocity_value[i])]

print(output)

output.to_csv(outputpath,sep=',',index=True,header=True)
