"""
# coding: utf-8
@author: Yuhao Zhang
last updated: 08/22/2024
data from: Xinchao Chen
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# 指定包含 CSV 文件的文件夹路径
data_folder = r'C:\Users\zyh20\Desktop\ET_data analysis\Null space\Syt2_conditional_tremor_mice2_lateral'
main_path = r'E:\xinchao\sorted neuropixels data\useful_data\20230602_Syt2_conditional_tremor_mice2_lateral\data'

def conctenate():
    # 获取所有CSV文件的文件名
    csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]

    # 创建一个空的DataFrame
    combined_df = pd.DataFrame()

    # 遍历每个CSV文件
    for file in csv_files:
        # 读取CSV文件
        file_path = os.path.join(data_folder, file)
        df = pd.read_csv(file_path)
        
        # 去掉文件名中的 .csv 后缀
        filename = os.path.splitext(file)[0]
        
        # 在最左侧添加文件名列
        df.insert(0, 'filename', filename)
        
        # 将当前DataFrame拼接到总的DataFrame
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    # 查看拼接后的DataFrame
    print(combined_df)
    return combined_df

def plot(df):
    treadmill = pd.read_csv(main_path+'/marker/treadmill_move_stop_velocity.csv',index_col=0)
    difference_array = treadmill['time_interval_right_end'] - treadmill['time_interval_left_end']
    trial_times = difference_array.to_numpy()
    '''
    filtered_times = []
    filtered_df = df
    for j in range(0,len(trial_times)):
        print(j)
        if trial_times[j] < 50:
            filtered_df = filtered_df[filtered_df['trial_num'] != j]
        else:
            filtered_times.append(trial_times[j])
    '''
    # Ensure that non-numeric columns are excluded when calculating the mean
    grouped = df.groupby(['region', 'trial_num'], as_index=False).agg({'constraint': 'mean'})

    # Plotting the line chart
    plt.figure(figsize=(10, 6))
    for region in grouped['region'].unique():
        region_data = grouped[grouped['region'] == region]
        plt.plot(region_data['trial_num'], region_data['constraint']/trial_times, marker='o', label=region)

    plt.xlabel('Trial Number')
    plt.ylabel('Constraint Value')
    plt.title('Constraint Value by Region and Trial Number')
    plt.legend(title='Region')
    plt.grid(True)
    plt.show()

combined_df = conctenate()
plot(combined_df)