"""
# coding: utf-8
@author: Yuhao Zhang
last updated: 08/21/2024
data from: Xinchao Chen
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns


# 指定包含 CSV 文件的文件夹路径
data_folder = r'C:\Users\zyh20\Desktop\ET_data analysis\runstop_center_distance'

def each_mice_distri():
    # 获取文件夹中所有文件的列表
    file_list = os.listdir(data_folder)

    # 遍历每个文件
    for file_name in file_list:
        if file_name.endswith('.csv'):  # 假设文件是CSV格式的
            file_path = os.path.join(data_folder, file_name)
            
            # 读取CSV文件
            df = pd.read_csv(file_path)
            
            # 获取第一列作为x，第三列作为y
            x = df.iloc[:, 0]  
            y = df.iloc[:, 2]  
            
            # 绘制柱状图
            plt.figure()
            # 设置横轴文字竖直显示
            plt.xticks(rotation='vertical')
            plt.bar(x, y)
            plt.ylabel('3D run stop distance')
            plt.title(f'{file_name[:-4]}')
            
            # 保存图像到当前文件夹
            plt.savefig(data_folder+f'/3D_run_stop_distance_{file_name[:-4]}.png',dpi=600,bbox_inches = 'tight')  # 去掉.csv扩展名，并保存为PNG格式

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

def scatter_compare(df):
    sns.set_theme(style="whitegrid", palette="muted")
    ax = sns.swarmplot(data=df, x="high_dim_dist", y="region", hue="filename")
    ax.set(ylabel="")
    plt.show()

def threeD_bar_compare(df):
    x_labels = df['filename'].unique()
    y_labels = df['region'].unique()

    x = np.array([np.where(x_labels == filename)[0][0] for filename in df['filename']])
    y = np.array([np.where(y_labels == region)[0][0] for region in df['region']])
    z = np.zeros(len(df))

    # 柱子的宽度和高度
    dx = dy =0.5
    dz = df['high_dim_dist']

    # 调整图形大小
    fig = plt.figure(figsize=(10, 7))  # 增加图形的尺寸
    ax = fig.add_subplot(111, projection='3d')

    # 绘制 3D 柱状图
    ax.bar3d(x, y, z, dx, dy, dz, color='b', zsort='average')

    # 设置轴标签
    ax.set_zlabel('High dimensional distance')

    # 设置 x 轴和 y 轴标签为原始字符标签
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=0, ha='right', fontsize=7)  # 旋转并调整字体大小

    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels, rotation=0, ha='left', va='center', fontsize=7)  # 旋转并调整字体大小
    plt.show()

df = conctenate()
threeD_bar_compare(df)
scatter_compare(df)