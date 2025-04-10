"""
# coding: utf-8
@author: Yuhao Zhang
last updated: 09/03/2024
data from: Xinchao Chen
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
from scipy.stats import expon
from scipy.stats import gaussian_kde

# 指定包含 CSV 文件的文件夹路径
data_folder = r'C:\Users\zyh20\Desktop\ET_data analysis\firingrate_distribution\all_time_no_filter'
fig_save_path = r'C:\Users\zyh20\Desktop\ET_data analysis\firingrate_distribution\all_time_no_filter'

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

def twoD_slice_plotcompare(df):
    # 获取所有的 region 类别
    regions = df['region'].unique()

    # 计算整个数据集中 firing_rate 的最小值和最大值
    firing_rate_min = df['firing_rate'].min()
    firing_rate_max = df['firing_rate'].max()

    # 设置统一的bin
    bins = np.linspace(firing_rate_min, firing_rate_max, 50)

    # 设置颜色调色板
    palette = sns.color_palette("husl", len(df['filename'].unique()))

    # 为每个 region 创建三维图，绘制概率直方图和概率分布曲线
    for i, region in enumerate(regions):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        df_region = df[df['region'] == region]
        filenames = df_region['filename'].unique()
        
        for j, filename in enumerate(filenames):
            df_file = df_region[df_region['filename'] == filename]
            
            # 使用统一的bin，并使用概率模式绘制直方图
            hist, bin_edges = np.histogram(df_file['firing_rate'], bins=bins, weights=np.ones(len(df_file['firing_rate'])) / len(df_file['firing_rate']))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # 绘制3D直方图的柱子
            ax.bar(bin_centers, hist, zs=j, zdir='y', width=(bin_edges[1] - bin_edges[0]), alpha=0.7, color=palette[j])
            
            # 使用scipy.stats.expon进行概率分布拟合
            params = expon.fit(df_file['firing_rate'])
            
            # 生成概率分布曲线的数据
            x_fit = np.linspace(bin_centers.min(), bin_centers.max(), 100)
            y_fit = expon.pdf(x_fit, *params) * (bin_edges[1] - bin_edges[0])  # 调整以匹配直方图的概率尺度
            
            # 在3D图上绘制概率分布曲线
            ax.plot(x_fit, y_fit, zs=j, zdir='y', color=palette[j], lw=2)

        # 设置轴标签
        ax.set_xlabel('Firing Rate')
        ax.set_zlabel('Probability')

        # 设置 y 轴刻度标签为 filename 名称
        ax.set_yticks(range(len(filenames)))
        ax.set_yticklabels(filenames,rotation=0, ha='left', va='center', fontsize=7)
        
        # 设置标题
        ax.set_title(f'{region}_firingrate_distribution_compare')

        plt.savefig(fig_save_path+f"/{region}_firingrate_distribution_compare.png",dpi=600,bbox_inches = 'tight')

def thrD_plotcompare(df):
    # 获取所有的 region 类别
    regions = df['region'].unique()
    filenames = df['filename'].unique()

    # 设置颜色调色板
    palette = sns.color_palette("husl", len(filenames))

    # 创建颜色字典，并手动设置 "littermate" 为黑色
    color_dict = {filename: ('black' if filename == 'littermate' else palette[i]) for i, filename in enumerate(filenames)}

    # 创建3D图
    fig = plt.figure(figsize=(15, 10))  # 增大图像的整体大小
    ax = fig.add_subplot(111, projection='3d')

    # 增加 bins 数量
    bin_count = 100

    # 遍历每个 region，并为每个 region 绘制一个切片
    for i, region in enumerate(regions):
        df_region = df[df['region'] == region]
        
        for j, filename in enumerate(filenames):
            df_file = df_region[df_region['filename'] == filename]
            
            if df_file.empty:
                continue
            
            # 绘制直方图  probability mode
            hist, bins = np.histogram(df_file['firing_rate'], bins=bin_count, range=(min(df['firing_rate']), max(df['firing_rate'])), density=True)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            # 绘制3D直方图的柱子
            ax.bar(bin_centers, hist, zs=i, zdir='y', width=(bins[1] - bins[0]), alpha=0.7, 
                color=color_dict[filename], label=filename if i == 0 else "")
            '''
            ## Now we don't use this fitting, because it covers the bar, we can't see the difference between bars
            # 使用scipy.stats.gaussian_kde拟合数据
            kde = gaussian_kde(df_file['firing_rate'])
            x_fit = np.linspace(min(bin_centers), max(bin_centers), 100)
            y_fit = kde(x_fit) * (bins[1] - bins[0])  # 确保概率密度与直方图一致
            
            # 在3D图上绘制拟合曲线
            ax.plot(x_fit, y_fit, zs=i, zdir='y', color=color_dict[filename], lw=2)

            ## or 使用scipy.stats.expon拟合数据
            params = expon.fit(df_file['firing_rate'])
            x_fit = np.linspace(min(bin_centers), max(bin_centers), 100)
            y_fit = expon.pdf(x_fit, *params) * len(df_file['firing_rate']) * (bins[1] - bins[0])
            
            # 在3D图上绘制拟合曲线
            ax.plot(x_fit, y_fit, zs=i, zdir='y', color=color_dict[filename], lw=2)
            '''

    # 设置轴标签
    ax.set_xlabel('Firing rate', fontsize=14)
    ax.set_ylabel('Brain subregion', fontsize=14,labelpad=45)
    ax.set_zlabel('Probability', fontsize=14)

    # 增大轴范围
    ax.set_xlim([min(df['firing_rate']), max(df['firing_rate'])])
    ax.set_ylim([-0.5, len(regions) - 0.5])  # Adjust as needed
    ax.set_zlim([0, max(hist) * 1.1])  # Adjust as needed

    # 设置 y 轴刻度标签为 region 名称
    ax.set_yticks(range(len(regions)))
    ax.set_yticklabels(regions,rotation=0, ha='left', va='center', fontsize=7)

    # 强制显示图例中所有的 filename 标签
    handles, labels = [], []
    for filename in filenames:
        handles.append(plt.Line2D([0], [0], color=color_dict[filename], lw=4))
        labels.append(filename)
    ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

combined_df = conctenate()
twoD_slice_plotcompare(combined_df)
#thrD_plotcompare(combined_df)