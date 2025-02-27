"""
# coding: utf-8
@author: Yuhao Zhang
last updated: 08/06/2024
data from: Xinchao Chen
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

# 指定包含 CSV 文件的文件夹路径
data_folder = r'C:\Users\zyh20\Desktop\ET_data analysis\manifold\subspace'

# 获取文件夹中所有文件的列表
file_list = os.listdir(data_folder)

# 遍历每个文件
for file_name in file_list:
    if file_name.endswith('.csv'):  # 假设文件是CSV格式的
        file_path = os.path.join(data_folder, file_name)
        
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 获取第一列作为x，第四列作为y
        x = df.iloc[:, 0]  # 第一列
        y = df.iloc[:, 2]  # 第四列
        
        # 绘制柱状图
        plt.figure()
        # 设置横轴文字竖直显示
        plt.xticks(rotation='vertical')
        plt.bar(x, y)
        plt.ylabel('null space dimension')
        plt.title(f'{file_name[:-4]}')
        
        # 保存图像到当前文件夹
        plt.savefig(data_folder+f'/null space dimension_{file_name[:-4]}.png',dpi=600,bbox_inches = 'tight')  # 去掉.csv扩展名，并保存为PNG格式