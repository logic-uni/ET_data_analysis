"""
# coding: utf-8
@author: Yuhao Zhang
last updated: 03/04/2025
data from: Xinchao Chen
"""

import pandas as pd
mice_name = '20230112_PVsyt2_tremor'
treadmill = pd.read_csv(rf'E:\xinchao\Data\useful_data\{mice_name}\Marker\treadmill_move_stop_velocity.csv',index_col=0)
save_path = rf'E:\xinchao\Data\useful_data\{mice_name}\Marker'  

df = pd.DataFrame(treadmill)

new_rows = []

for index, row in df.iterrows():
    left = row['time_interval_left_end']
    right = row['time_interval_right_end']
    delta = right - left
    print(delta)
    if delta < 29.9:
        continue

    if 29.9 < delta < 30:
        delta = 30

    n = int(delta // 30)
    
    for i in range(n):
        new_left = left + i * 30
        new_right = new_left + 30
        new_row = row.copy()
        new_row['time_interval_left_end'] = new_left
        new_row['time_interval_right_end'] = new_right
        new_rows.append(new_row)

new_df = pd.DataFrame(new_rows).reset_index(drop=True)

# 按原始列顺序重新排列（如果需要）
column_order = ['time_interval_left_end', 'time_interval_right_end', 'run_or_stop',
                'velocity_recording', 'velocity_level', 'velocity_theoretical']
new_df = new_df[column_order]

print(new_df)
new_df.to_csv(save_path+f'/treadmill_move_stop_velocity_segm_trial.csv')
