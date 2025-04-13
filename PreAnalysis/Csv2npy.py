import pandas as pd
import numpy as np

# 读取CSV文件逐块处理
chunk_size = 1000  # 根据内存情况调整块大小
LFP_path = r'E:\chaoge\sorted neuropixels data\20230523-condictional tremor1\20230523\raw\20230523_Syt2_449_1_Day50_g0\20230523_Syt2_449_1_Day50_g0_imec0'
npy_file = 'LFP.npy'

chunk_list = []
for chunk in pd.read_csv(LFP_path+'/20230523_Syt2_449_1_Day50_g0_t0.exported.imec0.lf.csv', chunksize=chunk_size):
    chunk_list.append(chunk.to_numpy())

# 将所有块合并为一个Numpy数组
np_array = np.vstack(chunk_list)

# 保存为npy文件
np.save(LFP_path+npy_file, np_array)
