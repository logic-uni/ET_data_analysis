"""
# coding: utf-8
@author: Yuhao Zhang
last updated: 03/27/2025
data from: Xinchao Chen
"""
import neo
import quantities as pq
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.pyplot import *
from ast import literal_eval
from sklearn.decomposition import PCA
from elephant.conversion import BinnedSpikeTrain
np.set_printoptions(threshold=np.inf)
np.seterr(divide='ignore',invalid='ignore')


# Lobule
#20230113_littermate    Lobules IV-V
#20230523_Syt2_conditional_tremor_mice1    Lobule III  Lobule II
#20230604_Syt2_conditional_tremor_mice2_medial   Lobule III
#20230602_Syt2_conditional_tremor_mice2_lateral  Lobule III
#20230623_Syt2_conditional_tremor_mice4  Lobule III  Lobule II
#  DCN
#20230604_Syt2_conditional_tremor_mice2_medial  xinchaosort  Vestibulocerebellar nucleus
#20230113_littermate   Interposed nucleus


region_name = 'Lobule III'
fr_bin = 1  # unit: ms
avoid_spikemore1 = True # 避免1ms的bin里有多个spike,对于1ms内多个spike的，强行置1

## ------ Easysort ------
# --- NEED CHANGE ---
mice_name = '20230602_Syt2_conditional_tremor_mice2_lateral'
mapping_file = 'unit_ch_dep_region_QC_isi_violations_ratio_pass_rate_60.17316017316017%.csv'
QC_method = 'Without_QC'  # Without_QC/QC_ISI_violation/etc
# --- NO NEED CHANGE ---
sorting_path = rf'E:\xinchao\Data\useful_data\NP1\{mice_name}\Sorted\Easysort\results_KS2\sorter_output'
save_path = rf'C:\Users\zyh20\Desktop\Research\01_ET_data_analysis\Research\spectrum_analysis\NP1\Easysort\{QC_method}\{mice_name}'  
neurons = pd.read_csv(rf'E:\xinchao\Data\useful_data\NP1\{mice_name}\Sorted\Easysort\mapping\{mapping_file}')  # different sorting have different nueron id

# ------ Xinchao_sort ------
# --- NEED CHANGE ---
#mice_name = '20230602_Syt2_conditional_tremor_mice2_lateral'
# --- NO NEED CHANGE ---
#sorting_path = rf'E:\xinchao\Data\useful_data\NP1\{mice_name}\Sorted\Xinchao_sort'
#save_path = rf'C:\Users\zyh20\Desktop\Research\01_ET_data_analysis\Research\spectrum_analysis\NP1\Xinchao_sort\{mice_name}'  
#neurons = pd.read_csv(sorting_path + '/neuron_id_region_firingrate.csv')  # different sorting have different nueron id

# --------- Main Program ----------
treadmill = pd.read_csv(rf'E:\xinchao\Data\useful_data\NP1\{mice_name}\Marker\treadmill_move_stop_velocity_segm_trial.csv',index_col=0)
treadmill_origin = pd.read_csv(rf'E:\xinchao\Data\useful_data\NP1\{mice_name}\Marker\treadmill_move_stop_velocity.csv',index_col=0)
# electrophysiology
sample_rate = 30000 #spikeGLX neuropixel sample rate
identities = np.load(sorting_path + '/spike_clusters.npy') # time series: unit id of each spike
times = np.load(sorting_path + '/spike_times.npy')  # time series: spike time of each spike
print(neurons)
print("Test if electrophysiology duration is equal to treadmill duration ...")
elec_dura = (times[-1]/sample_rate)[0]
treadmill_dura = treadmill_origin['time_interval_right_end'].iloc[-1]
print(f"Electrophysiology duration: {elec_dura}")
print(f"Treadmill duration: {treadmill_dura}")

# get single neuron spike train
def singleneuron_spiketimes(id):
    x = np.where(identities == id)
    y=x[0]
    #y = np.where(np.isin(identities, id))[0]
    spike_times=np.zeros(len(y))
    for i in range(0,len(y)):
        z=y[i]
        spike_times[i]=times[z]/sample_rate
    return spike_times

def popu_sptrain_trial(neuron_ids,marker_start,marker_end):
    for j in range(len(neuron_ids)): #第j个neuron
        spike_times = singleneuron_spiketimes(neuron_ids[j])
        spike_times_trail = spike_times[(spike_times > marker_start) & (spike_times < marker_end)]
        spiketrain = neo.SpikeTrain(spike_times_trail,units='sec',t_start=marker_start, t_stop=marker_end)
        fr = BinnedSpikeTrain(spiketrain, bin_size=fr_bin*pq.ms,tolerance=None)
        if avoid_spikemore1 == False:
            one_neuron = fr.to_array().astype(int)[0]
        else:
            fr_binar = fr.binarize()  # 对于可能在1ms内出现两个spike的情况，强制置为该bin下即1ms只能有一个spike
            one_neuron = fr_binar.to_array().astype(int)[0]
        
        if j == 0:
            neurons = one_neuron
        else:
            neurons = np.vstack((neurons, one_neuron))
    return neurons

def reduce_dimension(count,bin_size,n_components): # 默认: 0.1 感觉改bin_size影响不大，改firing rate的bin size影响较大
    #smooth data
    count = pd.DataFrame(count)
    rate = np.sqrt(count/bin_size)
    #对数据做均值  默认: window=50  min_periods=1  感觉改这些值影响不大，改firing的bin size影响较大
    rate = rate.rolling(window=50,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=2) 
    #reduce dimension
    pca = PCA(n_components)
    X_pca = pca.fit_transform(rate.values)   #对应的是Explained variance
    explained_variance_ratio = pca.explained_variance_ratio_   #每个主成分所解释的方差比例
    explained_variance_sum = np.cumsum(explained_variance_ratio)  #计算累积解释方差比例
    #X_isomap = Isomap(n_components = 3, n_neighbors = 21).fit_transform(rate.values)  #对应的是Residual variance
    #X_tsne = TSNE(n_components=3,random_state=21,perplexity=20).fit_transform(rate.values)  #t-SNE没有Explained variance，t-SNE 旨在保留局部结构而不是全局方差
    return X_pca

def SVD_matrix_spectrum(data):
    # 1. 进行奇异值分解
    U, S, Vt = np.linalg.svd(data)
    # 2. 绘制奇异值的分布
    plt.figure(figsize=(8, 5))
    plt.plot(S, marker='o', linestyle='-', color='b')
    plt.title('Singular Values from SVD')
    plt.xlabel('Index')
    plt.ylabel('Singular Value')
    plt.grid(True)
    plt.show()
    # 3. 计算最大奇异值对应的特征向量
    max_singular_value_index = np.argmax(S)
    max_singular_value_vector = U[:, max_singular_value_index]
    # 输出最大奇异值及其对应的特征向量
    print(f'Max Singular Value: {S[max_singular_value_index]}')
    print(f'Corresponding Feature Vector (Left Singular Vector): {max_singular_value_vector}')

def redu_eign(data):
    print(data.shape)
    data2pca=data.T
    redu_dim_data=reduce_dimension(data2pca,0.1,30)
    ## 这里的矩阵是人为确定过保证Neuron和time两个轴的维度相同，对这样截断的矩阵做谱分析，因此存在虚数
    print(redu_dim_data.shape)
    # 对非对称矩阵进行特征值分解
    eigenvalues, eigenvectors = np.linalg.eig(redu_dim_data)
    return eigenvalues

def covmatr_spectrum(data):
    ## covarience matrix是对称阵，所以只有实数值的特征值
    ATA = np.dot(data.T, data)
    # 对 A^T A 进行特征值分解
    eigenvalues, eigenvectors = np.linalg.eig(ATA)
    plt.figure(figsize=(8, 6))
    plt.hist(eigenvalues, bins=30, color='blue', edgecolor='black', alpha=0.7)
    plt.title('Histogram of Eigenvalues of $A^T A$')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    # 存储特征值最大的特征向量
    max_index = np.argmax(eigenvalues)  # 找到最大特征值的索引
    max_eigenvalue = eigenvalues[max_index]  # 最大特征值
    max_eigenvector = eigenvectors[:, max_index]  # 对应的特征向量
    return max_eigenvector

def eigen_vector_included_angle(A,B):
    # 将矩阵展平为一维数组
    A_flat = A.flatten()
    B_flat = B.flatten()
    # 计算点积和模长
    dot_product = np.dot(A_flat, B_flat)
    norm_A = np.linalg.norm(A_flat)
    norm_B = np.linalg.norm(B_flat)
    # 计算余弦相似度
    similarity = dot_product / (norm_A * norm_B)
    print(similarity)

def spectrum_analysis(region,popu_ids):
    neuron_num = len(popu_ids)
    squmat_interval = neuron_num  #单位ms
    trial_spiketr_len = 30000/fr_bin
    truncated_trial = int(trial_spiketr_len - (trial_spiketr_len % squmat_interval))   # 一个trial切分成多个小正方形矩阵，余下的不要
    for index, row in treadmill.iterrows():
        trial_start = row['time_interval_left_end']
        trial_end = row['time_interval_right_end']
        status = row['run_or_stop']
        if status == 0:
            color = 'b'
        else:
            color = 'r'
        data = popu_sptrain_trial(popu_ids,trial_start,trial_end)
        print(data.shape)
        for squleft in range(0,squmat_interval,truncated_trial):
            square_data = data[:,squleft:squleft+squmat_interval]
            print(square_data.shape)
            eigenvalues, eigenvectors = np.linalg.eig(square_data)
            print(np.abs(eigenvalues))
            plt.scatter(eigenvalues.real, eigenvalues.imag, c=color, marker='o')
    
    plt.title(f'spectrum_analysis_{region}')
    plt.text(0.05, 0.95, f'Each Truncated Time Length: {squmat_interval} ms', 
                    ha='left', va='top', transform=plt.gca() .transAxes, fontsize=10)
    plt.savefig(save_path+f"/{region}.png",dpi=600,bbox_inches = 'tight')
    plt.clf()

def main():
    # 谱分析画布固定内容
    plt.figure(figsize=(10, 10))
    plt.xlabel('Real')
    plt.xlim(-5,5)
    plt.ylim(-5,5)
    plt.ylabel('Imaginary')
    plt.grid(True)
    plt.gca().set_aspect('equal')
    theta = np.linspace(0, 2 * np.pi, 100)
    x = 1 * np.cos(theta)
    y = 1 * np.sin(theta)
    plt.plot(x, y,c='black')

    result = neurons.groupby('region')['cluster_id'].apply(list).reset_index(name='cluster_ids')
    print(result)
    for index, row in result.iterrows():
        region = row['region']
        popu_ids = row['cluster_ids']
        if region == region_name:
            spectrum_analysis(region,popu_ids)

    #max_eigen1 = covmatr_spectrum(data[:,(pert_s-40):(pert_s)])
    #max_eigen2 = covmatr_spectrum(data[:,(pert_s):(pert_s+40)])
    #eigen_vector_included_angle(max_eigen1,max_eigen2)
    #covmatr_spectrum(data[:,(pert_s-40):(pert_s)])
    #covmatr_spectrum(data[:,(pert_s-40):(pert_e+150)])
    #SVD_matrix_spectrum(data[:,(pert_s):(pert_s+40)])

main()
