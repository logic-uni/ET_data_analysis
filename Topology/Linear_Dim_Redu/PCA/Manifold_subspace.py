"""
# coding: utf-8
@author: Yuhao Zhang
last updated: 06/02/2025
data from: Xinchao Chen
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import neo
import quantities as pq
from matplotlib.animation import FuncAnimation
from sklearn.metrics import pairwise_distances
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from matplotlib import cm
from scipy import interpolate
from scipy.interpolate import interp1d
from elephant.conversion import BinnedSpikeTrain
np.set_printoptions(threshold=np.inf)

# ------- NEED CHANGE -------
data_path = '/data2/zhangyuhao/xinchao_data/Givenme/1423_15_control-Day1-1CVC-FM_g0'
save_path = '/home/zhangyuhao/Desktop/Result/ET/Manifold/NP2/givenme/1423_15_control-Day1-1CVC-FM_g0'
# ------- NO NEED CHANGE -------
fr_bin = 1
### Behavior
Marker = pd.read_csv(data_path+'/Behavior/marker.csv') 
print(Marker)

### Electrophysiology
fs = 30000  # spikeGLX neuropixel sample rate
identities = np.load(data_path+'/Sorted/kilosort4/spike_clusters.npy') # time series: unit id of each spike
times = np.load(data_path+'/Sorted/kilosort4/spike_times.npy')  # time series: spike time of each spike
neurons = pd.read_csv(data_path+'/Sorted/kilosort4/mapping_artifi.csv')
print(neurons)
# 按region分组，提取每组的第一列cluster_id
region_groups = neurons.groupby('region')
region_cluster_ids = {}
for region, group in region_groups:
    # 提取每组的第一列（cluster_id），去除缺失值
    cluster_ids = group.iloc[:, 0].dropna().astype(int).values
    region_cluster_ids[region] = cluster_ids

print("Test if Ephys duration same as motion duration...")
print(f"Ephys duration: {(times[-1]/fs)} s")  # for NP1, there's [0] after times[-1]/fs
print(f"motion duration: {Marker['time_interval_right_end'].iloc[-1]} s")
neuron_num = neurons.count().transpose().values

def singleneuron_spiketimes(id):
    x = np.where(identities == id)
    y=x[0]
    #y = np.where(np.isin(identities, id))[0]
    spike_times=np.empty(len(y))
    for i in range(0,len(y)):
        z=y[i]
        spike_times[i]=times[z]/fs
    return spike_times

def popu_fr_onetrial(neuron_ids,marker_start,marker_end):   
    for j in range(len(neuron_ids)): #第j个neuron
        spike_times = singleneuron_spiketimes(neuron_ids[j])
        spike_times_trail = spike_times[(spike_times > marker_start) & (spike_times < marker_end)]
        spiketrain = neo.SpikeTrain(spike_times_trail,units='sec',t_start=marker_start, t_stop=marker_end)
        fr = BinnedSpikeTrain(spiketrain, bin_size=fr_bin*pq.ms,tolerance=None)
        one_neruon = fr.to_array().astype(int)[0]
        if j == 0:
            neurons = one_neruon
        else:
            neurons = np.vstack((neurons, one_neruon))
    return neurons

def reduce_dimension(count,bin_size,region_name,n_components): # 默认: 0.1 感觉改bin_size影响不大，改firing rate的bin size影响较大
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

def reduce_dimension(count,bin_size,region_name,stage): # 默认: 0.1 感觉改bin_size影响不大，改firing rate的bin size影响较大
    #smooth data
    count = pd.DataFrame(count)
    rate = np.sqrt(count/bin_size)
    #对数据做均值  默认: window=50  min_periods=1  感觉改这些值影响不大，改firing的bin size影响较大
    rate = rate.rolling(window=50,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=2) 
    #reduce dimension
    ## PCA
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(rate.values)   #对应的是Explained variance
    explained_variance_ratio = pca.explained_variance_ratio_   #每个主成分所解释的方差比例
    explained_variance_sum = np.cumsum(explained_variance_ratio)  #计算累积解释方差比例
    #画explained_variance图
    x=list(range(len(explained_variance_ratio)))
    fig = plt.figure()
    plt.plot(x,explained_variance_ratio, color='blue', label='each PC ratio')
    plt.plot(x,explained_variance_sum, color='red', label='ratio sum')
    plt.title(f"{region_name}_{stage}_PC_explained variance ratio")
    plt.xlabel('PC')
    plt.ylabel('Value')
    plt.legend()
    #plt.show()
    plt.savefig(save_path+f"/{region_name}_{stage}_PC_explained_var_ratio.png",dpi=600,bbox_inches = 'tight')
    X_isomap = Isomap(n_components = 3, n_neighbors = 21).fit_transform(rate.values)  #对应的是Residual variance
    #X_tsne = TSNE(n_components=3,random_state=21,perplexity=20).fit_transform(rate.values)  #t-SNE没有Explained variance，t-SNE 旨在保留局部结构而不是全局方差
    return X_isomap

def reduce_dimension_ISOMAP(count,bin_size,region_name,stage): # 默认: 0.1 感觉改bin_size影响不大，改firing rate的bin size影响较大
    #smooth data
    count = pd.DataFrame(count)
    rate = np.sqrt(count/bin_size)
    #对数据做均值  默认: window=50  min_periods=1  感觉改这些值影响不大，改firing的bin size影响较大
    rate = rate.rolling(window=50,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=2) 
    #reduce dimension
    X_isomap = Isomap(n_components = 3, n_neighbors = 21).fit_transform(rate.values)  #对应的是Residual variance
    X=rate.values
    D_high = pairwise_distances(X, metric='euclidean')
    residual_variances = []
    # Calculate residual variance for different embedding dimensions
    for n_components in range(1, 6):
        isomap = Isomap(n_neighbors=5, n_components=n_components)
        X_low = isomap.fit_transform(X)
        D_low = pairwise_distances(X_low, metric='euclidean')
        residual_variance = 1 - np.sum((D_high - D_low) ** 2) / np.sum(D_high ** 2)
        residual_variances.append(residual_variance)
    fig = plt.figure()
    # Plot residual variance
    plt.plot(range(1, 6), residual_variances, marker='o')
    plt.xlabel('Number of Dimensions')
    plt.ylabel('Residual Variance')
    plt.title(f"{region_name}_{stage}_Isomap Residual Variance")
    #plt.show()
    plt.savefig(save_path+f"/{region_name}_{stage}_Isomap_Residual_Var.png",dpi=600,bbox_inches = 'tight')

    return X_isomap

def plot_hyper_plane(X, y,X_pca):
    # 使用线性回归拟合超平面
    model = LinearRegression()
    model.fit(X, y)
    # 超平面系数
    w = model.coef_
    b = model.intercept_
    # 计算超平面在PCA降维后的空间中的斜率和截距
    slope = -w[0] / w[1]
    intercept = -b / w[1]
    # 绘制数据点和超平面在PCA降维后的空间中
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolors='k')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    # 绘制超平面
    x_vals = np.array(plt.gca().get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--', color='red', label='Hyperplane')
    plt.legend()
    plt.title('Hyperplane in PCA-reduced Space')
    plt.savefig(save_path+f"/Hyperplane.png",dpi=600,bbox_inches = 'tight')

def interp_helper(values, num=50, kind='quadratic'):
    interp_i = np.linspace(min(values), max(values), num)
    return interp1d(np.linspace(min(values), max(values), len(values)), values, kind=kind)(interp_i)

def plot_normal_vector(normal_vector):
    #plot normal vector
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    a=[0,0,0]
    a_count=0
    b=[0,0,0]
    b_count=0
    for i in range(0,len(normal_vector)):
        if i ==0 or i ==5 or i ==10 or i ==11 or i ==18 or i ==19 or 24 <= i <= 26 or 33 <= i <= 34 or 38 <= i <= 40 or 44 <= i <= 46:
            ax.quiver(0,0,0,normal_vector[i,0],normal_vector[i,1],normal_vector[i,2],arrow_length_ratio=0.1,color='r',length=5, normalize=True)
            a=a+normal_vector[i]
            a_count=a_count+1
        if 1 <= i <= 4 or 6 <= i <= 9 or 12 <= i <= 17 or 20 <= i <= 23 or 27 <= i <= 32 or 35 <= i <= 37 or 41 <= i <= 43:
            ax.quiver(0,0,0,normal_vector[i,0],normal_vector[i,1],normal_vector[i,2],arrow_length_ratio=0.1,color='b',length=5, normalize=True)
            b=b+normal_vector[i]
            b_count=b_count+1
    # compute average motion vectors and average rest vectors
    a_av=a/a_count
    b_av=b/b_count
    ax.quiver(0,0,0,a_av[0],a_av[1],a_av[2],arrow_length_ratio=0.1,color='g',length=8, normalize=True)
    ax.quiver(0,0,0,b_av[0],b_av[1],b_av[2],arrow_length_ratio=0.1,color='y',length=8, normalize=True)
    #compute included angle
    cos_angle = np.dot(a_av, b_av) / (np.linalg.norm(a_av) * np.linalg.norm(b_av))
    angle = np.arccos(cos_angle)
    print('夹角为：', angle * 180 / np.pi, '度')
    ax.set_xlim(-8.05,8.05)
    ax.set_ylim(-8.05,8.05)
    ax.set_zlim(-8.05,8.05)
    plt.show()

def manifold_fitplane(X_isomap):
    for i in range(0,len(X_isomap)):
        ax.scatter(x_track[:,0], x_track[:,1], x_track[:,2], 'blue')
        x_track_s=[X_isomap[i,0],X_isomap[i,1],X_isomap[i,2]]
        x_track = np.vstack((x_track, x_track_s))
        plt.pause(0.01)
    #plot fixed 3D trajectory
    fig = plt.figure()   
    ax = fig.gca(projection='3d')
    ax.scatter(X_isomap[:,0],X_isomap[:,1],X_isomap[:,2],label='Essential Tremor Neural Manifold')  #分别取三列的值作为x,y,z的值
    ax.legend()
    plt.savefig(save_path+f"/fixed3D.png",dpi=600,bbox_inches = 'tight')
    # plot 3D colored smooth trajectory
    x_new, y_new, z_new = (interp_helper(i,800) for i in (X_isomap[:,0], X_isomap[:,1], X_isomap[:,2]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    zmax = np.array(z_new).max()
    zmin = np.array(z_new).min()
    for i in range(len(z_new) - 1):
        ax.plot(x_new[i:i + 2], y_new[i:i + 2], z_new[i:i + 2],
                color=plt.cm.jet(int((np.array(z_new[i:i + 2]).mean() - zmin) * 255 / (zmax - zmin))))
    plt.title('Essential Tremor Neural Manifold')
    plt.savefig(save_path+f"/3Dcolored.png",dpi=600,bbox_inches = 'tight')
    
    #fit plane
    v1=fit_plane(X_isomap[0:215,0],X_isomap[0:215,1],X_isomap[0:215,2],'r')
    #normal vector
    #vv1 = np.vstack((v1, v2))
    #print(vv14)
    # normlize x to same symbol
    for i in range(0,len(v)):
        if v[i,2] > 0:
            v[i,0]=(-1)*v[i,0]
            v[i,1]=(-1)*v[i,1]
            v[i,2]=(-1)*v[i,2]
    print(v)
    # amplify
    v=v*10
    print(v)
    plot_normal_vector(v)
    plt.savefig(save_path+f"/vector.png",dpi=600,bbox_inches = 'tight')
    return X_isomap

def plot_surface_2(x,y,z):
    f = interpolate.interp2d(x, y, z, kind='cubic')
    znew = f(x, y)
    #修改x,y，z输入画图函数前的shape
    xx1, yy1 = np.meshgrid(x, y)
    newshape = (xx1.shape[0])*(xx1.shape[0])
    y_input = xx1.reshape(newshape)
    x_input = yy1.reshape(newshape)
    z_input = znew.reshape(newshape)
    #画图
    sns.set(style='white')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(x_input,y_input,z_input,cmap=cm.coolwarm)
    plt.savefig(save_path+f"/surface.png",dpi=600,bbox_inches = 'tight')

def fit_plane(xs,ys,zs,color_name):
    ax = plt.subplot(projection = '3d')
    # do fit
    tmp_A = [] #存储x和y
    tmp_b = [] #存储z
    for i in range(len(xs)):
        tmp_A.append([xs[i], ys[i], 1])
        tmp_b.append(zs[i])
    b = np.matrix(tmp_b).T
    A = np.matrix(tmp_A)

    # Manual solution
    fit = (A.T * A).I * A.T * b  #该式由最小二乘法推导得出
    errors = b - A * fit #计算估计值与真实值的误差
    residual = np.linalg.norm(errors)  #求误差矩阵的范数，即残差平方和SSE
    error_withmean = np.mean(b)- A * fit #计算估计值与平均值的误差
    regression = np.linalg.norm(error_withmean)  #求回归误差矩阵的范数，即回归平方和SSR
    SST=residual+regression
    R2=1-(residual/SST)

    print("solution: %f x + %f y + %f = z" % (fit[0], fit[1], fit[2]))
    print("residual:" ,residual)
    print("R2:" ,R2)

    # plot plane
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    X,Y = np.meshgrid(np.arange(xlim[0], xlim[1]),
                    np.arange(ylim[0], ylim[1]))
    Z = np.zeros(X.shape)
    for r in range(X.shape[0]):
        for c in range(X.shape[1]):
            Z[r,c] = fit[0] * X[r,c] + fit[1] * Y[r,c] + fit[2]
    ax.plot_surface(X,Y,Z, color=color_name,alpha=0.5)
    normal_v=np.array([fit[0,0], fit[1,0], fit[2,0]])
    return normal_v

def interval_cuttage(marker):
    run_during=np.array([])
    stop_during=np.array([])
    run_time_dura=np.empty((0, 2)).astype(int) 
    stop_time_dura=np.empty((0, 2)).astype(int) 
    run=marker[marker['run_or_stop'] == 1]
    stop=marker[marker['run_or_stop'] == 0]
    for i in range(0,len(marker['run_or_stop'])):
        start=int(marker['time_interval_left_end'].iloc[i])
        end=int(marker['time_interval_right_end'].iloc[i])
        if marker['run_or_stop'].iloc[i] == 1:
            #由于treadmill运动和静止交替的持续时间随机，因此检测持续时间的最小长度，作为一个trail的长度，各个持续时间如果大于最小区间的X倍，按照X个trial计入
            run_during=np.append(run_during,end-start)  #获得所有运动区间的持续时间长度
        else:
            stop_during=np.append(stop_during,end-start) #获得所有静止区间的持续时间长度
    min_run=np.min(run_during)      #获得运动/静止 最小区间的时间长度
    min_stop=np.min(stop_during)
    run_multiple=np.floor(run_during/min_run).astype(int)      #获得每个时间区间可以被划分为最小时间区间的几倍
    stop_multiple=np.floor(stop_during/min_stop).astype(int)
    #获取所有以最小运动时间长度为基准的运动区间
    for j in range(0,len(run_multiple)):
        if run_multiple[j] != 1:
            for n in range(1,run_multiple[j]+1):  
                left=int(run['time_interval_left_end'].iloc[j])+min_run*(n-1)
                right=left+min_run
                time_dura=[int(left),int(right)]
                run_time_dura=np.vstack([run_time_dura, time_dura])
        else:
            left=int(run['time_interval_left_end'].iloc[j])
            right=left+min_run
            time_dura=[int(left),int(right)]
            run_time_dura=np.vstack([run_time_dura, time_dura])
    #获取所有以最小静止时间长度为基准的静止区间
    for k in range(0,len(stop_multiple)):
        if stop_multiple[k] != 1:
            for m in range(1,stop_multiple[k]+1):  
                left=int(stop['time_interval_left_end'].iloc[k])+min_stop*(m-1)
                right=left+min_stop
                time_dura=[int(left),int(right)]
                stop_time_dura=np.vstack([stop_time_dura, time_dura])
        else:
            left=int(stop['time_interval_left_end'].iloc[k])
            right=left+min_stop
            time_dura=[int(left),int(right)]
            stop_time_dura=np.vstack([stop_time_dura, time_dura])

    return run_time_dura,stop_time_dura

def trail_average(data,run_time_dura,stop_time_dura):
    ## run
    #run is a matrix with trials * neurons * timepoint, each value is the firing rate in this time point
    run = np.zeros((run_time_dura.shape[0], data.shape[0], run_time_dura[0][1]-run_time_dura[0][0]))
    for ti in range(0,run_time_dura.shape[0]):
        neuron_runpiece = data[:, run_time_dura[ti][0]:run_time_dura[ti][1]]   #firing rate * neurons矩阵，按照区间切片
        if neuron_runpiece.shape == run[ti, :, :].shape:
            run[ti, :, :] = neuron_runpiece
    # 三维run矩阵沿着第一个维度，对应相加求平均
    run_average=np.mean(run, axis=0)

    ## stop
    stop = np.zeros((stop_time_dura.shape[0], data.shape[0], stop_time_dura[0][1]-stop_time_dura[0][0]))
    for ti_stop in range(0,stop_time_dura.shape[0]):
        neuron_stoppiece = data[:, stop_time_dura[ti_stop][0]:stop_time_dura[ti_stop][1]]   #firing rate * neurons矩阵，按照区间切片
        if neuron_stoppiece.shape == stop[ti_stop-1, :, :].shape:
            stop[ti_stop, :, :] = neuron_stoppiece
    # 三维stop矩阵沿着第一个维度，对应相加求平均
    stop_average=np.mean(stop, axis=0)
    
    return run_average,stop_average

def normalize_fr(data2dis):
    '''
    #标准化方法1 z-score 会出现负值, PCA不适应报错
    # 计算每行的均值和标准差
    means = np.mean(data2dis, axis=1, keepdims=True)
    stds = np.std(data2dis, axis=1, keepdims=True)

    # 计算z-score
    z_scores = (data2dis - means) / stds
    '''
    #标准化方法2 标准化到0-1
    normalized_data = (data2dis - data2dis.min(axis=1, keepdims=True)) / (data2dis.max(axis=1, keepdims=True) - data2dis.min(axis=1, keepdims=True))
    return normalized_data

def main(neurons,marker):
    for region_name, neuron_id in region_cluster_ids.items():  # 遍历所有的脑区及其对应的neuron id
        print(f"Region: {region_name} ")
        print(f"Neuron IDs: {neuron_id}")
        marker_start = marker['time_interval_left_end'].iloc[0]
        marker_end = marker['time_interval_right_end'].iloc[-1]

        data,time_len = popu_fr_onetrial(neuron_id,marker_start,marker_end)
        data_norm=normalize_fr(data)
        data2pca=data_norm.T
        '''
        ### manifold surface
        data2pca=data.T
        redu_dim_data=reduce_dimension(data2pca,0.1,region_name,stage='all_session')
        plot_surface_2(redu_dim_data[:,0],redu_dim_data[:,1],redu_dim_data[:,2])
        '''
        ### manifold each trail
        data2pca_each_trail=data.T
        redu_dim_data=reduce_dimension(data2pca_each_trail,0.1,region_name,stage='all_session')
        # ISOMAP extract nolinear structure
        redu_dim_data_ISOMAP=reduce_dimension_ISOMAP(data2pca_each_trail,0.1,region_name,stage='all_session')
    
        ### manifold trial average
        run_time_dura,stop_time_dura=interval_cuttage(marker)
        run_average,stop_average=trail_average(data,run_time_dura,stop_time_dura)
        #print(run_average.shape)
        #print(stop_average.shape)

        run2pca=run_average.T
        run_redu_dim_aver=reduce_dimension(run2pca,0.1,region_name,stage='Run')
        manifold_fixed(run_redu_dim_aver,'Run',region_name)
        #manifold_dynamic(run_redu_dim_aver,'Run')

        stop2pca=stop_average.T
        stop_redu_dim_aver=reduce_dimension(stop2pca,0.1,region_name,stage='Stop')
        manifold_fixed(stop_redu_dim_aver,'Stop',region_name)
        #manifold_dynamic(stop_redu_dim_aver,'Stop')        

main(neurons,Marker)