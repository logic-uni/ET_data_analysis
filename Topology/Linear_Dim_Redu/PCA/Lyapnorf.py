from matplotlib.animation import FuncAnimation
def adjust_array(arr):
    if any(x < 0 for x in arr):
        min_val = min(arr)
        diff = -min_val
        arr = [x + diff for x in arr]
    return np.array(arr)

def oneDdynamic(count,bin_size): # 默认: 0.1 感觉改bin_size影响不大，改firing rate的bin size影响较大
    #smooth data
    count = pd.DataFrame(count)
    rate = np.sqrt(count/bin_size)
    #对数据做均值  默认: window=50  min_periods=1  感觉改这些值影响不大，改firing的bin size影响较大
    rate = rate.rolling(window = 50, win_type = 'gaussian', center = True, min_periods = 1).mean(std=2)  # axis 无需输入默认0
    #reduce dimension
    '''
    ## PCA
    pca = PCA(n_components=1)
    X_pca = pca.fit_transform(rate.values)   #对应的是Explained variance
    explained_variance_ratio = pca.explained_variance_ratio_   #每个主成分所解释的方差比例
    explained_variance_sum = np.cumsum(explained_variance_ratio)  #计算累积解释方差比例
    #画explained_variance图
    x=list(range(len(explained_variance_ratio)))
    '''
    fig = plt.figure()
    X_isomap = Isomap(n_components = 1, n_neighbors = 21).fit_transform(rate.values)  #对应的是Residual variance

    ### 原始降维到一维后的值随时间变化
    # 小球距离曲线的偏移量
    offset = 7
    array = np.transpose(X_isomap)[0]
    adjusted_array = adjust_array(array)
    # 初始化动画
    y = adjusted_array
    x = np.arange(0,len(y))
    fig, ax = plt.subplots()
    line, = ax.plot(x, y, linestyle='-', color='b')
    ball, = ax.plot([], [], marker='o', markersize=7, color='r')
    # 设置图形界面属性
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(y) - 1, np.max(y) + 1)
    ax.set_title(f"{region}_dynamic_1D")
    ax.set_xlabel("time")
    ax.set_ylabel("neural state")
    ax.grid(True)
    # 更新函数，用于每一帧的更新
    def update(frame):
        ball_x = x[frame]
        ball_y = y[frame] + offset  # 小球在曲线上方的位置偏移
        ball.set_data([ball_x],[ball_y])
        return line, ball,
    # 创建动画
    ani = FuncAnimation(fig, update, frames=len(x), interval=20, blit=True)
    # 使用imagemagick将动画保存为GIF图片
    ani.save(save_path+f"/1Ddynamic_{region}_dynamic_raw.gif", writer='pillow')

    ### y=x^2 李雅普诺夫能量函数
    # 小球距离曲线的偏移量
    offset = 7
    array = np.transpose(X_isomap)[0]
    adjusted_array = adjust_array(array)
    # 初始化动画
    x = adjusted_array
    y = x*x
    fig, ax = plt.subplots()
    line, = ax.plot(x, y, linestyle='-', color='b')
    ball, = ax.plot([], [], marker='o', markersize=7, color='r')
    # 设置图形界面属性
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(y) - 1, np.max(y) + 1)
    ax.set_title(f"{region}_dynamic_1D")
    ax.set_xlabel("time")
    ax.set_ylabel("neural energy")
    ax.grid(True)
    # 更新函数，用于每一帧的更新
    def update(frame):
        ball_x = x[frame]
        ball_y = y[frame] + offset  # 小球在曲线上方的位置偏移
        ball.set_data([ball_x], [ball_y])
        return line, ball,
    # 创建动画
    ani = FuncAnimation(fig, update, frames=len(x), interval=20, blit=True)
    # 使用imagemagick将动画保存为GIF图片
    ani.save(save_path+f"/1Ddynamic_{region}_dynamic_x^2.gif", writer='pillow')