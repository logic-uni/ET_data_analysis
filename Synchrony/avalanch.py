def detect_avalanches(mat, min_neurons=1):
    # 基于总激活神经元数的雪崩检测
    active = (np.sum(mat, axis=0) >= min_neurons).astype(int)
    avalanches = []
    current_length = 0
    for t in range(len(active)):
        if active[t]:
            current_length += 1
        elif current_length > 0:
            avalanches.append(current_length)
            current_length = 0
    return avalanches

# 动态调整阈值（例如至少1个神经元激活）
avalanches_sub3 = detect_avalanches(submatrices[2], min_neurons=1)

# 检查雪崩列表是否为空
if len(avalanches_sub3) == 0:
    print("No avalanches detected. Consider lowering min_neurons.")
else:
    # 绘制雪崩分布
    plt.figure(figsize=(8, 5))
    plt.hist(avalanches_sub3, bins=np.logspace(0, np.log10(20), 10), density=True)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Avalanche Size (log scale)')
    plt.ylabel('Probability Density (log scale)')
    plt.title('Avalanche Size Distribution (Submatrix 3)')
    plt.grid()
    plt.show()

    # 附加分析：检查是否符合幂律

    fit = Fit(avalanches_sub3, discrete=True)
    print(f"Powerlaw exponent: {fit.power_law.alpha:.2f}")
    fit.plot_pdf(color='b', linewidth=2)
    fit.power_law.plot_pdf(color='r', linestyle='--', ax=plt.gca())
    plt.show()