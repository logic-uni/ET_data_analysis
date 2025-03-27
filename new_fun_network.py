import numpy as np
import networkx as nx
from sklearn.metrics import mutual_info_score

def build_functional_network(matrix):
    """构建功能连接网络"""
    n_neurons = matrix.shape[0]
    connectivity = np.zeros((n_neurons, n_neurons))
    
    # 计算互信息
    for i in range(n_neurons):
        for j in range(i+1, n_neurons):
            mi = mutual_info_score(matrix[i], matrix[j])
            connectivity[i, j] = connectivity[j, i] = mi
    
    # 二值化（基于中位数阈值）
    threshold = np.median(connectivity)
    adj_matrix = (connectivity > threshold).astype(int)
    
    return nx.from_numpy_array(adj_matrix)

# 生成测试数据
matrix = np.random.randint(0, 2, (50, 500))  # 较小规模以便快速计算

# 构建网络
G = build_functional_network(matrix)

# 计算关键指标
clustering = nx.average_clustering(G)
communities = nx.algorithms.community.greedy_modularity_communities(G)
modularity = nx.algorithms.community.modularity(G, communities)

print(f"聚类系数: {clustering:.3f}, 模块度: {modularity:.3f}")

# 判断失稳（示例规则）
if modularity < 0.2 or clustering < 0.1:
    print("网络结构异常（失稳风险）")