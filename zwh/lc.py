import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import linkage
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from fcmeans import FCM
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import minmax_scale


def inter_cluster_distance(X, labels):
    n_clusters = len(np.unique(labels))
    distances = []
    for i in range(n_clusters):
        for j in range(i+1, n_clusters):
            mask_i = labels == i
            mask_j = labels == j
            distance = np.mean(pairwise_distances(X[mask_i], X[mask_j]))
            distances.append(distance)
    return np.mean(distances)

def intra_cluster_distance(X, labels):
    n_clusters = len(np.unique(labels))
    distances = []
    for i in range(n_clusters):
        mask = labels == i
        distance = np.mean(pairwise_distances(X[mask]))
        distances.append(distance)
    return np.mean(distances)

# def evaluate_clustering(X, labels):
#     '''
#        这段代码计算评价指标的方法如下：
#
#         对于每个簇，计算该簇中每个样本与该簇的中心点之间的距离，并将这些距离相加，得到该簇的类内距离和。
#
#         计算所有簇的类内距离和的平均值。
#         计算所有簇之间的距离，并找到最大的距离。
#         将最大距离除以平均类内距离和，得到评价指标。
#
#         这个评价指标越小，表示聚类结果越好。因为它表示了簇之间的差异相对于簇内差异的比例。
#         如果簇之间的差异很大，而簇内差异很小，那么评价指标就会很大，表示聚类结果不太好。
#         反之，如果簇之间的差异很小，而簇内差异很大，那么评价指标就会很小，表示聚类结果很好。
#     '''
#
#     # 计算每个样本与其所属簇中心的距离
#     intra_dists = []
#     for i in np.unique(labels):
#         cluster = X[labels == i]
#         centroid = cluster.mean(axis=0)
#         # intra_dists.append(pairwise_distances(cluster, centroid.reshape(1, -1)).sum())
#         intra_dists.append(pairwise_distances(cluster, centroid.reshape(1, -1)).mean())
#
#     # 计算每个簇之间的距离
#     inter_dists = linkage(X, method='ward')
#
#     # 计算评价指标
#     score = np.mean(intra_dists) / inter_dists[-1, 2]
#     return score

data = pd.read_csv('iris.csv')
x = data.iloc[:,1:5]
print(x.head())
# data = pd.read_csv('wine.csv')
# x = data.iloc[:,1:14]
xn = x.to_numpy()
# print(x)
# transformer = Normalizer().fit(x)
# x = transformer.transform(x)
x = minmax_scale(xn, axis=0)

scores = []
for k in range(2, 10):
    means = FCM(n_clusters=k)
    means.fit(x)
    # y_pred = KMeans(n_clusters=k, random_state=9).fit_predict(x)
    y_pred = means.predict(x)
    # score = evaluate_clustering(x, y_pred)
    # 类间距离和平均值 / 类内距离和平均值
    score = inter_cluster_distance(x, y_pred) / intra_cluster_distance(x, y_pred)
    scores.append(score)

print(scores)
plt.plot(range(2, 10), scores, 'bx-')
plt.xlabel('k')
plt.ylabel('the L(c) Score')
plt.title('the L(c) Method')
plt.show()