import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import linkage
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from fcmeans import FCM

def evaluate_clustering(X, labels):
    # 计算每个样本与其所属簇中心的距离
    intra_dists = []
    for i in np.unique(labels):
        cluster = X[labels == i]
        centroid = cluster.mean(axis=0)
        intra_dists.append(pairwise_distances(cluster, centroid.reshape(1, -1)).sum())

    # 计算每个簇之间的距离
    inter_dists = linkage(X, method='ward')

    # 计算评价指标
    score = inter_dists[-1, 2] / np.mean(intra_dists)
    return score

data = pd.read_csv('iris.csv')
x = data.iloc[:,1:5]
print(x.head())
# data = pd.read_csv('wine.csv')
# x = data.iloc[:,1:14]
x = x.to_numpy()

scores = []
for k in range(2, 10):
    means = FCM(n_clusters=k)
    means.fit(x)
    # y_pred = KMeans(n_clusters=k, random_state=9).fit_predict(x)
    y_pred = means.predict(x)
    score = evaluate_clustering(x, y_pred)
    scores.append(score)

print(scores)
plt.plot(range(2, 10), scores, 'bx-')
plt.xlabel('k')
plt.ylabel('the L(c) Score')
plt.title('the DB Method')
plt.show()