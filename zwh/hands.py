# 导包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from fcmeans import FCM

# 1.利用pandas读入数据
# data = pd.read_csv('wine.csv')# data_path换成你的需要聚类的数据所在路径，我的数据没有表头，所以设置header=None
data = pd.read_csv('wine.csv')
x = data.iloc[:,1:14] # 葡萄酒数据
# x = data.iloc[:,1:5] #鸢尾花数据
print(x.head())
x = x.to_numpy()


# 2.绘制手肘图
dispersions = []
for k in range(2, 10): # k:需要聚几类，按需修改，推荐至少10类
    # means = KMeans(n_clusters=k, random_state=9)
    means = FCM(n_clusters=k)
    means.fit(x)
    # y_pred = means.fit_predict(x)
    y_pred = means.predict(x)
    # dispersions.append(sum(np.min(cdist(x, means.cluster_centers_, 'euclidean'), axis=1)) / x.shape[0])
    dispersions.append(sum(np.min(cdist(x, means.centers, 'euclidean'), axis=1)) / x.shape[0])
print(dispersions)

plt.plot(range(2, 10), dispersions, 'bx-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Average Dispersion')
# 平均离散
plt.title('the Elbow Method')
plt.show()
