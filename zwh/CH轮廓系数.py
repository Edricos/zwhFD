import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from fcmeans import FCM

# data = pd.read_csv('wine.csv')
data = pd.read_csv('iris.csv')
x = data.iloc[:,1:5]
# x = data.iloc[:,1:14] # 葡萄酒数据
print(x.head())

x = x.to_numpy()

silhouette_scores = []
for k in range(2, 10):
    means = FCM(n_clusters=k)
    means.fit(x)

    # y_pred = KMeans(n_clusters=k, random_state=9).fit_predict(x)
    y_pred = means.predict(x)
    silhouette_scores.append(silhouette_score(x, y_pred))
print(silhouette_scores)

plt.plot(range(2, 10), silhouette_scores, 'bx-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('silhouette')
plt.title('the Silhouette Method')
plt.show()
