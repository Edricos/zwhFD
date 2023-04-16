import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from fcmeans import FCM
from sklearn.preprocessing import minmax_scale
# data = pd.read_csv('iris.csv')
# x = data.iloc[:,1:5]
# print(x.head())
data = pd.read_csv('wine.csv')
x = data.iloc[:,1:14]
x = x.to_numpy()
# x = minmax_scale(x, axis=0)

# 计算每列的均值和标准差
mean = np.mean(x, axis=0)
std = np.std(x, axis=0)
# 对数据进行标准化
x = (x - mean) / std

scores = []
for k in range(2, 10):
    means = FCM(n_clusters=k)
    means.fit(x)
    # y_pred = KMeans(n_clusters=k, random_state=9).fit_predict(x)
    y_pred = means.predict(x)
    score = metrics.davies_bouldin_score(x, y_pred)
    scores.append(score)

print(scores)
plt.plot(range(2, 10), scores, 'bx-')
plt.xlabel('k')
plt.ylabel('the Davies Bouldin Score')
plt.title('the DB Method')
plt.show()
