import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from fcmeans import FCM

data = pd.read_csv('wine.csv')
x = data.iloc[:,1:14]
x = x.to_numpy()

scores = []
for k in range(2, 10):
    means = FCM(n_clusters=k)
    means.fit(x)

    # y_pred = KMeans(n_clusters=k, random_state=9).fit_predict(x)
    y_pred = means.predict(x)
    score = metrics.calinski_harabasz_score(x, y_pred)
    scores.append(score)

print(scores)
plt.plot(range(2, 10), scores, 'bx-')
plt.xlabel('k')
plt.ylabel('the Calinski Carabasz Score')
plt.title('the CH Method')
plt.show()

