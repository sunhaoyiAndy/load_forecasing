import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

filename = "month.csv"
res = np.loadtxt(fname=filename,dtype=str,delimiter=',',usecols=8)
print(res.shape)
data = [] #一维数组

for i in range(len(res)):
    tt = eval(res[i])
    data.append(float(tt))
# print(len(data))
count_row = round(len(data) / 96)
data = np.array(data).reshape(count_row, 96)

# estimator =KMeans(n_clusters=4)   #构造一个聚类数为5的聚类器
estimator = AgglomerativeClustering()
estimator.fit(data)   #聚类
label_pred = estimator.labels_  #获取聚类标签
# centroids = estimator.cluster_centers_ #获取聚类中心
print(label_pred)

for i in range(len(label_pred)):
    if label_pred[i] == 0:
        x = [i for i in range(96)]
        plt.plot(x, data[i], '#e24fff')
    if label_pred[i] == 1:
        x = [i for i in range(96)]
        plt.plot(x, data[i], 'g')
    if label_pred[i] == 2:
        x = [i for i in range(96)]
        plt.plot(x, data[i], 'r')
    if label_pred[i] == 3:
        x = [i for i in range(96)]
        plt.plot(x, data[i], 'k')
    # if label_pred[i] == 4:
    #     x = [i for i in range(48)]
    #     plt.plot(x, data[i], 'c')
plt.show()













# df = pd.read_csv(filename)
# # df = df.set_index('时间')
# # print(df.dtypes)
# #
# # print(df.loc['2016-04'])
# print(pd.date_range('2017-08-01',periods=30))
# for i in pd.date_range('2017-08-01',periods=30):
#     print(type(i))