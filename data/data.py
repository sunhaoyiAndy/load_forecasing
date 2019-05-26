"""
Processing the data
"""
import copy
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
color = sns.color_palette()
from util.test_stationarity import *
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def process_data(train, test):
    df1 = pd.read_csv(train, encoding='utf-8', index_col='时间')
    data1 = pd.DataFrame(data=df1['实际功率'].values, index=df1.index, columns=['value'])
    # print(data1)
    ts = data1['value']
    print(ts)

    df2 = pd.read_csv(test, encoding='utf-8', index_col='时间')
    # print(df1.index)
    x_train = [] # 6列 -> 9
    for i in range(len(df1)):
        temp = []
        for item in df1.columns[:-2]:
            temp.append(df1[item].values[i])
        x_train.append(temp)
    y_train = df1[df1.columns[-1]].values  # 实际功率

    x_test = []
    for i in range(len(df2)):
        temp = []
        for item in df2.columns[:-2]:
            temp.append(df2[item].values[i])
        x_test.append(temp)
        # print(len(temp))
    y_test = df2[df2.columns[-1]].values  # 实际功率
    x_train, y_train, x_test, y_test = np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)


    # 归一化
    # scaler = StandardScaler().fit/(df1[attr].values)
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(y_test.reshape(-1, 1))

    temp = scaler.transform(y_test.reshape(1, -1))
    # print(temp)
    return x_train, y_train, x_test, y_test, scaler

    # 开启归一化
    # X_train, Y_train, X_test, Y_test = scaler.transform(x_train),  scaler.transform(y_train), scaler.transform(x_test.reshape(-1, 1)), scaler.transform(y_test.reshape(-1, 1))
    # print(X_train.shape)
    # print(Y_train.shape)
    # print(X_test.shape)
    # print(Y_test.shape)

    # return X_train, Y_train, X_test, Y_test, scaler


if __name__ == '__main__':
    file1 = 'train.csv'
    file2 = 'test.csv'
    df = pd.read_csv(file1)
    # print(df.head())
    X_train, y_train, x_test, y_test, scaler= process_data(file1, file2)
