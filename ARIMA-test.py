import copy
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
color = sns.color_palette()
from util.test_stationarity import *
import warnings 
warnings.filterwarnings('ignore')

# 导入数据，修改时序索引
filename = 'data/201707-201807_pd.xlsx'
df = pd.read_excel(filename)  # (17568, 2)
date_index = pd.DatetimeIndex(pd.date_range(start='7/20/2017', periods=17568, freq='30T').values)
data = pd.DataFrame(df['YY_MW'].values, index=date_index)
# print(data.head())

day = data.resample('1D', how='mean', closed='left')  # 日均
data = copy.deepcopy(day)
# print('size: ', data.shape)


def draw_ts(timeseries):
    plt.figure(figsize=(24, 8))
    timeseries.plot()
    plt.show()


ts = data[0]
print(ts)
# draw_ts(ts)  # 原始数据绘图

# arima 时序分解
decomposition = seasonal_decompose(ts[:-20], model='additive', freq=20, two_sided=False)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
decomposition.plot()
plt.show()
print(trend)