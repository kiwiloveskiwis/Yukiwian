__author__ = 'Hatsuyuki'

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

random_state = 2016
n_lags = 7
alpha = 2.0 / (n_lags + 1)
thres = 8

def expma(prev, cur, alpha, flag = True):
    if flag: return prev * (1 - alpha) + cur * alpha
    else: return cur

ts = pd.read_csv(r'../data/ts.csv', index_col = 0)
ts_like = pd.read_csv(r'../data/ts_like.csv', index_col = 0)
ts_down = pd.read_csv(r'../data/ts_down.csv', index_col = 0)

pred_range = pd.date_range('20150831', '20151030')
pred_size = np.size(pred_range)
pred = pd.DataFrame(data = np.zeros((pred_size, np.size(ts.columns))), index = pred_range, columns = ts.columns)

train = np.zeros((ts.shape[0] - n_lags, n_lags * 3 + 3))
y = np.zeros((ts.shape[0] - n_lags, 3))

c = 0

for songs in ts.columns:
    print c
    c += 1

    val = ts[songs].values
    if np.sum(val) < thres:
        pred[songs] = np.array([np.sum(val) * 1.0 / ts.shape[0] for i in range(pred_size)])
        continue

    val_like = ts_like[songs].values
    val_down = ts_down[songs].values

    clf = RandomForestRegressor(n_estimators = 50, random_state = random_state, n_jobs = 2)

    ma = 0, 0
    ma_like = 0, 0
    ma_down = 0, 0

    for i in np.arange(0, np.size(val) - n_lags):
        train[i] = np.concatenate((val[i:i + n_lags], val_like[i:i + n_lags], val_down[i:i + n_lags], [ma, ma_like, ma_down]))
        y[i] = [val[i + n_lags], val_like[i + n_lags], val_down[i + n_lags]]

        ma = expma(ma, val[i], alpha, i)
        ma_like = expma(ma_like, val_like[i], alpha, i)
        ma_down = expma(ma_down, val_down[i], alpha, i)

    clf.fit(train, y)
    print clf.feature_importances_

    for i in np.arange(pred_size):
        x_t = np.concatenate((val[-n_lags:], val_like[-n_lags:], val_down[-n_lags:], [ma, ma_like, ma_down]))
        y_t = clf.predict([x_t])
        val = np.append(val, y_t[0][0])
        val_like = np.append(val_like, y_t[0][1])
        val_down = np.append(val_down, y_t[0][2])

        ma = expma(ma, val[-n_lags - 1], alpha, True)
        ma_like = expma(ma_like, val_like[-n_lags - 1], alpha, True)
        ma_down = expma(ma_down, val_down[-n_lags - 1], alpha, True)

    pred[songs] = val[-pred_size:]

pred.to_csv(r'../data/pred.csv')
