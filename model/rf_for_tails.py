__author__ = 'Hatsuyuki'

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

random_state = 2016
n_lags = 7
thres = 40

def nor(x, y = 0):
    x_bar = np.mean(x)
    s_x = np.sqrt(np.sum((x - np.mean(x)) ** 2))
    if np.abs(s_x) > 1e-10: return (x - x_bar) / s_x, (y - x_bar) / s_x, x_bar, s_x
    else: return x - x_bar, y - x_bar, x_bar, 1

ts = pd.read_csv(r'../data/ts_for_artists.csv', index_col = 0)

tail = pd.read_csv(r'../data/tail.csv')

chosen = tail.loc[tail.dlag > thres, 'artists'].values

pred_range = pd.date_range('20150831', '20151030')
pred_size = np.size(pred_range)
pred = pd.DataFrame(data = np.zeros((pred_size, np.size(ts.columns))), index = pred_range, columns = ts.columns)

train = np.zeros((ts.shape[0] - n_lags, n_lags))
y = np.zeros((ts.shape[0] - n_lags, 1))

c = 0

for art in chosen:
    print c
    c += 1
    dlag = tail.loc[tail.artists == art, 'dlag'].values[0]
    clf = RandomForestRegressor(n_estimators = 100, random_state = random_state, n_jobs = 2)
    val = ts[art].iloc[dlag:].values
    for i in np.arange(0, np.size(val) - n_lags):
        train[i] ,y[i], x_bar, s_x = nor(val[i:i + n_lags], val[i + n_lags])
    clf.fit(train[:np.size(val) - n_lags], y[:np.size(val) - n_lags])
    for i in np.arange(pred_size):
        x, y_t, x_bar, s_x = nor(val[-n_lags:])
        y_t = clf.predict([x])
        y_t = y_t * s_x + x_bar
        val = np.append(val, y_t)
    pred[art] = val[-pred_size:]

pred.to_csv(r'../data/pred_for_tails.csv')
