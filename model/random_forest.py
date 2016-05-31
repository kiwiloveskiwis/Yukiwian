__author__ = 'Hatsuyuki'

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

random_state = 2016
n_lags = 7
alpha = 2.0 / (n_lags + 1)

def expma(prev, cur, alpha, flag = True):
    if flag: return prev * (1 - alpha) + cur * alpha
    else: return cur

def days_between(d1, d2):
    d1 = datetime.strptime(d1, "%Y%m%d")
    d2 = datetime.strptime(d2, "%Y%m%d")
    return (d2 - d1).days

ts = pd.read_csv(r'../data/ts.csv', index_col = 0)
ts_like = pd.read_csv(r'../data/ts_like.csv', index_col = 0)
ts_down = pd.read_csv(r'../data/ts_down.csv', index_col = 0)

pred_range = pd.date_range('20150831', '20151030')
pred_size = np.size(pred_range)
pred = pd.DataFrame(data = np.zeros((pred_size, np.size(ts.columns))), index = pred_range, columns = ts.columns)

song = pd.read_csv(r'../data/songs.csv')
song = song.drop(['artist', 'lang', 'gender'], axis = 1)
song_dict = dict(zip(song.song.values, np.arange(song.shape[0])))

train = np.zeros((ts.shape[0] - n_lags, n_lags * 3 + 4))
y = np.zeros((ts.shape[0] - n_lags,))

c = 0

for songs in ts.columns:
    print c
    c += 1

    clf = RandomForestRegressor(n_estimators = 100, random_state = random_state, n_jobs = 2)

    val = ts[songs].values
    val_like = ts_like[songs].values
    val_down = ts_like[songs].values

    prev, ma = 0, 0
    prev_like, ma_like = 0, 0
    prev_down, ma_down = 0, 0

    publish_time, init_play = song.iloc[song_dict[songs]].drop('song').values
    publish_time = days_between(str(publish_time), '20150301')

    for i in np.arange(0, np.size(val) - n_lags):
        if publish_time <= 0: ave_play = 0
        else: ave_play = init_play * 1.0 / publish_time

        train[i] = np.concatenate((val[i:i + n_lags], val_like[i:i + n_lags], val_down[i:i + n_lags], [ma, ma_like, ma_down, ave_play]))
        y[i] = val[i + n_lags]

        publish_time += 1
        init_play += val[i]

        ma = expma(prev, val[i], alpha, i)
        prev = ma
        ma_like = expma(prev_like, val_like[i], alpha, i)
        prev_like = ma_like
        ma_down = expma(prev_down, val_down[i], alpha, i)
        prev_down = ma_down

    clf.fit(train, y)
    print clf.feature_importances_

    for i in np.arange(pred_size):
        if publish_time <= 0: ave_play = 0
        else: ave_play = init_play * 1.0 / publish_time

        x_t = np.concatenate((val[-n_lags:], val_like[-n_lags:], val_down[-n_lags:], [ma, ma_like, ma_down, ave_play]))
        y_t = clf.predict([x_t])
        val = np.append(val, y_t)

        publish_time += 1
        init_play += val[-n_lags - 1]

        ma = expma(prev, val[-n_lags - 1], alpha, True)
        prev = ma
        ma_like = expma(prev_like, val_like[-n_lags - 1], alpha, True)
        prev_like = ma_like
        ma_down = expma(prev_down, val_down[-n_lags - 1], alpha, True)
        prev_down = ma_down

    pred[songs] = val[-pred_size:]
    #print pred[songs]

pred.to_csv(r'../data/pred.csv')