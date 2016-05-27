__author__ = 'Hatsuyuki'

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

random_state = 2016

ts = pd.read_csv(r'../data/ts.csv', index_col = 0)
song = pd.read_csv(r'../data/songs_with_dummies.csv')
song = song.drop(['artist'], axis = 1)
song_dict = dict(zip(song.song.values, np.arange(song.shape[0])))

pred_range = pd.date_range('20150831', '20151030')
pred_size = np.size(pred_range)
pred = pd.DataFrame(data = np.zeros((pred_size, np.size(ts.columns))), index = pred_range, columns = ts.columns)

n_lags = 7

train = np.zeros((ts.shape[1] * (ts.shape[0] - n_lags), song.shape[1] - 1 + n_lags))
y = np.zeros((ts.shape[1] * (ts.shape[0] - n_lags), 1))

c = 0
row = 0
for songs in ts.columns:
    if c % 100 == 0: print c
    c += 1
    val = ts[songs].values
    song_info = song.iloc[song_dict[songs]].drop('song').values
    for i in np.arange(0, np.size(val) - n_lags):
        train[row] = np.append(song_info, [val[i:i + n_lags]])
        y[row] = [val[i + n_lags]]
        row += 1
        #if train == []: train = [np.append(song_info, [val[i:i + n_lags]])]
        #else: train = np.concatenate((train, [np.append(song_info, [val[i:i + n_lags]])]))
        #y = np.append(y, val[i + n_lags])

train.dump(r'../data/train_all.np')
y.dump(r'../data/y_all.np')
clf = RandomForestRegressor(n_estimators = 300, random_state = random_state, verbose = 1, n_jobs = 2)
clf.fit(train, y)

c = 0
for songs in ts.columns:
    if c % 100 == 0: print c
    c += 1
    val = ts[songs].values
    song_info = song.iloc[song_dict[songs]].drop('song').values
    for i in np.arange(0, pred_size):
        val = np.append(val, clf.predict(np.append(song_info , [val[-n_lags:]])))
    pred[songs] = val[-pred_size:]

pred.to_csv('pred.csv', index = False)