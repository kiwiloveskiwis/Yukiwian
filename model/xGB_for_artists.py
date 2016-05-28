__author__ = 'Hatsuyuki'

import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split

random_state = 2016

def diff(a):
    n = np.size(a)
    return a - np.append([0], a)[:n]

def summation(a):
    n = np.size(a)
    for i in np.arange(1, n):
        a[i] += a[i - 1]
    return a

params = {
        "objective": "reg:linear",
        "booster" : "gbtree",
        "eta": 0.1,
        "max_depth": 7,
        #"subsample": 0.8,
        #"colsample_bytree": 0.8,
        "silent": 0,
        "seed": random_state
}

ts = pd.read_csv(r'../data/ts.csv', index_col = 0)

for songs in ts.columns:
    ts[songs] = diff(ts[songs].values)

song = pd.read_csv(r'../data/songs_with_dummies.csv')
song = song.drop(['artist'], axis = 1)
song_dict = dict(zip(song.song.values, np.arange(song.shape[0])))

pred_range = pd.date_range('20150831', '20151030')
pred_size = np.size(pred_range)
pred = pd.DataFrame(data = np.zeros((pred_size, np.size(ts.columns))), index = pred_range, columns = ts.columns)

n_lags = 10

train = np.zeros((ts.shape[1] * (ts.shape[0] - n_lags), song.shape[1] - 1 + n_lags))
y = np.zeros((ts.shape[1] * (ts.shape[0] - n_lags), 1))

c = 0
row = 0
for songs in ts.columns:
    if c % 100 == 0: print c
    c += 1
    val = ts[songs].values
    song_info = song.iloc[song_dict[songs]].drop('song').values
    for i in np.arange(1, np.size(val) - n_lags): #start with 1 for diff
        train[row] = np.append(song_info, [val[i:i + n_lags]])
        y[row] = [val[i + n_lags]]
        row += 1

print np.shape(train), np.shape(y)

dat_tmp = np.concatenate((y, train), axis = 1)
train, val = train_test_split(dat_tmp, test_size = 0.1, random_state = random_state)

print np.shape(train), np.shape(val)

dat_train = xgb.DMatrix(train[:, 1:], label = train[:, 0])
dat_val = xgb.DMatrix(val[:, 1:], label = val[:, 0])
eval = [(dat_train, 'train'), (dat_val, 'eval')]

gbm = xgb.train(params, dat_train, 100, evals = eval, early_stopping_rounds = 10)

c = 0
for songs in ts.columns:
    if c % 100 == 0: print c
    c += 1
    val = ts[songs].values
    song_info = song.iloc[song_dict[songs]].drop('song').values
    for i in np.arange(0, pred_size):
        val = np.append(val, gbm.predict(xgb.DMatrix(np.array([np.append(song_info , [val[-n_lags:]])])), ntree_limit = gbm.best_ntree_limit))
    pred[songs] = val[-pred_size:]

for songs in pred.columns:
    pred[songs].iloc[0] += ts[songs].iloc[-1]
    pred[songs] = summation(pred[songs].values)

pred.to_csv(r'../data/pred_xgb.csv')
