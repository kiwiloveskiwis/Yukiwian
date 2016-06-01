__author__ = 'Hatsuyuki'

import pandas as pd
import numpy as np
import xgboost as xgb

random_state = 2016
n_lags = 7
alpha = 2.0 / (n_lags + 1)
thres = 8
params = {
        "objective": "reg:linear",
        "booster" : "gbtree",
        "eta": 0.1,
        "max_depth": 8,
        "silent": 1,
        "seed": random_state
}

def expma(prev, cur, alpha, flag = True):
    if flag: return prev * (1 - alpha) + cur * alpha
    else: return cur

ts = pd.read_csv(r'../data/ts_for_artists.csv', index_col = 0)
ts_like = pd.read_csv(r'../data/ts_like_for_artists.csv', index_col = 0)
ts_down = pd.read_csv(r'../data/ts_down_for_artists.csv', index_col = 0)

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

    prev, ma = 0, 0
    prev_like, ma_like = 0, 0
    prev_down, ma_down = 0, 0

    for i in np.arange(0, np.size(val) - n_lags):
        train[i] = np.concatenate((val[i:i + n_lags], val_like[i:i + n_lags], val_down[i:i + n_lags], [ma, ma_like, ma_down]))
        y[i] = [val[i + n_lags], val_like[i + n_lags], val_down[i + n_lags]]

        ma = expma(prev, val[i], alpha, i)
        prev = ma
        ma_like = expma(prev_like, val_like[i], alpha, i)
        prev_like = ma_like
        ma_down = expma(prev_down, val_down[i], alpha, i)
        prev_down = ma_down

    dat_train = xgb.DMatrix(train, label = y[:, 0])
    dat_train_like = xgb.DMatrix(train, label = y[:, 1])
    dat_train_down = xgb.DMatrix(train, label = y[:, 2])
    eval = [(dat_train, 'train')]
    eval_like = [(dat_train_like, 'train')]
    eval_down = [(dat_train_down, 'train')]
    gbm = xgb.train(params, dat_train, 34, evals = eval)
    gbm_like = xgb.train(params, dat_train_like, 34, evals = eval_like)
    gbm_down = xgb.train(params, dat_train_down, 34, evals = eval_down)

    for i in np.arange(pred_size):
        x_t = np.concatenate((val[-n_lags:], val_like[-n_lags:], val_down[-n_lags:], [ma, ma_like, ma_down]))
        y_t = gbm.predict(xgb.DMatrix([x_t]))
        y_t_like = gbm_like.predict(xgb.DMatrix([x_t]))
        y_t_down = gbm_down.predict(xgb.DMatrix([x_t]))
        val = np.append(val, y_t)
        val_like = np.append(val_like, y_t_like)
        val_down = np.append(val_down, y_t_down)

        ma = expma(prev, val[-n_lags - 1], alpha, True)
        prev = ma
        ma_like = expma(prev_like, val_like[-n_lags - 1], alpha, True)
        prev_like = ma_like
        ma_down = expma(prev_down, val_down[-n_lags - 1], alpha, True)
        prev_down = ma_down

    pred[songs] = val[-pred_size:]
    #print val

pred.to_csv(r'../data/pred_xGB.csv')