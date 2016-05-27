__author__ = 'Hatsuyuki'

from statsmodels.tsa.arima_model import ARIMA
import pandas as pd
import numpy as np

def summation(a):
    n = np.size(a)
    for i in np.arange(1, n):
        a[i] += a[i - 1]
    return a

ts = pd.read_csv(r'../data/ts_for_artists.csv', index_col = 0)
ts.index = pd.to_datetime(ts.index)
date_range = pd.date_range(start = '20150831', end = '20151030', freq = 'D')
date_size = np.size(date_range)
pred = pd.DataFrame(np.zeros((date_size, np.size(ts.columns))), index = date_range, columns = ts.columns)

c = 0
for artist in ts:
    print 'The ' + str(c) + '-th artist'
    c += 1
    min_aic = 1e100
    for p in range(10):
        for q in range(4):
            print p, q
            try:
                arima_model = ARIMA(ts[artist], (p, 1, q))
                arima_res = arima_model.fit(trend = 'nc', disp = -1)
                #print arima_res.summary()
                if arima_res.aic < min_aic:
                    x = arima_res.predict(start = '20150831', end = '20151030').values
                    x[0] += ts[artist].iloc[-1]
                    pred[artist] = summation(x)
                    min_aic = arima_res.aic
            except:
                pass

pred.to_csv(r'../data/pred_ARIMA.csv')