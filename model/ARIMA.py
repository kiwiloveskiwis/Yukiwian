__author__ = 'Hatsuyuki'

from statsmodels.tsa.arima_model import ARIMA
import pandas as pd
import numpy as np

def summation(a):
    n = np.size(a)
    return a + np.append([0], a)[1:]

ts = pd.read_csv(r'../data/ts_for_artists.csv', index_col = 0)
ts.index = pd.to_datetime(ts.index)
date_range = pd.date_range(start = '20150831', end = '20151030', freq = 'D')
date_size = np.size(date_range)
pred = pd.DataFrame(np.zeros((date_size, np.size(ts.columns))), index = date_range, columns = ts.columns)

for artist in ts:
    min_aic = 1e100
    for p in range(8):
        for q in range(4):
            try:
                arima_model = ARIMA(ts[artist], (7, 1, 1))
                arima_res = arima_model.fit(transparams = True)
                #print arima_res.summary()
                if arima_res.aic < min_aic:
                    pred[artist] = summation(arima_res.predict(start = '20150831', end = '20151030'))
                    min_aic = arima_res.aic
            except:
                pass

pred.iloc[0] += ts.iloc[-1]

pred.to_csv('pred_ARIMA.csv')