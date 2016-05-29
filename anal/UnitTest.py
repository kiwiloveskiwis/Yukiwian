#!/usr/bin/env python
# encoding: utf-8
__author__ = 'Kiwi'
import pandas as pd
import numpy as np
import statsmodels.api as sm

def measure(array):
    UnitTestResult=sm.tsa.adfuller(array)
    print 'ADF: ', UnitTestResult[0]
    print 'P-VALUE: ', UnitTestResult[1]
    print 'Critical values:'
    for keys in UnitTestResult[4].keys():
        print keys , UnitTestResult[4][keys]
    if UnitTestResult[1] < 0.05:
        print 'Time series is stationary'
    else:
        print 'Time series is nonstationary'
    print '\n'

ts_for_artists = pd.read_csv(r'../data/ts_for_artists.csv', index_col = 0)

for artists in ts_for_artists:
    print 'Artists: ', artists
    measure(ts_for_artists[artists])
# print ts_for_artists.shape
