__author__ = 'Hatsuyuki'

import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
from ts_anal_tools.diff import diff

ts_for_artists = pd.read_csv(r'../data/ts_for_artists.csv', index_col = 0)
ts_for_artists.columns = np.arange(50, dtype = int)

for i in np.arange(0): ts_for_artists = ts_for_artists.apply(diff, axis = 0)
#ts_for_artists = ts_for_artists.apply(np.sum, axis = 1)
ts_for_artists.iloc[:, :].plot(figsize = (12, 8))

plt.show()