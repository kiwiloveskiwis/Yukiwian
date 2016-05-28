__author__ = 'Hatsuyuki'

import pandas as pd

def ensemble_ave(list_of_sub, weights, output_file_name):
    x = None
    for file_name,weight in zip(list_of_sub, weights):
        if x is None:
            x = pd.read_csv(r'../data/' + file_name, header = None)
            x[1] *= weight
        else:
            tmp = pd.read_csv(r'../data/' + file_name, header = None)
            x[1] += weight * tmp[1]
    x.to_csv(r'../data/' + output_file_name, index = False, header = False)
