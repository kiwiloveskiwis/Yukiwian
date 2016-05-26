__author__ = 'Hatsuyuki'

import pandas as pd
import numpy as np

def create_submission(pred = pd.read_csv(r'../data/pred.csv', index_col = 0)):
    songs = pd.read_csv(r'../data/songs.csv', index_col = 0)
    artist = np.unique(songs['artist'].values)
    submission = pd.DataFrame(data = np.zeros((np.size(pred.index), np.size(artist))), index = pred.index, columns = artist)
    #c = 0
    for col in pred:
        #c += 1
        #print c
        submission[songs.loc[col]['artist']] += pred[col]
    return submission

def write_to_file(name_of_file, submission):
    f = open(r'../data/sub_0525.csv', 'w')
    for col in submission:
        for ind in submission.index:
            if ind == '2015-08-31': continue
            s = col + ',' + str(int(submission.loc[ind, col])) + ',' + ind.replace('-','') + '\n'
            f.write(s)