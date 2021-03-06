__author__ = 'Hatsuyuki'

import pandas as pd
import numpy as np

def create_submission(pred):
    songs = pd.read_csv(r'../data/songs.csv', index_col = 0)
    artist = np.unique(songs['artist'].values)
    submission = pd.DataFrame(data = np.zeros((np.size(pred.index), np.size(artist))), index = pred.index, columns = artist)
    #c = 0
    for col in pred:
        #c += 1
        #print c
        #submission[songs.loc[col]['artist']] += pred[col].values.astype(int)
        submission[songs.loc[col]['artist']] += pred[col]
    return submission

def write_to_file(name_of_file, submission):
    f = open(r'../data/' + name_of_file, 'w')
    for col in submission:
        for ind in submission.index:
            if ind == '2015-08-31': continue
            #s = col + ',' + str(max(submission.loc[ind, col], 10.0)) + ',' + ind.replace('-','') + '\n'
            s = col + ',' + str(submission.loc[ind, col]) + ',' + ind.replace('-','') + '\n'
            f.write(s)
    f.close()

def one_step_convert(name_of_output_file, name_of_input_file, flag = False):
    submission = pd.read_csv(r'../data/' + name_of_input_file, index_col = 0)
    if flag == True: submission = create_submission(submission)
    write_to_file(name_of_output_file, submission)
