# https://archive.ics.uci.edu/ml/datasets/ISOLET

import pandas as pd


def get_voice_data():
    # IMPORT DATA AND MAKE FIRST COLUMN LABEL
    data_train = pd.read_csv('input/isolet1+2+3+4.data', header=None)
    data_test = pd.read_csv('input/isolet5.data', header=None)
    data = data_train.append(data_test).reset_index().drop(columns=['index'])
    data_label = data[617]
    data = data.drop(columns = [617])
    data.insert(0, 'label', data_label)
    return data

