import pandas as pd
import numpy as np
from sklearn.datasets import load_svmlight_file

#Data source: https://archive.ics.uci.edu/ml/datasets/Farm+Ads
def get_ads_data(path='input/farm-ads-vect', n=5000, random_state = 1):
    '''
    parameters n and random state allows to sample randomly less than
    the total 54877 columns, in order to reduce run time
    '''


    if n > 54877:
        n = 54877 #this is the max number of columns to sample
    elif n <= 0:
        print('Error: negative n for ads number of rows')

    X, y = load_svmlight_file(path)
    X = pd.DataFrame(X.todense())
    y = pd.Series(y)

    X = X.sample(n=n, random_state = random_state, axis = 1)

    y = np.array(y)
    y[np.where(y == -1)] = 0

    CLASS_NAME = 'ad_approved'

    X.insert(0, CLASS_NAME, y)
    X.reset_index(drop = True, inplace = True)
    column_names = [CLASS_NAME]
    column_names.extend(range(0, X.shape[1] - 1))
    X.columns = column_names


    return X
