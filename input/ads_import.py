import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD

import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file

#Data source: https://archive.ics.uci.edu/ml/datasets/Farm+Ads

def get_ads_data(path = 'input/farm-ads-vect', n_components = 2000, random_state = 0):
    X, y = load_svmlight_file(path)

    X = normalize(X)

    # n_components equal 1000 explains 86% of variance
    # n_components equal 2000 explains 98% of variance
    print('Fitting SVD with n_components: ' + str(n_components))
    svd = TruncatedSVD(random_state=random_state, n_components=n_components)
    svd.fit(X)
    print('Ads variance lost from SVD in data import: ' + str(round(100-svd.explained_variance_ratio_.sum() * 100.0, 2)) + '%')
    X = svd.transform(X)

    X = pd.DataFrame(X)

    y = np.array(y)
    y[np.where(y == -1)] = 0
    CLASS_NAME = 'ad_approved'

    X.insert(0, CLASS_NAME, y)

    print('ads data import succesful')
    return X

###
###
###
def OLD_get_ads_data(path='input/farm-ads-vect', n=5000, random_state = 1):
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
