# data source: https://www.kaggle.com/uciml/breast-cancer-wisconsin-data

import pandas as pd

def get_cancer_data(path = 'input/breast_cancer.csv'):
    data = pd.read_csv(path)
    data.insert(0, 'is_malignent', None)
    data.loc[data.diagnosis == 'M', 'is_malignent'] = 1
    data.loc[data.diagnosis == 'B', 'is_malignent'] = 0
    data.drop(columns = ['id', 'diagnosis', 'Unnamed: 32'], inplace = True)

    print('breast cancer data import succesful')
    return data
