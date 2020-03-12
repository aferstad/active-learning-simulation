# https://archive.ics.uci.edu/ml/datasets/ISOLET

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def __get_raw_voice_data():
    # IMPORT DATA AND MAKE FIRST COLUMN LABEL
    data_train = pd.read_csv('input/voice/isolet1+2+3+4.data', header=None)
    data_test = pd.read_csv('input/voice/isolet5.data', header=None)
    data = data_train.append(data_test).reset_index().drop(columns=['index'])
    data_label = data[617]
    data = data.drop(columns = [617])
    data.insert(0, 'label', data_label)
    return data


def get_voice_data(n_components = 300):
    data = __get_raw_voice_data()

    scaler = StandardScaler()
    pca = PCA(n_components=n_components)

    X = data.iloc[:,1:]
    y = data.iloc[:,0]

    scaler.fit(X)
    X = scaler.transform(X)

    pca.fit(X)
    X = pca.transform(X)

    pct_variance_kept = sum(pca.explained_variance_ratio_[:n_components])

    print('variance kept due to PCA: ' + str(round(pct_variance_kept, 2)))

    data_pca = pd.DataFrame(X)
    data_pca.insert(0, 'label', y)

    return data_pca
