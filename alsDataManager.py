import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from scipy import spatial  # for nearest neighbour


def get_X_y(df):
    '''
    Returns first column of df as y and rest as X
    '''
    X = df.drop(df.columns[0], axis=1)
    y = df.iloc[:, 0]

    return X, y


def nearest_neighbour(points_a, points_b):
    '''
    for each point in A, return the closes point in B
    '''
    tree = spatial.cKDTree(points_b)
    return pd.Series(tree.query(points_a)[1])


class AlsDataManager:

    def __init__(self, als):
        self.als = als

    # PARTITIONING FUNCTIONS:
    def set_partitions(self):
        self.__set_partition_sizes()

        data = {}
        remaining_data = self.als.unsplit_data.copy()

        for key in self.als.partitions_sizes:
            partition_size = self.als.partitions_sizes[key]
            data[key] = remaining_data.sample(n=partition_size,
                                              random_state=self.als.seed)

            remaining_data = remaining_data.drop(data[key].index)

        self.als.data = data

    def __set_partition_sizes(self):
        n_points_test = int(self.als.unsplit_data.shape[0] * self.als.pct_points_test)
        n_points_unlabeled = self.als.unsplit_data.shape[
                                 0] - self.als.n_points_labeled_keep - self.als.n_points_labeled_delete - n_points_test

        self.als.partitions_sizes = {
            'labeled_keep': self.als.n_points_labeled_keep,
            'labeled_delete': self.als.n_points_labeled_delete,
            'unlabeled': n_points_unlabeled,
            'unknown': n_points_test
        }

    # DATA FUNCTIONS:
    def get_labeled_data(self):
        if self.als.data['labeled_delete'].shape[0] == 0:
            return self.als.data['labeled_keep'].copy()
        else:
            labeled_data = pd.concat(
                [self.als.data['labeled_keep'], self.als.data['labeled_delete']],
                axis=0,
                ignore_index=False,
                sort=False)
            return pd.DataFrame(labeled_data.copy())

    def delete_data(self):
        self.als.data['deleted'] = self.als.data['labeled_delete'].copy()
        self.als.data['labeled_delete'] = pd.DataFrame()



    # PARTITION FUNCTIONS
    def get_partitions(self):
        return self.als.data.copy()

    def get_partition_sizes(self):
        return self.als.partitions_sizes.copy()

    # DATA TRANSFORMATION FUNCTIONS:
    def transform_data(self, use_pca, scale):
        '''
        scales data if scale == True
        pca transfroms data if use_pca == True
        '''
        if scale:
            # Fit scaler on known covariates only, not uknown testing data
            known_X = self.get_all_known_X()
            self.als.scaler.fit(known_X)
            self.scale_data()

        if use_pca:
            known_X = self.get_all_known_X()
            self.als.pca.fit(known_X) # TODO: add functionality to set n output components of PCA
            self.pca_transform_data()


    def get_all_known_X(self):
        X, y = get_X_y(self.get_labeled_data())
        X_unlabled, y_unlabeled = get_X_y(self.data['unlabeled'])
        all_known_X = pd.concat([X.copy(), X_unlabled.copy()],
                                axis=0,
                                ignore_index=True)

        return all_known_X

    def scale_data(self):
        for key in self.als.data:
            if self.als.data[key].shape[0] == 0:
                # cannot scale nonexistent data
                continue

            X, y = get_X_y(self.als.data[key])
            X = self.als.scaler.transform(X)
            X = pd.DataFrame(X.copy())

            label_name = y.name
            labels = list(y.copy())

            X.insert(0, label_name, labels)
            self.data[key] = X

    def pca_transform_data(self):
        for key in self.als.data:
            if self.als.data[key].shape[0] == 0:
                continue
            X, y = get_X_y(self.als.data[key])
            X = self.als.pca.transform(X)
            X = pd.DataFrame(X.copy())
            label_name = y.name
            labels = list(y.copy())

            X.insert(0, label_name, labels)
            self.als.data[key] = pd.DataFrame()
            self.als.data[key] = X.copy()

