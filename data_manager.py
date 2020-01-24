



class Data_manager():
    """
    INPUT:
        dataframe with first column binary 0-1 class and rest of columns features
    OUTPUTS:
        various data that Simulator() needs
    """


    def __init__(self, unsplit_data, n_points_labeled_keep, n_points_labeled_delete, pct_points_test = 0.25):
        # self.unsplit_data = unsplit_data
        self.n_points_labeled_keep = n_points_labeled_keep
        self.n_points_labeled_delete = n_points_labeled_delete

        self.pct_points_test = pct_points_test # default 0.25

        self.data = Data_manager.create_partitions(unsplit_data)


    # PARTITIONING FUNCTIONS:
    def set_partitions(self):
        self.set_partition_sizes()

        data = {}
        remaining_data = self.unsplit_data.copy()

        for key in self.partitions_sizes:
            partition_size = self.partitions_sizes[key]
            data[key] = remaining_data.sample(n=partition_size,
                                              random_state=self.seed)

            remaining_data = remaining_data.drop(data[key].index)

        self.data = data

    def set_partition_sizes(self):
        n_points_test = int(self.unsplit_data.shape[0] * self.pct_points_test)
        n_points_unlabeled = self.unsplit_data.shape[
            0] - self.n_points_labeled_keep - self.n_points_labeled_delete - n_points_test

        self.partitions_sizes = {
            'labeled_keep': self.n_points_labeled_keep,
            'labeled_delete': self.n_points_labeled_delete,
            'unlabeled': n_points_unlabeled,
            'unknown': n_points_test
        }

    # DATA FUNCTIONS:
    def get_labeled_data(self):
        if self.data['labeled_delete'].shape[0] == 0:
            return self.data['labeled_keep'].copy()
        else:
            labeled_data = pd.concat(
                [self.data['labeled_keep'], self.data['labeled_delete']],
                axis=0,
                ignore_index=False,
                sort=False)
            return pd.DataFrame(labeled_data.copy())

    def delete_data(self):
        self.data['deleted'] = self.data['labeled_delete'].copy()
        self.data['labeled_delete'] = pd.DataFrame()

    def get_X_y(df):
        '''
        Returns first column of df as y and rest as X
        '''
        X = df.drop(df.columns[0], axis=1)
        y = df.iloc[:, 0]

        return X, y