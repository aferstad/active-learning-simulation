
# ACTIVE LEARNING SIMULATOR
# author: Andreas Ferstad

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bayesianLogisticRegression import BayesianLogisticRegression # my custom model

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from scipy import spatial  # for nearest neighbour
import pymc3 as pm # for bayesian models


# TODO: CREATE TESTS


class ALS:

    def __init__(self,
                 unsplit_data,
                 learning_method,
                 model_type='lr',
                 seed=0,
                 n_points_labeled_keep=25,
                 n_points_labeled_delete=25,
                 use_pca=False,
                 scale=False,
                 n_points_to_add_at_a_time=1,
                 certainty_ratio_threshold=2,
                 pct_unlabeled_to_label=1.00,
                 pct_points_test=0.25,
                 cores = 4):
        """
        input: unsplit prepared data, learning_method, and possibility to change default parameters
        """

        # TODO: Decrease size of init method by moving some processes away

        # Learning Parameters
        self.learning_method = learning_method
        if learning_method == 'bayesian_random' and model_type != 'lr':
            print('bayesian_random code not supported for other models than lr. Setting model to lr...')
            self.model_type = 'lr'
        self.model_type = model_type

        self.n_points_to_add_at_a_time = n_points_to_add_at_a_time
        #if learning_method == 'bayesian_random' and n_points_to_add_at_a_time < 25:
        #    print('setting n_points_to_add_at_a_time = 25, remove code in ALS to avoid this')
        #    self.n_points_to_add_at_a_time = 25
        self.certainty_ratio_threshold = certainty_ratio_threshold
        self.pct_unlabeled_to_label = pct_unlabeled_to_label

        self.similar_learning_method_initiated = False
        self.similar_learning_method_closest_unlabeled_rows = pd.DataFrame()

        # Data Parameters
        self.unsplit_data = unsplit_data
        self.seed = seed
        self.n_points_labeled_keep = n_points_labeled_keep
        self.n_points_labeled_delete = n_points_labeled_delete
        self.use_pca = use_pca
        self.scale = scale
        if learning_method == 'bayesian_random':
            self.scale = True

        #if learning_method == 'bayesian_random':
        #    self.add_intercept_column = True
        #else:
        #    self.add_intercept_column = False
        self.add_intercept_column = False

        self.pct_points_test = pct_points_test

        self.scaler = StandardScaler()
        self.pca = PCA()
        self.data = {}

        self.set_partitions()
        self.transform_data(use_pca, scale, self.add_intercept_column)
        # TODO: decide whether I always should scale data?

        # Performance Metrics
        self.accuracies = []
        self.consistencies = []
        self.similar_uncertainties = []
        self.max_uncertainties = []
        self.certainties = []

        # Model Parameters
        labeled = self.get_labeled_data()

        if len(labeled.iloc[:, 0].unique()) != 2:
            print('Error: initial labeled data only contains one class')

        #self.bayesian_model_initialized = False
        #self.bayesian_model = pm.Model()
        #self.trace = []
        self.traces = []
        self.latest_trace = None
        self.cores = cores
        self.model_initial = None

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

    # MODEL FUNCTIONS:
    def fit_model(self):
        labeled = self.get_labeled_data()
        X, y = ALS.get_X_y(labeled)

        if self.learning_method == 'bayesian_random':
            model = BayesianLogisticRegression()
            # self.latest_trace = model.fit(X, y, previous_trace = self.latest_trace, cores = self.cores) # returns trace when fitting
            # self.traces.append(self.latest_trace)
            # TODO: verify that the fitting below works
            if self.model_initial is not None:
                model.fit(X,
                          y,
                          prior_trace=self.model_initial.trace,
                          cores=self.cores,
                          prior_index=self.model_initial.training_data_index)
            else:
                model.fit(X,
                          y,
                          cores=self.cores)
        elif self.model_type == 'KNN':
            best_k = ALS.KNN_cv(X, y)
            model = KNeighborsClassifier(n_neighbors=best_k)
            model.fit(X, y)
        elif self.model_type == 'lr':
            model = LogisticRegression(
                solver='liblinear', max_iter=1000
            )  # can add random state here, can also change parameters
            model.fit(X, y)

        return model

    def get_model_accuracy(m, data):
        X, y = ALS.get_X_y(data)
        y_pred = m.predict(X)
        return sum(y_pred == y) / len(y)

    def get_model_consistency(self, m):
        '''
      compares model m with initial model
      '''

        X, y = ALS.get_X_y(self.data['unknown'])
        y_pred_initial = self.model_initial.predict(X)
        y_pred_current = m.predict(X)

        return sum(y_pred_initial == y_pred_current) / len(y_pred_initial)

    def get_point_certainty(self, rows):
        X, y = ALS.get_X_y(rows)
        proba_class_1 = self.model_current.predict_proba(X)[:, 1]
        class_certainty = np.abs(proba_class_1 - 0.5)[0] / 0.5  # NOTE: assumes only one element in rows
        return class_certainty

    def get_certainties(self):
        df = pd.DataFrame()
        df.insert(0, 'min_certainty', self.max_uncertainties)
        df.insert(0, 'min_certainty_of_similar', self.similiar_uncertainties)

        return df

    # ACTIVE LEARNING FUNCTIONS
    def run_experiment(self):
        """
        Runs experiments with the parameters specified when initializing the object
        """
        self.accuracies = []
        self.consistencies = []

        self.model_initial = self.fit_model()
        self.accuracies.append(
            ALS.get_model_accuracy(self.model_initial, self.data['unknown']))
        self.delete_data()

        self.model_parameters = []
        if self.model_type == 'KNN':
            self.model_parameters.append(self.model_initial.n_neighbors)

        # decide all new points to label:
        self._label_new_points()

        #self.model_final = self.fit_model()

       #self.model_initial_accuracy, self.model_final_accuracy = ALS.compare_models(
        #    self.model_initial, self.model_final, self.data['unknown'])

    def _label_new_points(self):
        """
        called by run_experiment to manage the process of labeling points and refitting models
        """
        self.model_current = self.fit_model()
        self.accuracies.append(
            ALS.get_model_accuracy(self.model_current,
                                   self.data['unknown']))

        n_points_to_add = int(self.data['unlabeled'].shape[0] *
                              self.pct_unlabeled_to_label)
        n_points_added = 0
        n_quartile_complete = 0

        while n_points_added + self.n_points_to_add_at_a_time < n_points_to_add:
            pct_complete = round(100.0 * n_points_added / n_points_to_add)
            if pct_complete // 25 >= n_quartile_complete:
                print('[Current learning_method ' + self.learning_method + '] [pct complete: ' +
                      str(pct_complete) + '%]')
                n_quartile_complete += 1

            n_points_added += self.n_points_to_add_at_a_time

            # MOST IMPORTANT FUNCTION CALLED IN LOOP:
            rows_to_add = self.get_rows_to_add()

            self.data['unlabeled'] = self.data['unlabeled'].drop(
                rows_to_add.index)
            self.data['labeled_keep'] = self.data['labeled_keep'].append(
                rows_to_add)

            self.model_current = self.fit_model()
            self.accuracies.append(
                ALS.get_model_accuracy(self.model_current,
                                       self.data['unknown']))
            self.consistencies.append(
                self.get_model_consistency(self.model_current))

            if self.model_type == 'KNN':
                self.model_parameters.append(self.model_current.n_neighbors)

    def get_rows_to_add(self):
        if self.learning_method == 'random' or self.learning_method == 'bayesian_random':
            random_rows = self.data['unlabeled'].sample(
                n=self.n_points_to_add_at_a_time, random_state=self.seed)
            return random_rows

        elif self.learning_method == 'uncertainty':
            return self.get_most_uncertain_rows(self.data['unlabeled'])

        elif self.learning_method == 'similar':
            if not self.similar_learning_method_initiated:
                self.initiate_similar_learning_method()
            if self.similiar_learning_method_closest_unlabeled_rows.shape[0] == 0:
                return self.get_rows_to_add('uncertainty')

            most_uncertain_similar_rows = self.get_most_uncertain_rows(
                self.similiar_learning_method_closest_unlabeled_rows)

            self.similiar_learning_method_closest_unlabeled_rows.drop(
                most_uncertain_similar_rows.index, inplace=True)
            return most_uncertain_similar_rows
        elif self.learning_method == 'similar_uncertainty_optimization':
            # TODO: Move the code below into its own function to decrease clutter
            if not self.similar_learning_method_initiated:
                self.initiate_similar_learning_method()
            if self.similiar_learning_method_closest_unlabeled_rows.shape[0] == 0:
                return self.get_rows_to_add('uncertainty')

            most_uncertain_similar_rows = self.get_most_uncertain_rows(
                self.similiar_learning_method_closest_unlabeled_rows)
            most_uncertain_rows = self.get_most_uncertain_rows(
                self.data['unlabeled'])

            if self.n_points_to_add_at_a_time != 1:
                print(
                    'Error: similiar_uncertainty_ratio not possible to calculate when n_points_to_add_at_a_time != 1'
                )
                return None

            max_uncertainty_of_similar = self.get_point_certainty(
                most_uncertain_similar_rows)
            self.similiar_uncertainties.append(max_uncertainty_of_similar)

            max_uncertainty = self.get_point_certainty(most_uncertain_rows)
            self.max_uncertainties.append(max_uncertainty)

            certainty_ratio = max_uncertainty_of_similar / max_uncertainty
            #print('RATIO: ' + str(certainty_ratio))
            #row = [max_uncertainty_of_similar, max_uncertainty, certainty_ratio]
            #self.certainties.append(row)

            if certainty_ratio >= self.certainty_ratio_threshold:
                return most_uncertain_rows
            else:
                self.similiar_learning_method_closest_unlabeled_rows.drop(
                    most_uncertain_similar_rows.index, inplace=True)
                return most_uncertain_similar_rows

    # learning_method SPECIFIC FUNCTIONS
    def initiate_similar_learning_method(self):
        # NOTE: Comments below is code used for popularity vote, no longer used

        # if there are no more deleted points
        if self.data['deleted'].shape[0] == 0:
            # print('WARNING: learning_method is set to *similar*, but no data has been deleted')
            # print('--> defaulting back to learning_method *uncertainty*')
            return self.get_rows_to_add('uncertainty')

        # For each deleted point, find get the index of the nearest unlabeled neighbour to that deleted point:
        # NOTE: the output indexes are 0,1,2.. not the original indexes of the rows of unlabeled_data

        # print("self.data['deleted']")
        # print(self.data['deleted'])

        # print("self.data['unlabeled']")
        # print(self.data['unlabeled'])

        closest_row_indexes = ALS.nearest_neighbour(self.data['deleted'],
                                                    self.data['unlabeled'])

        # TODO: Decide whether to use commented out code below:
        # Then count whether any unlabeled point is the closest to more than one deleted point
        # popularity = closest_rows.value_counts().reset_index()
        # popularity.columns = ['unlabeled_row_index', 'cnt']
        # popularity.sort_values('cnt',inplace = True, ascending = False)
        # Select the unlabeled points that have the most deleted neighbours
        # most_popular = popularity[popularity.cnt == max(popularity.cnt)]

        unique_indexes = list(closest_row_indexes.unique())

        closest_rows = []
        for index in unique_indexes:
            row = self.data['unlabeled'].iloc[index]
            closest_rows.append(row)

        closest_rows = pd.DataFrame(closest_rows)

        self.similar_learning_method_initiated = True
        self.similiar_learning_method_closest_unlabeled_rows = closest_rows

    def get_most_uncertain_rows(self, rows):
        '''
          returns n_points_to_add_at_a_time number of most uncertain rows
        '''

        X, y = ALS.get_X_y(rows)

        proba_class_1 = self.model_current.predict_proba(X)[:, 1]

        # gives the length from probability 0.5, more length means more certainty
        X['class_certainty'] = np.abs(proba_class_1 - 0.5)

        most_uncertain_rows_indexes = X.class_certainty.sort_values(
        ).index[:self.n_points_to_add_at_a_time]

        most_uncertain_rows = rows.loc[most_uncertain_rows_indexes, :]
        return most_uncertain_rows

    def nearest_neighbour(points_a, points_b):
        '''
        for each point in A, return the closes point in B
        '''
        tree = spatial.cKDTree(points_b)
        return pd.Series(tree.query(points_a)[1])

    # ------ UNUSED FUNCTIONS
    # MODEL learning_methodS:
    def KNN_cv(X, y, print_results=False):
        '''
      finds best K for KNN
      '''
        k_scores = []
        k_ceiling = int(
            X.shape[0] * 0.8
        )  # when do 5 fold cross validation max number of neighbors is 80% of data points
        k_ceiling = min(100,
                        k_ceiling)  # TODO: find out best way to maximize K
        k_range = range(3, k_ceiling)

        # use iteration to caclulator different k in models,
        # then return the average accuracy based on the cross validation
        for k in k_range:
            if print_results:
                print('k = ' + str(k))
            knn = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
            k_scores.append(scores.mean())

        if print_results:
            # plot to see clearly
            plt.plot(k_range, k_scores)
            plt.fill_between
            plt.xlabel('Value of K for KNN')
            plt.ylabel('Cross-Validated Accuracy')
            plt.show()

        best_k = max(
            np.array(k_range)[max(k_scores) <= k_scores + np.std(k_scores)])

        return best_k

    def compare_models(m1, m2, data):
        m1_accuracy = ALS.get_model_accuracy(m1, data)
        m2_accuracy = ALS.get_model_accuracy(m2, data)

        return m1_accuracy, m2_accuracy

    # PARTITION FUNCTIONS
    def get_partitions(self):
        return self.data.copy()

    def get_partition_sizes(self):
        return self.partitions_sizes.copy()

    # DATA TRANSFORMATION FUNCTIONS:
    def transform_data(self, use_pca, scale, add_intercept_column):
        '''
        scales data if scale == True
        pca transfroms data if use_pca == True
        '''
        if scale:
            # Fit scaler on known covariates only, not uknown testing data
            known_X = self.get_all_known_X()
            self.scaler.fit(known_X)
            self.scale_data()

        if use_pca:
            known_X = self.get_all_known_X()
            self.pca.fit(known_X) # TODO: add functionality to set n output components of PCA
            self.pca_transform_data()

        if add_intercept_column:
            for key in self.data:
                if self.data[key].shape[0] == 0:
                    continue
                self.data[key].insert(1, 'intercept', 1)

    def get_all_known_X(self):
        X, y = ALS.get_X_y(self.get_labeled_data())
        X_unlabled, y_unlabeled = ALS.get_X_y(self.data['unlabeled'])
        all_known_X = pd.concat([X.copy(), X_unlabled.copy()],
                                axis=0,
                                ignore_index=True)

        return all_known_X

    def scale_data(self):
        for key in self.data:
            if self.data[key].shape[0] == 0:
                # cannot scale nonexistent data
                continue

            X, y = ALS.get_X_y(self.data[key])
            X = self.scaler.transform(X)
            X = pd.DataFrame(X.copy())

            label_name = y.name
            labels = list(y.copy())

            X.insert(0, label_name, labels)
            self.data[key] = X

    def pca_transform_data(self):
        for key in self.data:
            if self.data[key].shape[0] == 0:
                continue
            X, y = ALS.get_X_y(self.data[key])
            X = self.pca.transform(X)
            X = pd.DataFrame(X.copy())
            label_name = y.name
            labels = list(y.copy())

            X.insert(0, label_name, labels)
            self.data[key] = pd.DataFrame()
            self.data[key] = X.copy()
