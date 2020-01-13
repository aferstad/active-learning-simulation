import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from scipy import spatial # for nearest neighbour https://codegolf.stackexchange.com/questions/45611/fastest-array-by-array-nearest-neighbour-search

# TODO: measure and plot the accuracy and LOSS at each new point, comparing random and other methods

class Experiment():
    pct_unlabeled_to_label = 1.00
    n_points_to_add_at_a_time = 1
    pct_points_test = 0.25
    accuracies = []
    consistencies = []
    similar_method_initiated = False
    similiar_method_closest_unlabeled_rows = pd.DataFrame()

    def __init__(self, unsplit_data, seed, model_type, n_points_labeled_keep, n_points_labeled_delete):
        '''
        input: unsplit prepared data and possibility to change defaults
        desc: sets partitions
        '''
        self.accuracies = []
        self.consistencies = []
        self.similar_method_initiated = False
        self.similiar_method_closest_unlabeled_rows = pd.DataFrame()

        self.seed = seed
        self.unsplit_data = unsplit_data
        self.model_type = model_type
        self.n_points_labeled_keep = n_points_labeled_keep
        self.n_points_labeled_delete = n_points_labeled_delete

        self.set_partitions()

    # DATA METHODS:
    def set_partition_sizes(self):
        n_points_test = int(self.unsplit_data.shape[0] * self.pct_points_test) # int floors numbers
        n_points_unlabeled = self.unsplit_data.shape[0] - self.n_points_labeled_keep - self.n_points_labeled_delete - n_points_test

        self.partitions_sizes = {
            'labeled_keep' : self.n_points_labeled_keep,
            'labeled_delete' : self.n_points_labeled_delete,
            'unlabeled' : n_points_unlabeled,
            'unknown' : n_points_test
        }

    def get_partition_sizes(self):
        return self.partitions_sizes.copy()

    def set_partitions(self):
        self.set_partition_sizes()

        data = {}
        remaining_data = self.unsplit_data.copy()

        for key in self.partitions_sizes:
            partition_size = self.partitions_sizes[key]
            data[key] = remaining_data.sample(n=partition_size, random_state = self.seed)

            remaining_data = remaining_data.drop(data[key].index)

        self.data = data

    def get_partitions(self):
        return self.data.copy()

    def get_labeled_data(self):
        return pd.concat(
            [self.data['labeled_keep'], self.data['labeled_delete']], axis=0)

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

    # MODEL METHODS:
    def KNN_cv(X, y, print_results = False):
        '''
        finds best K for KNN
        '''
        k_scores = []
        k_ceiling = int(X.shape[0] * 0.8) # when do 5 fold cross validation max number of neighbors is 80% of data points
        k_ceiling = min(100, k_ceiling) # TODO: find out best way to maximize K
        k_range = range(3, k_ceiling)

        # use iteration to caclulator different k in models, then return the average accuracy based on the cross validation
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

        #best_k = k_scores.index((max(k_scores))) + 1
        best_k = max(np.array(k_range)[max(k_scores) <= k_scores + np.std(k_scores)])

        return best_k

    def compare_models(m1, m2, data):
        m1_accuracy = Experiment.get_model_accuracy(m1, data)
        m2_accuarcy = Experiment.get_model_accuracy(m2, data)

        return m1_accuracy, m2_accuarcy

    def get_model_accuracy(m, data):
        X, y = Experiment.get_X_y(data)
        y_pred = m.predict(X)

        #print('ACCURACY')
        #print(sum(y_pred == y) / len(y))
        return sum(y_pred == y) / len(y)

    def get_model_consistency(self, m):
        '''
        compares model m with initial model
        '''

        X, y = Experiment.get_X_y(self.data['unknown'])
        y_pred_initial = self.model_initial.predict(X)
        y_pred_current = m.predict(X)

        #print('CONSITENCY')
        #print(sum(y_pred_initial == y_pred_current) / len(y_pred_initial))
        return sum(y_pred_initial == y_pred_current) / len(y_pred_initial)

    def fit_model(self):
        labeled = self.get_labeled_data()
        X, y = Experiment.get_X_y(labeled)

        if self.model_type == 'KNN':
            best_k = Experiment.KNN_cv(X, y)
            model = KNeighborsClassifier(n_neighbors=best_k)
            model.fit(X, y)
        elif self.model_type == 'lr':
            model = LogisticRegression(solver='liblinear', max_iter=1000) # can add random state here, can also change parameters
            model.fit(X, y)

        return model

    # ACTIVE LEARNING METHODS:
    def get_rows_to_add(self, method):
        if method == 'random':
            random_rows = self.data['unlabeled'].sample(n=self.n_points_to_add_at_a_time, random_state = self.seed)
            return random_rows

        elif method == 'uncertainty':

            return self.get_most_uncertain_rows(self.data['unlabeled'])

        elif method == 'similar':
            if not self.similar_method_initiated:
                self.initiate_similar_method()
            if self.similiar_method_closest_unlabeled_rows.shape[0] == 0:
                return self.get_rows_to_add('uncertainty')

            most_uncertain_similar_rows = self.get_most_uncertain_rows(self.similiar_method_closest_unlabeled_rows)

            #print('Shape')
            #print(self.similiar_method_closest_unlabeled_rows.shape[0])

            self.similiar_method_closest_unlabeled_rows.drop(most_uncertain_similar_rows.index, inplace = True)

            return most_uncertain_similar_rows


    def initiate_similar_method(self):
        # if there are no more deleted points
        if self.data['deleted'].shape[0] == 0:
            #print('WARNING: Method is set to *similar*, but no data has been deleted')
            #print('--> defaulting back to method *uncertainty*')
            return self.get_rows_to_add('uncertainty')


        # For each deleted point, find get the index of the nearest unlabeled neighbour to that deleted point:
        # NOTE: the output indexes are 0,1,2.. not the original indexes of the rows of unlabeled_data

        #print("self.data['deleted']")
        #print(self.data['deleted'])

        #print("self.data['unlabeled']")
        #print(self.data['unlabeled'])

        closest_row_indexes = Experiment.nearest_neighbour(self.data['deleted'], self.data['unlabeled'])

        # TODO: Decide whether to use commented out code below:
            # Then count whether any unlabeled point is the closest to more than one deleted point
            #popularity = closest_rows.value_counts().reset_index()
            #popularity.columns = ['unlabeled_row_index', 'cnt']
            #popularity.sort_values('cnt',inplace = True, ascending = False)
            # Select the unlabeled points that have the most deleted neighbours
            #most_popular = popularity[popularity.cnt == max(popularity.cnt)]

        unique_indexes = list(closest_row_indexes.unique())

        closest_rows = []
        for index in unique_indexes:
            row = self.data['unlabeled'].iloc[index]
            closest_rows.append(row)

        closest_rows = pd.DataFrame(closest_rows)

        self.similar_method_initiated = True
        self.similiar_method_closest_unlabeled_rows = closest_rows

    def get_most_uncertain_rows(self, rows):
        X, y = Experiment.get_X_y(rows)

        proba_class_1 = self.model_current.predict_proba(X)[:,1]
        #gives the length from probability 0.5, more length means more certainty
        X['class_certainty'] = np.abs(proba_class_1 - 0.5)
        #print(X_unlabled.class_certainty.sort_values().index[:1])
        most_uncertain_rows_indexes = X.class_certainty.sort_values().index[:self.n_points_to_add_at_a_time]

        most_uncertain_rows = rows.loc[most_uncertain_rows_indexes,:]
        return most_uncertain_rows

    def nearest_neighbour(points_a, points_b):
        #print('points_a')
        #print(points_a)

        #print('points_a')
        #print(points_b)


        tree = spatial.cKDTree(points_b)
        return pd.Series(tree.query(points_a)[1])

    def label_new_points(self, method):
        '''
        Ouputs labeled + newly_labeled chosen by method
        '''
        self.model_current = self.fit_model()

        n_points_to_add = int(self.data['unlabeled'].shape[0] * self.pct_unlabeled_to_label)
        n_points_added = 0

        while n_points_added < n_points_to_add:
            #print('Percentage points added: ' + str(round(100.0 * n_points_added / n_points_to_add)))

            n_points_added += self.n_points_to_add_at_a_time

            rows_to_add = self.get_rows_to_add(method) # TODO: fix this function

            self.data['unlabeled'] = self.data['unlabeled'].drop(rows_to_add.index)
            self.data['labeled_keep'] = self.data['labeled_keep'].append(rows_to_add)

            self.model_current = self.fit_model()
            self.accuracies.append(Experiment.get_model_accuracy(self.model_current, self.data['unknown']))
            self.consistencies.append(self.get_model_consistency(self.model_current))

            if self.model_type == 'KNN':
                self.model_parameters.append(self.model_current.n_neighbors)

    def run_experiment(self,
                   method='random'):
        '''
        input raw data, output initial and final model accuracy
        '''

        labeled = self.get_labeled_data()
        if len(labeled.iloc[:,0].unique()) != 2:
            print('Error: initial labeled data only contains one class')
            return 0, None


        self.model_initial = self.fit_model()

        self.accuracies = []
        self.consistencies = []

        self.accuracies.append(Experiment.get_model_accuracy(self.model_initial, self.data['unknown']))

        self.delete_data()


        self.model_parameters = []
        if self.model_type == 'KNN':
            self.model_parameters.append(self.model_initial.n_neighbors)

        # decide all new points to label:
        self.label_new_points(method)

        self.model_final = self.fit_model()

        self.model_initial_accuracy, self.model_final_accuracy = Experiment.compare_models(
            self.model_initial, self.model_final, self.data['unknown'])

        return self.model_initial_accuracy, self.model_final_accuracy
