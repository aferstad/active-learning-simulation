import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

class Experiment():
    # TODO: measure and plot the accuracy and LOSS at each new point, comparing random and other methods
    Experiment.seed = 1
    Experiment.pct_unlabeled_to_label = 1
    Experiment.n_points_to_add_at_a_time = 1
    Experiment.n_initial_points_labeled = 50

    def __init__(self, unsplit_data, seed = Experiment.seed, pct_unlabeled_to_label = Experiment.pct_unlabeled_to_label, n_points_to_add_at_a_time = Experiment.n_points_to_add_at_a_time):
        '''
        input: unsplit prepared data and possibility to change defaults
        desc: sets partitions
        '''
        # sets to default if nothing else given:
        self.seed = seed
        self.pct_unlabeled_to_label = pct_unlabeled_to_label
        self.n_points_to_add_at_a_time = n_points_to_add_at_a_time

        self.unsplit_data = unsplit_data

        self.set_partition_sizes()
        self.set_partitions()


    def set_partition_sizes(self, n_points_labeled_keep = 50, n_points_labeled_delete = 0, pct_points_test = 0.25):
        n_points_test = int(self.unsplit_data.shape[0] * pct_points_test) # int floors numbers
        n_points_unlabeled = self.unsplit_data.shape[0] - n_points_labeled_keep - n_points_labeled_delete - n_points_test

        self.partitions_sizes = {
            'labeled_keep' : n_points_labeled_keep,
            'labeled_delete' : n_points_labeled_delete,
            'unlabeled' : n_points_unlabeled,
            'unknown' : n_points_test
        }

    def get_partition_sizes(self):
        return self.partitions_sizes.copy()

    def set_partitions(self):
        data = {}
        remaining_data = self.unsplit_data.copy()

        for key in self.partitions_sizes:
            partition_size = self.partitions_sizes[key]
            data[key] = remaining_data.sample(n=partition_size, random_state = self.seed)

            remaining_data = remaining_data.drop(data[key].index)

        self.data = data

    def get_partitions(self):
        return self.data.copy()

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

        return accuracy_score(y, y_pred)

    def get_X_y(df):
        '''
        Returns first column of df as y and rest as X
        '''
        X = df.drop(df.columns[0], axis=1)
        y = df.iloc[:, 0]

        return X, y

    def random_active_learning(data, pct_unlabeled_to_label, seed):
        '''
        Input: split data
        Output: labeled + newly_labeled chosen randomly
        '''
        n_points_to_add = int(data['unlabeled'].shape[0] * pct_unlabeled_to_label)

        unlabeled = data['unlabeled'].copy()
        labeled = pd.concat([data['labeled_keep'], data['labeled_delete']], axis=0)

        for i in range(n_points_to_add):
            new_point

        newly_labeled = unlabeled.sample(n=n_points_to_add, random_state = seed)
        remaining_unlabeled = unlabeled.drop(newly_labeled.index)
        labeled = labeled.append(newly_labeled)

        return labeled



    def uncertainty_active_learning(self):
        '''
        Ouputs labeled + newly_labeled chosen by uncertainty
        '''
        labeled_current = self.labeled_initial.copy()
        unlabeled = self.data['unlabeled'].copy()

        model_current = KNeighborsClassifier(n_neighbors=self.best_k_initial)
        model_current.fit(self.X_initial, self.y_initial)

        n_points_to_add = int(np.floor(unlabeled.shape[0] * self.pct_to_label))
        n_points_added = 0

        while n_points_added < n_points_to_add:
            print('Percentage points added: ' + str(round(100.0 * n_points_added / n_points_to_add)))

            n_points_added += self.n_points_to_add_at_a_time

            X_unlabled, y_unlabeled = Experiment.get_X_y(unlabeled)

            indexes_to_add = self.get_indexes_to_add(self.n_points_to_add_at_a_time, method = 'random')

            proba_class_1 = model_current.predict_proba(X_unlabled)[:,1]

            #gives the length from probability 0.5, more length means more certainty
            X_unlabled['class_certainty'] = np.abs(proba_class_1 - 0.5)

            most_uncertain_rows_indexes = X_unlabled.class_certainty.sort_values().index[:self.n_points_to_add_at_a_time]

            #most_uncertain_row_index = X_unlabled.class_certainty.idxmin()

            rows_to_add = unlabeled.loc[most_uncertain_rows_indexes,:]


            #newly_labeled = newly_labeled.append(rows_to_add)
            unlabeled = unlabeled.drop(most_uncertain_rows_indexes)
            labeled_current = labeled_current.append(rows_to_add)

            X_current, y_current = Experiment.get_X_y(labeled_current)

            model_current.fit(X_current, y_current)

        return labeled_current

    def get_rows_to_add(self, method):
        if method == 'random':
            random_rows = self.data['unlabeled'].sample(n=self.n_points_to_add_at_a_time, random_state = self.seed)
            return random_rows

        elif method == 'uncertainty':
            proba_class_1 = self.model_current.predict_proba(self.X_unlabled)[:,1]
            #gives the length from probability 0.5, more length means more certainty
            self.X_unlabled['class_certainty'] = np.abs(proba_class_1 - 0.5)
            most_uncertain_rows_indexes = self.X_unlabled.class_certainty.sort_values().index[:self.n_points_to_add_at_a_time]

            most_uncertain_rows = self.data['unlabeled'].loc[most_uncertain_rows_indexes,:]

            return most_uncertain_rows

    def label_new_points(self, method):
        '''
        Ouputs labeled + newly_labeled chosen by method
        '''
        self.model_current = Experiment.fit_model(self.data)

        n_points_to_add = int(self.data['unlabeled'].shape[0] * self.pct_unlabeled_to_label)
        n_points_added = 0

        while n_points_added < n_points_to_add:
            print('Percentage points added: ' + str(round(100.0 * n_points_added / n_points_to_add)))

            n_points_added += self.n_points_to_add_at_a_time

            rows_to_add = self.get_rows_to_add(method) # TODO: fix this function

            self.data['unlabeled'] = self.data['unlabeled'].drop(rows_to_add.index)
            self.data['labeled_keep'] = self.data['labeled_keep'].append(rows_to_add)

            self.model_current = Experiment.fit_model(self.data)
            self.model_current.n_neighbors
            self.accuracies.append(Experiment.get_model_accuracy(self.model_current, self.data['unknown']))
            self.best_ks.append(self.model_current.n_neighbors)


    def fit_model(data):
        labeled = pd.concat(
            [data['labeled_keep'], data['labeled_delete']], axis=0)
        X, y = Experiment.get_X_y(labeled)
        best_k = Experiment.KNN_cv(X, y)
        model = KNeighborsClassifier(n_neighbors=best_k)
        model.fit(X, y)

        return model

    def run_experiment(self,
                   method='random'):
        '''
        input raw data, output initial and final model accuracy
        '''
        self.model_initial = Experiment.fit_model(self.data)

        self.accuracies = []
        self.accuracies.append(Experiment.get_model_accuracy(self.model_initial, self.data['unknown']))

        self.best_ks = []
        self.best_ks.append(self.model_initial.n_neighbors)

        # decide all new points to label:
        self.label_new_points(method)

        self.model_final = Experiment.fit_model(self.data)

        self.model_initial_accuracy, self.model_final_accuracy = Experiment.compare_models(
            self.model_initial, self.model_final, self.data['unknown'])

        return self.model_initial_accuracy, self.model_final_accuracy


# OLD SCRAPS
'''
    FRACTIONS = {
    'labeled_keep': 0.005, #data to keep
    'labeled_delete': 0.00, #data to delete
    'unlabeled': 0.695, #data to get labeled points from
    'unknown': 0.3 #used to test initial and final model differences
    }


def divide_data(unsplit_data, by_fractions = False, fractions = FRACTIONS, seed = SEED):
'''
    #divides data into labeled_keep, labeled_delete, unlabeled and unknown
'''
    data = {}
    n_rows = unsplit_data.shape[0]
    remaining = unsplit_data.copy()

    if by_fractions:
        for key in fractions:
            n_rows_key = int(np.floor(fractions[key] * n_rows))
            data[key] = remaining.sample(n=n_rows_key, random_state = seed)

            remaining = remaining.drop(data[key].index)
    else:
        print('none')
'''
