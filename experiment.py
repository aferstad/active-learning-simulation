import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

class Experiment():
    # CLASS DEFAULTS
    FRACTIONS = {
    'labeled_keep': 0.005,
    'labeled_delete': 0.00,
    'unlabeled': 0.695,
    'unknown': 0.3
    }
    SEED = 1
    PCT_TO_LABEL = 0.05
    N_POINTS_TO_ADD_AT_A_TIME = 1

    def __init__(self, unsplit_data, seed = SEED, pct_to_label = PCT_TO_LABEL, n_points_to_add_at_a_time = N_POINTS_TO_ADD_AT_A_TIME):
        # set to default if nothing else given:
        self.seed = seed
        self.pct_to_label = pct_to_label
        self.n_points_to_add_at_a_time = n_points_to_add_at_a_time

        self.unsplit_data = unsplit_data
        self.split_data = Experiment.divide_data(unsplit_data, seed = self.seed)

        self.labeled_initial = pd.concat(
            [self.split_data['labeled_keep'], self.split_data['labeled_delete']], axis=0)

        self.X_initial, self.y_initial = Experiment.get_X_y(self.labeled_initial)

    def divide_data(unsplit_data, fractions = FRACTIONS, seed = SEED):
        '''
        divides data into labeled_keep, labeled_delete, unlabeled and unknown
        '''
        split_data = {}
        n_rows = unsplit_data.shape[0]
        remaining = unsplit_data.copy()

        for key in fractions:
            n_rows_key = int(np.floor(fractions[key] * n_rows))
            split_data[key] = remaining.sample(n=n_rows_key, random_state = seed)

            remaining = remaining.drop(split_data[key].index)

        return split_data


    def KNN_cv(X, y, k_range = range(1, 30, 2), print_results = False):
        '''
        finds best K for KNN
        '''
        k_scores = []

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
            plt.xlabel('Value of K for KNN')
            plt.ylabel('Cross-Validated Accuracy')
            plt.show()

        best_k = k_scores.index((max(k_scores))) + 1

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

    def random_active_learning(split_data, pct_to_label, seed):
        '''
        Input: split data
        Output: labeled + newly_labeled chosen randomly
        '''
        n_points_to_add = int(np.floor(split_data['unlabeled'].shape[0] * pct_to_label))

        unlabeled = split_data['unlabeled'].copy()
        labeled = pd.concat([split_data['labeled_keep'], split_data['labeled_delete']], axis=0)

        newly_labeled = unlabeled.sample(n=n_points_to_add, random_state = seed)
        remaining_unlabeled = unlabeled.drop(newly_labeled.index)
        labeled = labeled.append(newly_labeled)

        return labeled

    def uncertainty_active_learning(self):
        '''
        Ouputs labeled + newly_labeled chosen by uncertainty
        '''

        labeled_current = self.labeled_initial.copy()
        unlabeled = self.split_data['unlabeled'].copy()

        model_current = KNeighborsClassifier(n_neighbors=self.best_k_initial)
        model_current.fit(self.X_initial, self.y_initial)

        n_points_to_add = int(np.floor(unlabeled.shape[0] * self.pct_to_label))
        n_points_added = 0

        while n_points_added < n_points_to_add:
            clear_output()
            print('Percentage points added: ' + str(round(100.0 * n_points_added / n_points_to_add)))

            n_points_added += self.n_points_to_add_at_a_time

            X_unlabled, y_unlabeled = Experiment.get_X_y(unlabeled)

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

    def run_experiment(self,
                   pct_to_label = PCT_TO_LABEL,
                   method='random',
                   fractions=FRACTIONS):
        '''
        input raw data, output initial and final model accuracy
        '''
        #split_data = divide_data(unsplit_data, fractions, seeds)

        #labeled_initial = pd.concat(
        #    [split_data['labeled_keep'], split_data['labeled_delete']], axis=0)
        #X, y = get_X_y(labeled_initial)

        self.best_k_initial = Experiment.KNN_cv(self.X_initial, self.y_initial)

        self.model_initial = KNeighborsClassifier(n_neighbors=self.best_k_initial)
        self.model_initial.fit(self.X_initial, self.y_initial)

        if method == 'random':
            self.labeled_final = Experiment.random_active_learning(self.split_data,
                                                   pct_to_label, self.seed)
        elif method == 'uncertainty':
            self.labeled_final = self.uncertainty_active_learning()

        self.X_final, self.y_final = Experiment.get_X_y(self.labeled_final)
        self.best_k_final = Experiment.KNN_cv(self.X_final, self.y_final)

        self.model_final = KNeighborsClassifier(n_neighbors=self.best_k_final)
        self.model_final.fit(self.X_final, self.y_final)

        self.model_initial_accuracy, self.model_final_accuracy = Experiment.compare_models(
            self.model_initial, self.model_final, self.split_data['unknown'])

        return self.model_initial_accuracy, self.model_final_accuracy
