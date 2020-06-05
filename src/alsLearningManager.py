import alsModelManager
import alsDataManager

import numpy as np
import pandas as pd
from scipy.stats import entropy


class AlsLearningManager:

    def __init__(self, als):
        self.als = als
        self.n_als_performed = None
        self.n_als_to_perform = None

    # ACTIVE LEARNING FUNCTIONS
    def run_experiment(self, n_als_performed, n_als_to_perform):
        """
        Runs experiments with the parameters specified when initializing the als object
        """
        self.als.accuracies = []
        self.als.consistencies = []
        self.n_als_performed = n_als_performed
        self.n_als_to_perform = n_als_to_perform

        self.als.model_initial = self.als.modelManager.fit_model(with_tuning=True)  # currently only xgboost that tunes
        self.als.accuracies.append(
            alsModelManager.get_model_accuracy(self.als.model_initial, self.als.data['unknown']))
        self.als.dataManager.delete_data()

        # decide all new points to label:
        self._label_new_points()

        # self.model_final = self.fit_model()

        # self.model_initial_accuracy, self.model_final_accuracy = ALS.compare_models(
        #    self.model_initial, self.model_final, self.data['unknown'])

    def _label_new_points(self):
        """
        called by run_experiment to manage the process of labeling points and refitting models
        """
        self.als.model_current = self.als.modelManager.fit_model(with_tuning=True) # currently only xgboost that tunes
        self.als.accuracies.append(
            alsModelManager.get_model_accuracy(self.als.model_current,
                                               self.als.data['unknown']))

        n_points_to_add = int(self.als.data['unlabeled'].shape[0] *
                              self.als.pct_unlabeled_to_label)
        n_points_added = 0
        n_third_complete = 1

        while n_points_added + self.als.n_points_to_add_at_a_time < n_points_to_add:
            pct_complete = round(100.0 * n_points_added / n_points_to_add)
            if pct_complete // 25 >= n_third_complete:
                pct_grid_complete = round(100. * (self.n_als_performed + pct_complete/100) / self.n_als_to_perform)
                print('### PCT COMPLETE: ' + str(pct_grid_complete) + '% ###')
                #print('[Current learning_method ' + self.als.learning_method + '] [pct complete: ' +
                #      str(pct_complete) + '%]')
                n_third_complete += 1

            n_points_added += self.als.n_points_to_add_at_a_time

            # MOST IMPORTANT FUNCTION CALLED IN LOOP:
            rows_to_add = self.get_rows_to_add()

            self.als.data['unlabeled'] = self.als.data['unlabeled'].drop(
                rows_to_add.index)
            self.als.data['labeled_keep'] = self.als.data['labeled_keep'].append(
                rows_to_add)

            # TODO: decide when to tune xgboost
            if False:  # n_points_added == self.als.n_points_labeled_delete or n_points_added + 1 == n_points_to_add:
                self.als.model_current = self.als.modelManager.fit_model(with_tuning=True)
            else:
                self.als.model_current = self.als.modelManager.fit_model(with_tuning=False)

            self.als.accuracies.append(
                alsModelManager.get_model_accuracy(self.als.model_current,
                                                   self.als.data['unknown']))
            self.als.consistencies.append(
                self.als.modelManager.get_model_consistency(self.als.model_current))

    def get_rows_to_add(self, learning_method=None):
        """
        :param learning_method: if None, set to self.learning_method
        :return: next rows to add, according to learning_method
        """

        if learning_method is None:
            learning_method = self.als.learning_method

        if learning_method == 'random' or learning_method == 'bayesian_random':
            random_rows = self.als.data['unlabeled'].sample(
                n=self.als.n_points_to_add_at_a_time, random_state=self.als.seed)
            return random_rows

        elif learning_method == 'uncertainty':
            return self.get_most_uncertain_rows(self.als.data['unlabeled'])

        elif learning_method == 'similar':
            if not self.als.similar_learning_method_initiated:
                self.initiate_similar_learning_method()
            if self.als.similar_learning_method_closest_unlabeled_rows.shape[0] == 0:
                return self.get_rows_to_add(learning_method='uncertainty')

            most_uncertain_similar_rows = self.get_most_uncertain_rows(
                self.als.similar_learning_method_closest_unlabeled_rows)

            self.als.similar_learning_method_closest_unlabeled_rows.drop(
                most_uncertain_similar_rows.index, inplace=True)
            return most_uncertain_similar_rows
        elif learning_method == 'similar_uncertainty_optimization':
            # TODO: Move the code below into its own function to decrease clutter
            if not self.als.similar_learning_method_initiated:
                self.initiate_similar_learning_method()
            if self.als.similar_learning_method_closest_unlabeled_rows.shape[0] == 0:
                return self.get_rows_to_add(learning_method='uncertainty')

            most_uncertain_similar_rows = self.get_most_uncertain_rows(
                self.als.similar_learning_method_closest_unlabeled_rows)
            most_uncertain_rows = self.get_most_uncertain_rows(
                self.als.data['unlabeled'])

            if self.als.n_points_to_add_at_a_time != 1:
                print(
                    'Error: similar_uncertainty_ratio not possible to calculate when n_points_to_add_at_a_time != 1'
                )
                return None

            max_uncertainty_of_similar = self.als.modelManager.get_point_certainty(
                most_uncertain_similar_rows)
            self.als.similar_uncertainties.append(max_uncertainty_of_similar)

            max_uncertainty = self.als.modelManager.get_point_certainty(most_uncertain_rows)
            self.als.max_uncertainties.append(max_uncertainty)

            certainty_ratio = max_uncertainty_of_similar / max_uncertainty
            # print('RATIO: ' + str(certainty_ratio))
            # row = [max_uncertainty_of_similar, max_uncertainty, certainty_ratio]
            # self.certainties.append(row)

            if certainty_ratio >= self.als.certainty_ratio_threshold:
                return most_uncertain_rows
            else:
                self.als.similar_learning_method_closest_unlabeled_rows.drop(
                    most_uncertain_similar_rows.index, inplace=True)
                return most_uncertain_similar_rows

    # learning_method SPECIFIC FUNCTIONS
    def initiate_similar_learning_method(self):
        # NOTE: Comments below is code used for popularity vote, no longer used

        # if there are no more deleted points
        if self.als.data['deleted'].shape[0] == 0:
            # print('WARNING: learning_method is set to *similar*, but no data has been deleted')
            # print('--> defaulting back to learning_method *uncertainty*')
            return self.get_rows_to_add(learning_method='uncertainty')

        # For each deleted point, find get the index of the nearest unlabeled neighbour to that deleted point:
        # NOTE: the output indexes are 0,1,2.. not the original indexes of the rows of unlabeled_data

        # print("self.data['deleted']")
        # print(self.data['deleted'])

        # print("self.data['unlabeled']")
        # print(self.data['unlabeled'])

        closest_row_indexes = alsDataManager.nearest_neighbour(self.als.data['deleted'], self.als.data['unlabeled'])

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
            row = self.als.data['unlabeled'].iloc[index]
            closest_rows.append(row)

        closest_rows = pd.DataFrame(closest_rows)

        self.als.similar_learning_method_initiated = True
        self.als.similar_learning_method_closest_unlabeled_rows = closest_rows

    def get_most_uncertain_rows(self, rows, entropy_based = True, return_uncertainty = False):
        """
        :param rows: get most uncertain rows from rows
        :param entropy_based: whether to use entropy or max_probability
        :return_uncertainty: if true return uncertainty instead of row
        :return: the n_points_to_add_at_a_time number of most uncertain rows
        """
        X, y = alsDataManager.get_X_y(rows)

        probas = self.als.model_current.predict_proba(X)

        if entropy_based:
            # the point with the maximum entropy is the least certain
            # base = n_classes makes entropy be bound in the interval 0 to 1
            entropy_of_each_point = entropy(probas, axis = 1, base = self.als.n_classes)
            max_uncertainty = max(entropy_of_each_point)
            min_class_certainty_index = entropy_of_each_point.argmax()
        else:
            class_certainty = probas.max(1)  # max(1) gives max of each row
            max_uncertainty = min(class_certainty)
            min_class_certainty_index = class_certainty.argmin()  # gives index of min element

        most_uncertain_row = rows.iloc[min_class_certainty_index,:]

        if return_uncertainty:
            return max_uncertainty
        else:
            return pd.DataFrame(most_uncertain_row).transpose()  # convert series row to dataframe to allow for input in model later

    def get_performance_results(self):
        """
        :return: dict with keys as metric_strs and values as a list of that metric per learning step
        """
        results = {}
        results['accuracy'] = self.als.accuracies
        results['consistencies'] = self.als.consistencies
        results['similar_uncertainties'] = self.als.similar_uncertainties
        results['max_uncertainties'] = self.als.max_uncertainties

        return results



