import pandas as pd
import numpy as np
import alsDataManager
from models.xgBoostModel import XGBoostModel # custom XGBoostModel with tuning
from models.bayesianLogisticRegression import BayesianLogisticRegression  # my custom model
from sklearn.linear_model import LogisticRegression
from scipy.stats import entropy



def get_model_accuracy(m, data):
    """
    :param m: model to measure accuracy of
    :param data: data to predict labels of
    :return: percentage correctly labeled points
    """
    X, y = alsDataManager.get_X_y(data)
    y_pred = m.predict(X)
    return sum(y_pred == y) / len(y)


class AlsModelManager:

    def __init__(self, als):
        self.als = als

    def fit_model(self, with_tuning = False):
        """
        :return: model fitted on labeled data
        """
        labeled = self.als.dataManager.get_labeled_data()
        X, y = alsDataManager.get_X_y(labeled)

        if self.als.model_type == 'xgboost':
            model = XGBoostModel()
            model.fit(X, y, with_tuning=with_tuning)

        elif self.als.learning_method == 'bayesian_random':
            model = BayesianLogisticRegression()

            # self.latest_trace = model.fit(X, y, previous_trace = self.latest_trace, cores = self.cores) # returns trace when fitting
            # self.traces.append(self.latest_trace)
            # TODO: verify that the fitting below works
            if self.als.model_initial is not None:
                model.fit(X,
                          y,
                          prior_trace=self.als.model_initial.trace,
                          cores=self.als.cores,
                          prior_index=self.als.model_initial.training_data_index)
            else:
                model.fit(X,
                          y,
                          cores=self.als.cores)
        elif self.als.model_type == 'lr':
            model = LogisticRegression(
                solver='liblinear', max_iter=1000
            )  # can add random state here, can also change parameters
            model.fit(X, y)
        else:
            model = None
            print('Error in fit_model(): model_type or learning_learning no supported')

        return model

    def get_model_consistency(self, m):
        """
        :param m: model
        :return: consistency between model m and initial model
        """
        X, y = alsDataManager.get_X_y(self.als.data['unknown'])
        y_pred_initial = self.als.model_initial.predict(X)
        y_pred_current = m.predict(X)

        return sum(y_pred_initial == y_pred_current) / len(y_pred_initial)

    def get_point_certainty(self, rows, entropy_based = True):
        """
        :param rows: current_model predicts P(Y=1) for each row in rows
        :return: certainty per row
        """
        X, y = alsDataManager.get_X_y(rows)
        probas = self.als.model_current.predict_proba(X)

        if entropy_based:
            entropy_of_each_point = entropy(probas, axis = 1, base=2)
            return -1 * entropy_of_each_point[0]  # multiply by -1 because higher entropy means less certainty
        else:
            class_certainty = probas.max(1)  # max(1) gives max of each row
            return class_certainty[0]  # assume only one input row, make list to allow for json saving later


        # if probas.shape[1] == 2:
        #    proba_class_1 = probas[:, 1]
        # elif probas.shape[1] > 2:
        #    max_probas = probas.max(1)  # max(1) gives max of each row
        #
        #    min_max_proba_index = max_probas.argmin()  # gives index of min element
        #    return rows.iloc[min_max_proba_index, :]
        # class_certainty = np.abs(proba_class_1 - 0.5)[0] / 0.5  # NOTE: assumes only one element in rows
        # return class_certainty[0]  # assume only one input row, make list to allow for json saving later

    def get_certainties(self):
        """
        :return: df with columns = [min_certainty, min_certainty_of_similar]
                used to get certainty ratios per step in als
        """
        df = pd.DataFrame()
        df.insert(0, 'min_certainty', self.als.max_uncertainties)
        df.insert(0, 'min_certainty_of_similar', self.als.similiar_uncertainties)

        return df
