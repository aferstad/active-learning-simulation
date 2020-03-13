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
            model = XGBoostModel(n_classes=self.als.n_classes)
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
        :return: certainty for row
        """

        uncertainty = self.als.learningManager.get_most_uncertain_rows(rows=rows, entropy_based = entropy_based, return_uncertainty = True)
        return 1-uncertainty

    def get_certainties(self):
        """
        :return: df with columns = [min_certainty, min_certainty_of_similar]
                used to get certainty ratios per step in als
        """
        df = pd.DataFrame()
        df.insert(0, 'min_certainty', self.als.max_uncertainties)
        df.insert(0, 'min_certainty_of_similar', self.als.similiar_uncertainties)

        return df
