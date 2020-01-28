import pandas as pd
import numpy as np
import alsDataManager

from bayesianLogisticRegression import BayesianLogisticRegression  # my custom model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression



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

    def fit_model(self):
        """
        :return: model fitted on labeled data
        """
        labeled = self.als.dataManager.get_labeled_data()
        X, y = alsDataManager.get_X_y(labeled)

        if self.als.learning_method == 'bayesian_random':
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
        # elif self.als.model_type == 'KNN':
        #    best_k = self.KNN_cv(X, y)
        #    model = KNeighborsClassifier(n_neighbors=best_k)
        #    model.fit(X, y)
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

    def get_point_certainty(self, rows):
        """
        :param rows: current_model predicts P(Y=1) for each row in rows
        :return: certainty per row
        """
        X, y = alsDataManager.get_X_y(rows)
        proba_class_1 = self.als.model_current.predict_proba(X)[:, 1]
        class_certainty = np.abs(proba_class_1 - 0.5)[0] / 0.5  # NOTE: assumes only one element in rows
        return class_certainty

    def get_certainties(self):
        """
        :return: df with columns = [min_certainty, min_certainty_of_similar]
                used to get certainty ratios per step in als
        """
        df = pd.DataFrame()
        df.insert(0, 'min_certainty', self.als.max_uncertainties)
        df.insert(0, 'min_certainty_of_similar', self.als.similiar_uncertainties)

        return df


"""
    # NOT USED:
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
            plt.xlabel('Value of K for KNN')
            plt.ylabel('Cross-Validated Accuracy')
            plt.show()
    
        best_k = max(
            np.array(k_range)[max(k_scores) <= k_scores + np.std(k_scores)])
    
        return best_k
"""