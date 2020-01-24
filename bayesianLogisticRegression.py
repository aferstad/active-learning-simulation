# Author: Andreas Ferstad

import numpy as np
import pymc3 as pm
import time


class BayesianLogisticRegression:

    def __init__(self, fit_intercept=True):
        """
        :param fit_intercept: default true, makes first element in beta_hat be intercept
        """
        self.trace = None
        self.beta_hat = None
        self.fit_intercept = fit_intercept

        self.coef_ = None
        self.intercept_ = None

    # TODO: maybe return trace to some traceObserver object?
    def fit(self, x, y_obs, previous_trace=None):
        """
        :param x: training features
        :param y_obs: training binary class labels 0/1
        :param previous_trace: default None, used as prior if not None
        :return: trace, to be used as next prior

        finds distribution for coefficients in logistic regression
        sets beta_hat to mean vector of MvDistribution
        """
        x = np.array(x)
        y_obs = np.array(y_obs)

        if self.fit_intercept:
            ones = np.array([1] * x.shape[0])
            x = np.column_stack((ones, x))

        n_features = x.shape[1]

        if True: #__name__ == 'bayesianLogisticRegression':
            try:
                with pm.Model() as model:

                    if previous_trace is None: # then model has not been initialized with original prior
                        # original prior:
                        mu = np.zeros(n_features)
                        cov = np.identity(n_features)
                        betas = pm.MvNormal('betas', mu=mu, cov=cov, shape=n_features)
                    else:
                        # previous_trace is the sample from the latest found posterior
                        # here we find the new prior by estimating the parameters of the latest posterior
                        nu = previous_trace['betas'].shape[
                            0]  # number of degrees of freedom for MvStudentT is assumed to be number of points in sample
                        mu = previous_trace['betas'].mean(0)  # mean 0 gives mean of each column, i.e. coefficient beta_i
                        cov = ((1. * nu) / (nu - 2)) * np.cov(previous_trace['betas'].T)

                        betas = pm.MvStudentT('betas', mu=mu, cov=cov, nu=nu, shape=n_features)

                    p = pm.math.invlogit(x @ betas)  # give the probability in a logistic regression model

                    # Define likelihood
                    y = pm.Bernoulli('y', p, observed=y_obs)

                    # Inference:
                    self.trace = pm.sample(2000) # cores = 1, if runtime error

                self.beta_hat = self.trace['betas'].mean(0)

                if self.fit_intercept:
                    self.coef_ = self.beta_hat[1:]
                    self.intercept_ = self.beta_hat[0]
                else:
                    self.coef_ = self.beta_hat
                    self.intercept_ = None
            except RuntimeError as err:
                print('Runtime error: {0}'.format(err))
                #time.sleep(10)  # Wait for 10 seconds

                return self.trace

    def predict(self, x):
        """
        :param x: test training features
        :return: predicted class label 0/1
        """
        return self.predict_proba(x)[:, 1].round()

    def predict_proba(self, x):
        """
        :param x: test training features
        :return: Nx2 matrix, first column proba class 0 and second column proba class 1
        """
        if self.beta_hat is None:
            print("ERROR: Model not fitted")
            return None

        x = np.array(x)
        if self.fit_intercept:
            ones = np.array([1] * x.shape[0])
            x = np.column_stack((ones, x))

        y_class_1_proba = 1. / (1+np.exp(-1. * x @ self.beta_hat))
        y_class_0_proba = 1 - y_class_1_proba

        # stack probas vertically, then transpose,
        # to get matrix with first column probas0 and second column probas1
        # just as the sklearn.logisticRegression does
        y_proba = np.vstack((y_class_0_proba, y_class_1_proba)).T

        return y_proba
