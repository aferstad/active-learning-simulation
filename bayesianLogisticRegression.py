import numpy as np
import pymc3 as pm


class BayesianLogisticRegression:

    def __init__(self):
        self.initialized = False
        self.trace = []
        self.traces = []

    def fit(self, x, y_obs):
        n_features = x.shape[1]

        with pm.Model() as model:

            if not self.initialized:
                mu = np.zeros(n_features)
                cov = np.identity(n_features)
                betas = pm.MvNormal('betas', mu=mu, cov=cov, shape=n_features)

                self.initialized = True
            else:
                # trace is the sample from the latest found posterior
                # here we find the new prior by estimating the parameters of the latest posterior
                nu = self.trace['betas'].shape[
                    0]  # number of degrees of freedom for MvStudentT is assumed to be number of points in sample
                mu = self.trace['betas'].mean(0)  # mean 0 gives mean of each column, i.e. coefficient beta_i
                cov = ((1. * nu) / (nu - 2)) * np.cov(self.trace['betas'].T)

                betas = pm.MvStudentT('betas', mu=mu, cov=cov, nu=nu, shape=n_features)

            p = pm.math.invlogit(x @ betas)  # give the probability in a logistic regression model

            # Define likelihood
            y = pm.Bernoulli('y', p, observed=y_obs)

            # Inference:
            self.trace = pm.sample(2000)
            self.traces.append(self.trace)

            # TODO: CONTINUE WORKING HERE (24/1)
