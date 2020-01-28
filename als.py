# ACTIVE LEARNING SIMULATOR
# author: Andreas Ferstad

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bayesianLogisticRegression import BayesianLogisticRegression # my custom model
from alsDataManager import AlsDataManager
import alsDataManager

import alsModelManager
from alsModelManager import AlsModelManager

from alsLearningManager import AlsLearningManager


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from scipy import spatial  # for nearest neighbour
import pymc3 as pm # for bayesian models


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

        self.dataManager = AlsDataManager(self)
        self.modelManager = AlsModelManager(self)
        self.learningManager = AlsLearningManager(self)



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

        self.model_parameters = []




        self.dataManager.set_partitions()

        self.dataManager.transform_data(use_pca, scale, self.add_intercept_column)
        # TODO: decide whether I always should scale data?

        # Performance Metrics
        self.accuracies = []
        self.consistencies = []
        self.similar_uncertainties = []
        self.max_uncertainties = []
        self.certainties = []

        # Model Parameters
        labeled = self.dataManager.get_labeled_data()

        if len(labeled.iloc[:, 0].unique()) != 2:
            print('Error: initial labeled data only contains one class')

        #self.bayesian_model_initialized = False
        #self.bayesian_model = pm.Model()
        #self.trace = []
        self.traces = []
        self.latest_trace = None
        self.cores = cores
        self.model_initial = None

        self.similar_uncertainties = []













