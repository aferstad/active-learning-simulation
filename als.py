# ACTIVE LEARNING SIMULATOR
# author: Andreas Ferstad

from alsDataManager import AlsDataManager
from alsModelManager import AlsModelManager
from alsLearningManager import AlsLearningManager

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd


class ALS:

    def __init__(self,
                 unsplit_data,
                 learning_method,
                 model_type='lr',
                 seed=0,
                 n_points_labeled_keep=25,
                 n_points_labeled_delete=25,
                 use_pca=False,  # TODO: decide whether I always should do PCA?
                 scale=True,  # TODO: decide whether I always should scale?
                 n_points_to_add_at_a_time=1,
                 certainty_ratio_threshold=2,
                 pct_unlabeled_to_label=1.00,
                 pct_points_test=0.25,
                 cores = 4):
        """
        input: unsplit prepared data, learning_method, and possibility to change default parameters
        """
        # Data Parameters Init
        self.unsplit_data = unsplit_data
        self.data = {}
        self.seed = seed
        self.n_points_labeled_keep = n_points_labeled_keep
        self.n_points_labeled_delete = n_points_labeled_delete
        self.pct_points_test = pct_points_test
        self.use_pca = use_pca
        self.scale = scale
        self.scaler = StandardScaler()
        self.pca = PCA()

        # Model Parameters Init
        self.model_type = model_type
        self.certainties = []
        self.traces = []
        self.latest_trace = None
        self.cores = cores
        self.model_initial = None
        self.model_current = None

        # Performance Metrics Init
        self.accuracies = []
        self.consistencies = []
        self.similar_uncertainties = []
        self.max_uncertainties = []

        # Manager Init
        self.dataManager = AlsDataManager(self)
        self.modelManager = AlsModelManager(self)
        self.learningManager = AlsLearningManager(self)

        # Learning Parameters Init
        self.learning_method = learning_method
        self.n_points_to_add_at_a_time = n_points_to_add_at_a_time
        self.certainty_ratio_threshold = certainty_ratio_threshold
        self.pct_unlabeled_to_label = pct_unlabeled_to_label
        self.similar_learning_method_initiated = False
        self.similar_learning_method_closest_unlabeled_rows = pd.DataFrame()


        # Prepare Data
        self.dataManager.set_partitions()
        self.dataManager.transform_data(use_pca, scale)
        # TODO: decide whether I always should scale data?

        if not self.is_legal_init():
            print('ERROR: Init not legal')

    def is_legal_init(self):
        """
        :return: True if init parameters and experiment prep legal
        """

        if self.learning_method == 'bayesian_random' and self.model_type != 'lr':
            print('bayesian_random code not supported for other models than lr. Setting model to lr...')
            self.model_type = 'lr'
            return False

        #if learning_method == 'bayesian_random' and n_points_to_add_at_a_time < 25:
        #    print('setting n_points_to_add_at_a_time = 25, remove code in ALS to avoid this')
        #    self.n_points_to_add_at_a_time = 25

        if self.learning_method == 'bayesian_random' and self.scale is not True:
            print('bayesian should always scale')
            self.scale = True
            return False

        labeled_data = self.dataManager.get_labeled_data()

        if len(labeled_data.iloc[:, 0].unique()) != 2:
            print('Error: initial labeled data only contains one class')
            return False

        return True




