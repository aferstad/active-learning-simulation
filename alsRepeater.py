# REPEATS ALS WITH THE EXACT SAME PARAMETERS AND RETURNS AVERAGE PERFORMANCE PER STEP

from als import ALS
import numpy as np
import alsDataManager

from joblib import Parallel, delayed
#from tqdm import tqdm
from datetime import datetime


class AlsRepeater:

    def __init__(self, input_dict = None, id = None):
        if input_dict is None:
            default_values = ALS.__init__.__defaults__
            argument_names = ALS.__init__.__code__.co_varnames[1:]  # [1:] to remove 'self' as input argument

            non_default_values = [None] * (len(argument_names) - len(default_values))

            values = non_default_values.copy()
            values.extend(default_values)

            self.input_dict = dict(zip(argument_names, values))
        else:
            self.input_dict = input_dict

        self.results = []
        self.id = id

    def run(self, n_reps, n_als_to_perform, n_als_performed, n_jobs=4):
        """
        :param n_reps: number of repetitions of als to perform with the exact same parameters (except seed)
        :param n_jobs: number of cores to use
        :return: nothing, self.results gets new values
        """
        self.results = []

        seeds = list(range(n_reps))
        inputs = seeds
        #inputs = tqdm(seeds)

        def myfunction(seed):
            input_dict_copy = self.input_dict.copy()
            input_dict_copy['seed'] = seed

            als = ALS(**input_dict_copy)  # ** allows to pass arguments as dict
            als.learningManager.run_experiment()

            # result is a dict with keys as metric_strs and values as list of that metric per learning step
            result = als.learningManager.get_performance_results()
            return result.copy()

        # print(__name__)
        # if __name__ == 'alsRepeater':
        #    self.results = Parallel(n_jobs=n_jobs)(delayed(myfunction)(i) for i in inputs)

        for i in range(n_reps):
            self.input_dict['seed'] = i
            als = ALS(**self.input_dict)  # ** allows to pass arguments as dict
            als.learningManager.run_experiment(n_als_performed=n_als_performed, n_als_to_perform=n_als_to_perform)
            n_als_performed = n_als_performed + 1
            # result is a dict with keys as metric_strs and values as list of that metric per learning step
            result = als.learningManager.get_performance_results()
            self.results.append(result)
            #if n_reps > 1:  #
            self.save_temp_results()

    def save_temp_results(self):
        print(self.results)
        temp_dict = {}
        temp_dict['results'] = self.results.copy()
        temp_dict['input_dict'] = self.input_dict.copy()

        now = datetime.now()
        path = 'output/jsons/temp/alsRepeater_id' + str(self.id) + '_' + now.strftime("%Y-%m-%d-%H%M-%s") + '.txt'
        #print(path)
        #print(self.results)

        alsDataManager.save_dict_as_json(self.results, path)


    def get_mean_results(self):
        """
        :return: dict with a key for each performance metric (accuracy, consistency, etc.)
            and value as list with an average value for that metric at that learning step
        """
        if len(self.results) == 0:
            return None

        # results is a list of dicts
        # where each dict has keys as metric_strs and values as list of that metric per learning step

        # first get all metric strings
        metric_strs = list(self.results[0].keys())

        # second initialize a empty list per mean_results metric_key key
        dict_of_result_matrices = {}
        for metric_str in metric_strs:
            dict_of_result_matrices[metric_str] = []

        for result in self.results:
            for metric_str in result:
                metric = result[metric_str]
                dict_of_result_matrices[metric_str].append(metric)
        # e.g. dict_of_result_matrices['accuracy'] will be a matrix
        # where each column is an accuracy list for an experiment

        mean_results = {}
        for metric_str in dict_of_result_matrices:
            matrix = alsDataManager.create_matrix(columns_of_unequal_length=dict_of_result_matrices[metric_str])

            mean_of_columns = np.nanmean(matrix, axis = 0) # axis = 0 gives row means

            mean_results[metric_str] = list(mean_of_columns)  # make list again to be able to save as json

        return mean_results



