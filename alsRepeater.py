# REPEATS ALS WITH THE EXACT SAME PARAMETERS AND RETURNS AVERAGE PERFORMANCE PER STEP

from als import ALS
import numpy as np


class AlsRepeater:

    def __init__(self, input_dict = None):
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

    def run(self, n_reps):
        """
        :param n_reps: number of repetitions of als to perform with the exact same parameters (except seed)
        :return: nothing, self.results gets new values
        """
        self.results = []
        for i in range(n_reps):
            self.input_dict['seed'] = i
            als = ALS(**self.input_dict)  # ** allows to pass arguments as dict
            als.learningManager.run_experiment()

            # result is a dict with keys as metric_strs and values as list of that metric per learning step
            result = als.learningManager.get_performance_results()
            self.results.append(result)

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
            mean_results[metric_str] = np.array(dict_of_result_matrices[metric_str]).mean(0)

        return mean_results



