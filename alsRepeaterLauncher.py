
from alsRepeater import AlsRepeater
from als import ALS
import json


class AlsRepeaterLauncher:

    def __init__(self, input_dict = None, reps =5, n_jobs = 5):
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
        self.reps = reps
        self.n_jobs = n_jobs

    def run_varied_reps(self, argument_str, argument_value_range):
        """
        :param argument_str: str of argument which value to vary
        :param argument_value_range: list of values to run experiments with
        :return: dict with keys as argument_value and value as mean_result dict
        """
        results = {}
        for argument_value in argument_value_range:
            alsr = AlsRepeater(self.input_dict)
            alsr.input_dict[argument_str] = argument_value
            alsr.run(n_reps=self.reps)
            result = alsr.get_mean_results()

            results[argument_value] = result
        return results

    def run_3_dimensional_varied_reps(self, argument_value_dict):
        """
        :param argument_value_dict: dict with keys as arguments, and value as list of argument values
        :return: 4 dimensional dict,
            first key = first argument to vary on + argument value
                second key = second argument to very on + argument value
                    third key = third argument to very on + argument value
                        fourth key = performance metric (e.g. accuracy)
                            value = average of performance metric per learning step across reps
        """

        argument_strs = list(argument_value_dict.keys())
        n_arguments = len(argument_strs)

        if n_arguments != 3:
            print('ERROR: argument_value_dict must have 3 keys. To only vary along 1 or 2 axis, include a key with value as a list with only one value')

        results = {} # initiate output 3D dict

        # To print progress percentages later:
        n_als_to_perform = self.reps
        for key in argument_value_dict:
            n_als_to_perform = n_als_to_perform * len(argument_value_dict[key])
        n_als_performed = 0

        for i in argument_value_dict[argument_strs[0]]:  # iterate over argument values of first argument string
            input_dict_altered = self.input_dict.copy()
            input_dict_altered[argument_strs[0]] = i  # alter argument1 to have value i

            result_key1 = argument_strs[0] + '_' + str(i)
            results[result_key1] = {}  # initiate sub dictionary
            for j in argument_value_dict[argument_strs[1]]:  # iterate over argument values of second argument string
                input_dict_altered[argument_strs[1]] = j  # alter argument2 to have value j

                result_key2 = argument_strs[1] + '_' + str(j)
                results[result_key1][result_key2] = {}  # initiate sub sub dictionary
                for k in argument_value_dict[argument_strs[2]]:
                    input_dict_altered[argument_strs[2]] = k  # alter argument3 to have value k

                    # run reps with the two arguments having value i and j
                    alsr = AlsRepeater(input_dict_altered)
                    alsr.run(n_reps=self.reps, n_jobs=self.n_jobs,  n_als_to_perform=n_als_to_perform, n_als_performed=n_als_performed)
                    n_als_performed = n_als_performed + self.reps

                    result_key3 = argument_strs[2] + '_' + str(k)
                    results[result_key1][result_key2][result_key3] = alsr.get_mean_results()

                    print('#### COMPLETED REPS: ' + result_key1 + ' | ' + result_key2 + ' | ' + result_key3 + '####')

        return results


