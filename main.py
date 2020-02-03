from input.heart_import import get_heart_data
from alsRepeaterLauncher import AlsRepeaterLauncher
import alsDataManager
import sys  # to get arguments from terminal


if __name__ == '__main__':  # to avoid multiprocessor children to begin from start

    input_arguments = sys.argv

    if len(input_arguments) == 1:
        raise Exception('ERROR: no save path specified')

    save_path = input_arguments[1]

    data = get_heart_data()
    launcher = AlsRepeaterLauncher()

    launcher.input_dict['unsplit_data'] = data
    launcher.reps = 10
    launcher.input_dict['n_points_labeled_keep'] = 15

    argument_value_dict = {}
    argument_value_dict['learning_method'] =  ['random',
                                              'uncertainty',  # 'bayesian_random',
                                              'similar',
                                              'similar_uncertainty_optimization']

    argument_value_dict['certainty_ratio_threshold'] = [2, 10, 50, 500]
    argument_value_dict['n_points_labeled_delete'] = [10, 20, 30]


    test = True

    if test:
        print('RUNNING AS TEST, EDIT TEST = FALSE TO AVOID THIS')
        argument_value_dict['learning_method'] = ['random',
                                                  'uncertainty',  # 'bayesian_random',
                                                  'similar',
                                                  'similar_uncertainty_optimization']

        argument_value_dict['certainty_ratio_threshold'] = [50]
        argument_value_dict['n_points_labeled_delete'] = [10]
        launcher.reps = 1

    results = launcher.run_3_dimensional_varied_reps(argument_value_dict)

    json_path = save_path + '.txt'

    alsDataManager.save_dict_as_json(results, json_path)


