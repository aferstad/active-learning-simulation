from input.heart_import import get_heart_data
from alsRepeaterLauncher import AlsRepeaterLauncher
import alsDataManager


if __name__ == '__main__':  # to avoid multiprocessor children to begin from start

    data = get_heart_data()
    launcher = AlsRepeaterLauncher()

    launcher.input_dict['unsplit_data'] = data
    launcher.reps = 5

    argument_value_dict = {}
    argument_value_dict['learning_method'] =  ['random',
                                              'uncertainty',
                                              'bayesian_random',
                                              'similar',
                                              'similar_uncertainty_optimization']

    argument_value_dict['n_points_labeled_keep'] = [10, 20, 30]
    argument_value_dict['n_points_labeled_delete'] = [0, 10, 20, 30]


    test = True

    if test:
        print('RUNNING AS TEST, EDIT TEST = FALSE TO AVOID THIS')
        argument_value_dict['learning_method'] = ['bayesian_random']
        #argument_value_dict['learning_method'] = ['similar_uncertainty_optimization']

        argument_value_dict['n_points_labeled_keep'] = [20]
        argument_value_dict['n_points_labeled_delete'] = [20]
        launcher.reps = 1
        launcher.input_dict['n_points_to_add_at_a_time'] = 100

    results = launcher.run_3_dimensional_varied_reps(argument_value_dict)

    json_path = 'results.txt'

    alsDataManager.save_dict_as_json(results, json_path)


