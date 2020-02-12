from input.heart_import import get_heart_data
from input.ads_import import get_ads_data
from alsRepeaterLauncher import AlsRepeaterLauncher
import alsDataManager
import sys  # to get arguments from terminal


if __name__ == '__main__':  # to avoid multiprocessor children to begin from start

    input_arguments = sys.argv

    launcher = AlsRepeaterLauncher()
    #launcher.reps = 10
    if len(input_arguments) == 1:
        #raise Exception('ERROR: no save path specified')
        save_path = 'no_save_path_specified'
    else:
        save_path = input_arguments[1]

    data = get_heart_data()
    launcher = AlsRepeaterLauncher()

    launcher.input_dict['unsplit_data'] = data
    launcher.input_dict['model_type'] = 'xgboost'
    launcher.reps = 100
    #launcher.input_dict['n_points_labeled_keep'] = 10

    # arguments to vary on:
    argument_value_dict = {}
    argument_value_dict['learning_method'] = ['random',
                                              'uncertainty',  # 'bayesian_random',
                                              'similar',
                                              'similar_uncertainty_optimization']

    argument_value_dict['certainty_ratio_threshold'] = [5, 50, 100, 500]
    argument_value_dict['n_points_labeled_delete'] = [0, 10, 20, 30]
    launcher.input_dict['n_points_labeled_keep'] = 15


    if len(input_arguments) == 1:
        print('ERROR: no save path specified, setting save path to "no_save_path_specified"')
        save_path = 'no_save_path_specified'
    else:
        save_path = input_arguments[1]

    if len(input_arguments) == 3:
        data_str = input_arguments[2]
    else:
        print('ERROR: no data specified, setting data path to heart')
        data_str = 'heart'

    if data_str == 'heart':
        launcher.input_dict['unsplit_data'] = get_heart_data()
    elif data_str == 'ads':
        launcher.input_dict['unsplit_data'] = get_ads_data(n_components = 1000)
        launcher.input_dict['pct_unlabeled_to_label'] = 0.30
        launcher.input_dict['n_points_labeled_delete'] = 300
        argument_value_dict['n_points_labeled_keep'] = [400, 500, 600, 700]

    test = True #False #True #False #False #True


    if test:
        print('RUNNING AS TEST, EDIT TEST = FALSE TO AVOID THIS')
        launcher.reps = 2  # run tests with 2 reps to test that mean() functions are working properly
        #launcher.input_dict['n_points_labeled_keep'] = 15

        argument_value_dict = {}
        argument_value_dict['learning_method'] = ['random']
                                            #      'uncertainty',  # 'bayesian_random',
                                            #      'similar',
                                             #     'similar_uncertainty_optimization']

        argument_value_dict['certainty_ratio_threshold'] = [2, 500]
        argument_value_dict['n_points_labeled_delete'] = [0, 40]
        #argument_value_dict['n_points_labeled_keep'] = [400, 700]

    results = launcher.run_3_dimensional_varied_reps(argument_value_dict)

    json_path = save_path + '.txt'

    alsDataManager.save_dict_as_json(results, json_path)
