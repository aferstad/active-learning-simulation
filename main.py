from input.heart_import import get_heart_data
from input.ads_import import get_ads_data
from alsRepeaterLauncher import AlsRepeaterLauncher
import alsDataManager
import sys  # to get arguments from terminal
import multiprocessing

# import matplotlib
# matplotlib.use('Agg')


if __name__ == '__main__':  # to avoid multiprocessor children to begin from start

    launcher = AlsRepeaterLauncher()

    launcher.n_jobs = multiprocessing.cpu_count()
    #launcher.input_dict['model_type'] = 'xgboost'

    launcher.reps = 10
    launcher.input_dict['n_points_labeled_keep'] = 400
    launcher.input_dict['n_points_labeled_delete'] = 300
    launcher.input_dict['pct_unlabeled_to_label'] = 0.25

    # arguments to vary on:
    argument_value_dict = {}
    argument_value_dict['learning_method'] = ['random',
                                              'uncertainty',
                                              'similar',
                                              'similar_uncertainty_optimization']  # bayesian_random can be added

    argument_value_dict['certainty_ratio_threshold'] = [50]  # [2, 10, 50, 250]
    argument_value_dict['n_points_labeled_delete'] = [300]  # , 20, 30]

    input_arguments = sys.argv

    if len(input_arguments) == 1:
        #raise Exception('ERROR: no save path specified')
        print('WARNING: NO SAVE PATH SPECIFIED. SETTING SAVE PATH TO "no_save_path_specified"')
        save_path = 'output/jsons/no_save_path_specified'
    else:
        save_path = 'output/jsons/' + input_arguments[1]
        print('Save path set to ' + save_path)

    if len(input_arguments) == 3:
        data_str = input_arguments[2]
    else:
        #data_str = 'heart'
        data_str = 'ads'
        print('ERROR: no data specified, setting data path to ' + data_str)
        #data_str = 'heart'
        #data_str = 'ads'

    if data_str == 'heart':
        launcher.input_dict['unsplit_data'] = get_heart_data()
    elif data_str == 'ads':
        launcher.input_dict['unsplit_data'] = get_ads_data(n_components = 1000)
        # launcher.input_dict['pct_unlabeled_to_label'] = 0.30
        # launcher.input_dict['n_points_labeled_delete'] = 300
        # argument_value_dict['n_points_labeled_keep'] = [400, 500, 600, 700]

    test = True

    if test:
        print('RUNNING AS TEST, EDIT TEST = FALSE TO AVOID THIS')
        launcher.reps = 2  # run tests with 2 reps to test that mean() functions are working properly

        # launcher.input_dict['n_points_labeled_keep'] = 15
        argument_value_dict = {}
        argument_value_dict['learning_method'] = ['random',
                                                  'uncertainty']  # 'bayesian_random',
                                            #      'similar',
                                             #     'similar_uncertainty_optimization']

        #argument_value_dict['certainty_ratio_threshold'] = [2, 500]
        argument_value_dict['n_points_labeled_keep'] = [15, 30]
        argument_value_dict['n_points_labeled_delete'] = [0, 50]
        #argument_value_dict['n_points_labeled_keep'] = [400, 700]

    results = launcher.run_3_dimensional_varied_reps(argument_value_dict)

    json_path = save_path + '.txt'
    alsDataManager.save_dict_as_json(results, json_path)
