'''
RUNS KEEP_DELETE EXPERIMENTS
'''
import pandas as pd
import time  # to calculate run time
import sys  # to get arguments from terminal

# CUSTOM METHOD IMPORT:
from als_repeater import run_experiments, run_certainty_experiments, plot_results, plot_certainty_results

# CUSTOM DATA IMPORTS:
#from input.census_import import get_census_data
from input.heart_import import get_heart_data
from input.admission_import import get_admission_data
from input.alcohol_import import get_alcohol_data
from input.breast_cancer_import import get_cancer_data
from input.ads_import import get_ads_data



# PROGRAM START:
start_time = time.time()
input_arguments = sys.argv
dataset_str = sys.argv[1]

# if not test:
n_components = 1000
n_points_to_add_at_a_time = 1
keeps = [10, 20] # range(10, 70, 10)
deletes = [0, 20, 30, 40, 50] # range(0, 70, 10)
reps = 20
save_path_accuracy = 'output/' + dataset_str + '_keep_delete_accuracy_50_rep_grid.png'
save_path_consistency = 'output/' + dataset_str + '_keep_delete_consistency_50_rep_grid.png'
save_path_certainty = 'output/' + dataset_str + '_keep_delete_certainty_50_rep_grid.png'

test = True
try:
    test_argument = sys.argv[2]
    if test_argument != 'not_test':
        raise Exception()
    test = False
    try:
        rep_argument = sys.argv[3]
        reps = int(rep_argument)
        print('repetitions set to: ' + str(reps))
    except:
        print('repetitions set to: ' + str(reps))
except:
    print('RUNNING AS TEST. Did you forget the argument *not_test*?')

if test:
    # TODO: edit this!
    n_points_to_add_at_a_time = 1
    n_components = 100
    keeps = [10, 20]
    deletes = [0, 50]
    reps = 1
    save_path_accuracy = 'output/test_' + dataset_str + '_keep_delete_accuracy_50_rep_grid.png'
    save_path_consistency = 'output/test_' + dataset_str + '_keep_delete_consistency_50_rep_grid.png'
    save_path_certainty = 'output/test_' + dataset_str + '_keep_delete_certainty_50_rep_grid.png'

#methods = ['random', 'uncertainty', 'similar', 'similar_uncertainty_optimization']
methods = ['bayesian_random', 'random', 'uncertainty']
method_colors = {
    methods[0]: 'dodgerblue',
    methods[1]: 'orange',
    methods[2]: 'brown'
}

data = []
if dataset_str == 'heart':
    data = get_heart_data()
elif dataset_str == 'admission':
    data = get_admission_data()
elif dataset_str == 'alcohol':
    data = get_alcohol_data()
elif dataset_str == 'cancer':
    data = get_cancer_data()
elif dataset_str == 'ads':
    if test:
        n_points_to_add_at_a_time = 100

    KEEP = 50
    DELETE = 300
    certainty_ratio_thresholds = [5, 50, 100, 500]
    reps = 10

    keeps = [20, 100]
    deletes = [0, 100]
    data = get_ads_data(
        n_components=n_components
    )  # TODO: edit n and random state? Try with all data too?

if True: #__name__ == '__main__':
    accuracy_results, consistency_results, certainty_results = run_experiments(
        data=data,
        reps=reps,
        keeps=keeps,
        deletes=deletes,
        methods=methods,
        use_pca=False,
        scale=False,
        n_points_to_add_at_a_time=n_points_to_add_at_a_time,
        certainty_ratio_threshold=1000000)

    end_time = time.time()
    run_time = round((end_time - start_time) / 60,
                     2)  # gives number of minutes of run_time

    plot_results(accuracy_results,
                 reps,
                 keeps,
                 deletes,
                 save_path_accuracy,
                 methods,
                 method_colors,
                 dataset_str=dataset_str,
                 ylabel='accuracy',
                 run_time=run_time)
    plot_results(consistency_results,
                 reps,
                 keeps,
                 deletes,
                 save_path_consistency,
                 methods,
                 method_colors,
                 dataset_str=dataset_str,
                 ylabel='consistency',
                 run_time=run_time)

    plot_results(certainty_results,
                 reps,
                 keeps,
                 deletes,
                 save_path_certainty,
                 ['similar_uncertainty_optimization'],
                 method_colors,
                 dataset_str=dataset_str,
                 ylabel='certainty',
                 run_time=run_time,
                 plot_certainties=True)