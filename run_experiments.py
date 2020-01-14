'''
RUNS KEEP_DELETE EXPERIMENTS
'''

import importlib # to reload packags
import pandas as pd
import time # to calculate run time
import sys # to get arguments from terminal

# CUSTOM METHOD IMPORT:
import keep_delete_experiments
importlib.reload(keep_delete_experiments)
from keep_delete_experiments import run_experiments, plot_results

# CUSTOM DATA IMPORTS:
#import input.census_import
import input.heart_import
import input.admission_import
import input.alcohol_import
import input.breast_cancer_import
import input.ads_import

#importlib.reload(input.census_import)
importlib.reload(input.heart_import)
importlib.reload(input.admission_import)
importlib.reload(input.alcohol_import)
importlib.reload(input.breast_cancer_import)
importlib.reload(input.ads_import)

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
keeps = range(10, 70, 10)
deletes = range(0, 70, 10)
reps = 100
save_path_accuracy = 'output/' + dataset_str + '_keep_delete_accuracy_50_rep_grid.png'
save_path_consistency = 'output/' + dataset_str + '_keep_delete_consistency_50_rep_grid.png'

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
    n_points_to_add_at_a_time = 50
    n_components = 100
    keeps = [20, 100]
    deletes = [0, 100]
    reps = 1
    save_path_accuracy = 'output/test_' + dataset_str + '_keep_delete_accuracy_50_rep_grid.png'
    save_path_consistency = 'output/test_' + dataset_str + '_keep_delete_consistency_50_rep_grid.png'

methods = ['random', 'uncertainty', 'similar']
method_colors = {methods[0] : 'dodgerblue', methods[1] : 'orange', methods[2] : 'brown'}

data = []
if dataset_str == 'heart':
    data = get_heart_data('input/heart.csv')[1]
elif dataset_str == 'admission':
    data = get_admission_data()
elif dataset_str == 'alcohol':
    data = get_alcohol_data()
elif dataset_str == 'cancer':
    data = get_cancer_data()
elif dataset_str == 'ads':
    keeps = [20, 100]
    deletes = [0, 100]
    data = get_ads_data(n_components = n_components) # TODO: edit n and random state? Try with all data too?

#if test:
    #methods = ['random']
    #method_colors = {methods[0] : 'dodgerblue'}

accuracy_results, consistency_results = run_experiments(data = data, reps = reps, keeps = keeps, deletes=deletes, methods = methods, use_pca=False, scale = False, n_points_to_add_at_a_time = n_points_to_add_at_a_time)

end_time = time.time()
run_time = round((end_time - start_time) / 60, 2) # gives number of minutes of run_time

plot_results(accuracy_results, reps, keeps, deletes, save_path_accuracy, methods, method_colors, dataset_str=dataset_str, ylabel = 'accuracy', run_time = run_time)
plot_results(consistency_results, reps, keeps, deletes, save_path_consistency, methods, method_colors,dataset_str=dataset_str, ylabel = 'consistency', run_time = run_time)

if test:
    print('Test succesful')
