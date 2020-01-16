import matplotlib
matplotlib.use('Agg') # Do not move this behind any other import, otherwise error will be given due to no display available

import experiment
import importlib
importlib.reload(experiment)
from experiment import Experiment

import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt





def run_repetitions(data, reps, keep, delete, methods, print_progress=True, use_pca = False, scale = False, n_points_to_add_at_a_time = 1, certainty_ratio_threshold = 2):
    '''
    INPUT
        reps : number of experiments to run and average for current keep-delete pair
        keep : number of points to keep after training initial model
        delete : number of points to delete after training initial model
    OUTPUT
        accuracy_results : df with mean accuracy per point added, columns are random and uncertainty method
        consistency_results : df with mean consistency per point added, columns are random and uncertainty method
    DESCRIPTION
        run uncertainty and random experiment with reps repitions,
        all with the same keep and delete parameters
        outputs dataframe with mean accuracy per step for all repetitions
        one column for random and one for uncertainty method
    '''
    accuracies = {}
    consistencies = {}
    certainties = {}

    for method in methods:
        accuracies[method] = []
        consistencies[method] = []
        if method == 'similar_uncertainty_optimization':
            certainties[method] = []

        for i in range(reps):
            print('Keep: ' + str(keep) + ' | Delete: ' + str(delete) + ' | Repition: ' + str(i))
            if print_progress:
                print(i)

            my_experiment = Experiment(data, i, 'lr', keep, delete, use_pca = use_pca, scale = scale, n_points_to_add_at_a_time = n_points_to_add_at_a_time, certainty_ratio_threshold = certainty_ratio_threshold)
            my_experiment.run_experiment(method=method)

            accuracies[method].append(my_experiment.accuracies)
            consistencies[method].append(my_experiment.consistencies)
            if method == 'similar_uncertainty_optimization':
                certainties[method].append(my_experiment.certainties)

        accuracies[method] = pd.DataFrame(accuracies[method]).mean(axis=0)
        consistencies[method] = pd.DataFrame(consistencies[method]).mean(axis=0)
        if method == 'similar_uncertainty_optimization':
            certainties[method] = pd.DataFrame(certainties[method]).mean(axis=0)

    accuracy_results = pd.concat(accuracies, axis = 1)
    accuracy_results.columns = methods

    consistency_results = pd.concat(consistencies, axis = 1)
    consistency_results.columns = methods

    return accuracy_results, consistency_results, certainties['similar_uncertainty_optimization']

'''
RUNS KEEP_DELETE EXPERIMENTS
'''

import importlib # to reload packags
import pandas as pd
import time # to calculate run time
import sys # to get arguments from terminal



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

keep = 50
delete = 10 #TODO: CHANGE
certainty_ratio_threshold = 2




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
        try:
            certainty_ratio_argument = sys.argv[4]
            certainty_ratio_threshold = int(certainty_ratio_argument)
            print('certainty_ratio_threshold set to: ' + str(certainty_ratio_threshold))
        except:
            print('certainty_ratio_threshold set to: ' + str(certainty_ratio_threshold))
    except:
        print('repetitions set to: ' + str(reps))
except:
    print('RUNNING AS TEST. Did you forget the argument *not_test*?')

save_path_accuracy = 'output/' + dataset_str + '_' + str(certainty_ratio_threshold) + '_accuracy_certainty_test.csv'
save_path_consistency = 'output/' + dataset_str + '_' + str(certainty_ratio_threshold) +'_consistency_certainty_test.csv'
save_path_certainty = 'output/' + dataset_str + '_' + str(certainty_ratio_threshold) + '_certainty_certainty_test.csv'



if test:
    # TODO: edit this!
    n_points_to_add_at_a_time = 1
    n_components = 100
    #keeps = [20, 100]
    #deletes = [0, 100]
    reps = 1
    #save_path_accuracy = 'output/test_' + dataset_str + '_keep_delete_accuracy_50_rep_grid.png'
    #save_path_consistency = 'output/test_' + dataset_str + '_keep_delete_consistency_50_rep_grid.png'

methods = ['random', 'uncertainty', 'similar', 'similar_uncertainty_optimization']
method_colors = {methods[0] : 'dodgerblue', methods[1] : 'orange', methods[2] : 'brown', methods[3]: 'green'}

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
    #keeps = [20, 100]
    #deletes = [0, 100]
    data = get_ads_data(n_components = n_components) # TODO: edit n and random state? Try with all data too?

#if test:
    #methods = ['random']
    #method_colors = {methods[0] : 'dodgerblue'}

accuracy_results, consistency_results, certainty_results = run_repetitions(data = data, reps = reps, keep = keep, delete=delete, methods = methods, use_pca=False, scale = False, n_points_to_add_at_a_time = n_points_to_add_at_a_time, certainty_ratio_threshold = certainty_ratio_threshold)
print(accuracy_results)
print(consistency_results)
print(certainty_results)



accuracy_results.to_csv(save_path_accuracy)
consistency_results.to_csv(save_path_consistency)
certainty_results.to_csv(save_path_certainty)

end_time = time.time()
run_time = round((end_time - start_time) / 60, 2) # gives number of minutes of run_time




if test:
    print('Test succesful')
