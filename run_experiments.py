#import matplotlib
#matplotlib.use('Agg')

#import input.census_import
import input.heart_import
import input.admission_import
import input.alcohol_import
import input.breast_cancer_import


#import experiment
import keep_delete_experiments

import importlib
importlib.reload(keep_delete_experiments)
#importlib.reload(input.census_import)
importlib.reload(input.heart_import)
importlib.reload(input.admission_import)
importlib.reload(input.alcohol_import)
importlib.reload(input.breast_cancer_import)


from keep_delete_experiments import run_experiments, plot_results
#from input.census_import import get_census_data
from input.heart_import import get_heart_data
from input.admission_import import get_admission_data
from input.alcohol_import import get_alcohol_data
from input.breast_cancer_import import get_cancer_data


import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt

import time
start_time = time.time()



import sys
input_arguments = sys.argv
#print(input_arguments)
dataset = sys.argv[1]

test = True
try:
    test_argument = sys.argv[2]
    if test_argument == 'not_test':
        test = False
except:
    print('running test')

data = []
if dataset == 'heart':
    data = get_heart_data('input/heart.csv')[1]
elif dataset == 'admission':
    data = get_admission_data()
elif dataset == 'alcohol':
    data = get_alcohol_data()
elif dataset == 'cancer':
    data = get_cancer_data()

keeps = range(10, 70, 10)
deletes = range(0, 70, 10)
reps = 100
save_path_accuracy = 'output/' + dataset + '_keep_delete_accuracy_50_rep_grid.png'
save_path_consistency = 'output/' + dataset + '_keep_delete_consistency_50_rep_grid.png'

if test:
    keeps = [10, 20]
    deletes = [10, 20]
    reps = 1
    save_path_accuracy = 'output/test_' + dataset + '_keep_delete_accuracy_50_rep_grid.png'
    save_path_consistency = 'output/test_' + dataset + '_keep_delete_consistency_50_rep_grid.png'

methods = ['random', 'uncertainty', 'similar']
method_colors = {methods[0] : 'dodgerblue', methods[1] : 'orange', methods[2] : 'brown'}


accuracy_results, consistency_results = run_experiments(data = data, reps = reps, keeps = keeps, deletes=deletes, methods = methods)

end_time = time.time()
run_time = round((end_time - start_time) / 60, 2) # gives number of minutes of run_time

plot_results(accuracy_results, reps, keeps, deletes, save_path_accuracy, methods, method_colors, dataset=dataset, ylabel = 'accuracy', run_time = run_time)
plot_results(consistency_results, reps, keeps, deletes, save_path_consistency, methods, method_colors,dataset=dataset, ylabel = 'consistency', run_time = run_time)

if test:
    print('Test succesful')
