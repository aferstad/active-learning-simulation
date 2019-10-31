import experiment
import heart_import

import importlib
importlib.reload(experiment)
importlib.reload(heart_import)

from experiment import Experiment
from heart_import import get_heart_data

import pandas as pd
import numpy as np



heart, heart_with_dummies = get_heart_data('input_data/heart.csv')
experiments = []
accuracies = pd.DataFrame()
best_ks = pd.DataFrame()

for i in range(10):
    print(i)
    experiments.append(Experiment(heart_with_dummies, seed=i))
    experiments[i].run_experiment(method='random')


    accuracies.insert(0, 'experiment' + str(i), experiments[i].accuracies)
    best_ks.insert(0, 'experiment' + str(i), experiments[i].best_ks)

    #results.loc[i] = [experiments[i].model_initial_accuracy, experiments[i].model_final_accuracy]
    accuracies.to_csv('accuracies.csv')
    best_ks.to_csv('best_ks')
