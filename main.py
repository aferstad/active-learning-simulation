import experiment
import census_import

import importlib
importlib.reload(experiment)
importlib.reload(census_import)

from experiment import Experiment
from census_import import get_census_data

import pandas as pd
import numpy as np

from IPython.display import clear_output

census, census_with_dummies = get_census_data('input_data/adult.data', 'input_data/adult.test')

experiments = []
results = pd.DataFrame(columns=['initial_accuracy', 'final_accuracy'])

for i in range(2):
    print(i)
    experiments.append(Experiment(census_with_dummies, seed=i, n_points_to_add_at_a_time = 500))
    experiments[i].run_experiment(method='uncertainty')

    results.loc[i] = [experiments[i].model_initial_accuracy, experiments[i].model_final_accuracy]
    results.to_csv('results.csv')
