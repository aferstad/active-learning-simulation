import experiment
import census_import
import heart_import

import importlib
importlib.reload(experiment)
importlib.reload(census_import)
importlib.reload(heart_import)

from experiment import Experiment
from census_import import get_census_data
from heart_import import get_heart_data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def run_repetitions(reps, keep, delete, print_progress=True):
    '''
    run uncertainty and random experiment with reps repitions,
    all with the same keep and delete parameters

    outputs dataframe with mean accuracy per step for all repetitions
    one column for random and one for uncertainty method
    '''
    accuracies = []
    accuracies_uncertainty = []

    for i in range(reps):
        if print_progress:
            print(i)

        my_experiment = Experiment(heart_with_dummies, i, 'lr', keep, delete)
        my_experiment.run_experiment(method='random')

        accuracies.append(my_experiment.accuracies)

        my_experiment = Experiment(heart_with_dummies, i, 'lr', keep, delete)
        my_experiment.run_experiment(method='uncertainty')

        accuracies_uncertainty.append(my_experiment.accuracies)

    a = pd.DataFrame(accuracies).mean(axis=0)
    au = pd.DataFrame(accuracies_uncertainty).mean(axis=0)

    results = pd.concat([a, au], axis=1)
    results.columns = ['random', 'uncertainty']

    return results

def run_experiments(keeps, deletes, reps):
    '''
    runs experiment for each combination of keep and delete number
    '''
    results = {}
    n_grid_items_compelete = 0
    for keep in keeps:
        #print('keep = ' + str(keep))
        results[keep] = {}
        for delete in deletes:
            print('Pct grid items complete: ' + str(
                round(100.0 * n_grid_items_compelete /
                      (len(keeps) * len(deletes)))))
            results[keep][delete] = run_repetitions(reps,
                                                    keep,
                                                    delete,
                                                    print_progress=False)
            n_grid_items_compelete += 1

    return results

def plot_results(results, keeps, deletes, save_path_name):
    '''
    plots results in grid, and saves to png
    '''
    fig, axs = plt.subplots(len(keeps), len(deletes), sharex=True, sharey=True)
    fig.set_size_inches(30, 20)

    for i in range(len(keeps)):
        for j in range(len(deletes)):
            df = results[keeps[i]][deletes[j]]

            axs[i, j].plot(df.random, color = 'dodgerblue')
            axs[i, j].plot(df.uncertainty, color = 'orange')

            title = 'n_keep:' + str(keeps[i]) + ', '
            title += 'n_delete: ' + str(deletes[j])

            axs[i, j].set_title(title)

    # iterates over all subplots:
    for ax in axs.flat:
        ax.set(xlabel='n points added', ylabel='accuracy')
        ax.grid()
        #ax.legend()

        ax.label_outer(
        )  #hides x labels and tick labels for top plots and y ticks for right plots.

    #fig.legend()
    fig.savefig(save_path_name, dpi=200)


heart, heart_with_dummies = get_heart_data('input_data/heart.csv')

keeps = range(10, 60, 10)
deletes = range(0, 60, 10)
reps = 50
save_path_name = 'keep_delete_50_rep_grid.png'

results = run_experiments(keeps, deletes, reps)
plot_results(results, keeps, deletes, save_path_name)
