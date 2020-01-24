'''
KEEP-DELETE EXPERIMENTS
author: Andreas Ferstad
'''
import matplotlib
# Do not move this behind any other import, otherwise error will be given due to no display available
matplotlib.use('Agg')

import als
import importlib
importlib.reload(als)
from als import ALS

import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt


def run_repetitions(data,
                    reps,
                    keep,
                    delete,
                    methods,
                    print_progress=True,
                    use_pca=False,
                    scale=False,
                    n_points_to_add_at_a_time=1,
                    certainty_ratio_threshold=2):
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
    certainties = pd.DataFrame()

    for method in methods:
        accuracies[method] = []
        consistencies[method] = []

        for i in range(reps):
            print('Threshold: ' + str(certainty_ratio_threshold) + '| Keep: ' + str(keep) + ' | Delete: ' + str(delete) +
                  ' | Repetition: ' + str(i))
            if print_progress:
                print(i)

            my_experiment = ALS(
                unsplit_data=data,
                seed=i,
                model_type='lr',
                n_points_labeled_keep=keep,
                n_points_labeled_delete=delete,
                learning_method=method,
                use_pca=use_pca,
                scale=scale,
                n_points_to_add_at_a_time=n_points_to_add_at_a_time,
                certainty_ratio_threshold=certainty_ratio_threshold,
                pct_unlabeled_to_label=0.25)
            my_experiment.run_experiment()

            accuracies[method].append(my_experiment.accuracies)
            consistencies[method].append(my_experiment.consistencies)
            if method == 'similar_uncertainty_optimization':
                certainties = pd.concat(
                    [certainties, my_experiment.get_certainties()], axis=1)

        accuracies[method] = pd.DataFrame(accuracies[method]).mean(axis=0)
        consistencies[method] = pd.DataFrame(
            consistencies[method]).mean(axis=0)
        if method == 'similar_uncertainty_optimization':
            certainties = certainties.groupby(
                by=certainties.columns, axis=1).apply(lambda g: g.mean(axis=1))
            #certainties[
            #    'ratio'] = certainties.min_certainty_of_similar / certainties.min_certainty
            #print(certainties['ratio'])
            #certainties.loc[certainties.ratio > 200, 'ratio'] = 200
            # certainties becomes df with two columns,
            # ['min_certainty': mean_min_certainty of all repetitions,
            # min_certainty_of_similar = mean min certainty of all similar points ]

    accuracy_results = pd.concat(accuracies, axis=1)
    accuracy_results.columns = methods

    consistency_results = pd.concat(consistencies, axis=1)
    consistency_results.columns = methods

    return accuracy_results, consistency_results, certainties


def run_experiments(data,
                    reps,
                    keeps,
                    deletes,
                    methods,
                    use_pca=False,
                    scale=False,
                    n_points_to_add_at_a_time=1,
                    certainty_ratio_threshold=2):
    '''
    runs experiments for all combinations of keep and delete parameters
    returns 2 dataframes with accuracy results and consistency results
    '''
    accuracy_results = {}
    consistency_results = {}
    certainty_results = {}
    n_grid_items_complete = 0
    for keep in keeps:
        #print('keep = ' + str(keep))
        accuracy_results[keep] = {}
        consistency_results[keep] = {}
        certainty_results[keep] = {}
        for delete in deletes:
            print('Pct grid items complete: ' + str(
                round(100.0 * n_grid_items_complete /
                      (len(keeps) * len(deletes)))))
            accuracy_results[keep][delete], consistency_results[keep][
                delete], certainty_results[keep][delete] = run_repetitions(
                    data,
                    reps,
                    keep,
                    delete,
                    methods,
                    print_progress=False,
                    use_pca=use_pca,
                    scale=scale,
                    n_points_to_add_at_a_time=n_points_to_add_at_a_time,
                    certainty_ratio_threshold=certainty_ratio_threshold)
            n_grid_items_complete += 1

    return accuracy_results, consistency_results, certainty_results


def run_certainty_experiments(data,
                              reps,
                              keep,
                              delete,
                              methods,
                              use_pca=False,
                              scale=False,
                              n_points_to_add_at_a_time=1,
                              certainty_ratio_thresholds=[2]):
    '''
    runs experiments for all certainty_ratio_thresholds
    returns 3 dataframes with accuracy, consistency and certainty results
    '''
    accuracy_results = {}
    consistency_results = {}
    certainty_results = {}
    n_grid_items_complete = 0
    for certainty_ratio_threshold in certainty_ratio_thresholds:
        accuracy_results[certainty_ratio_threshold], consistency_results[
            certainty_ratio_threshold], certainty_results[
                certainty_ratio_threshold] = run_repetitions(
                    data,
                    reps,
                    keep,
                    delete,
                    methods,
                    print_progress=False,
                    use_pca=use_pca,
                    scale=scale,
                    n_points_to_add_at_a_time=n_points_to_add_at_a_time,
                    certainty_ratio_threshold=certainty_ratio_threshold)
        n_grid_items_complete += 1
    return accuracy_results, consistency_results, certainty_results


def plot_results(results,
                 reps,
                 keeps,
                 deletes,
                 save_path_name,
                 methods,
                 method_colors,
                 dataset_str,
                 ylabel,
                 run_time,
                 plot_certainties=False):
    '''
    plots results in grid, and saves to png
    '''
    fig, axs = plt.subplots(len(keeps), len(deletes), sharex=True, sharey=True)
    fig.set_size_inches(30, 20)

    for i in range(len(keeps)):
        for j in range(len(deletes)):
            df = results[keeps[i]][deletes[j]]
            if plot_certainties:
                for column in df.columns:
                    axs[i, j].plot(df[column], label=column)
            else:
                for method in methods:
                    axs[i, j].plot(df[method],
                                   color=method_colors[method],
                                   label=method + ' method')

                axs[i, j].axhline(y=df.random[0],
                                  color='green',
                                  alpha=0.5,
                                  label='intitial ' + str(ylabel))
            axs[i, j].axvline(x=deletes[j],
                              color='maroon',
                              alpha=0.5,
                              label='# points deleted')

            title = 'n_keep:' + str(keeps[i]) + ', '
            title += 'n_delete: ' + str(deletes[j])

            axs[i, j].set_title(title)

    # iterates over all subplots:
    for ax in axs.flat:
        ax.set(xlabel='n points added', ylabel=ylabel)
        ax.grid()
        #ax.legend()

        #ax.label_outer()  #hides x labels and tick labels for top plots and y ticks for right plots.

    handles, labels = axs[len(keeps) - 1,
                          len(deletes) - 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')

    title_string = 'Dataset: ' + dataset_str + ', repetitions per keep-delete pair: ' + str(
        reps) + ', minutes run time: ' + str(run_time)
    fig.suptitle(title_string)

    #fig.legend()
    fig.savefig(save_path_name, dpi=200)

import numpy as np

def plot_certainty_results(results,
                 reps,
                 keep,
                 delete,
                 save_path_name,
                 methods,
                # method_colors,
                 dataset_str,
                 ylabel,
                 run_time,
                 plot_certainties=False,
                 certainty_ratio_thresholds = [2]):
    '''
    INPUT:
        results : dictionary of dataframes to plot
            each element in certainty_ratio_thresholds is key in dictionary
            each dataframe has 3 columns: min_certainty, min_certainty_of_similar, ratio
    OUPUT:
        2x2 grid plot with each certainty_ratio_threshold
    '''
    height = 2
    width = 2

    fig, axs = plt.subplots(height, width, sharex=True, sharey=True) # Note: hardcoded for 2x2
    fig.set_size_inches(30, 20)

    current_index = 0
    current_threshold = None

    for i in range(height):
        for j in range(width):
            try:
                current_threshold = certainty_ratio_thresholds[current_index]
            except:
                break

            current_index = current_index + 1

            df = results[current_threshold]

            for column in df.columns:
                axs[i, j].plot(df[column], label = column)


            #axs[i, j].plot(results[current_threshold].min_certainty,
            #               label ='min_certainty')
            #axs[i, j].plot(results[current_threshold].min_certainty_of_similar,
            #               label ='min_certainty_of_similar')
            axs[i, j].axvline(x=delete,
                              color='black',
                              alpha=0.5,
                              label='# points deleted')
            axs[i, j].axhline(y=df.iloc[0,0],
                              color='purple',
                              alpha=0.5,
                              label='intitial ' + str(ylabel))

            title = 'Current Threshold: ' + str(current_threshold)
            axs[i, j].set_title(title)



    # iterates over all subplots:
    for ax in axs.flat:
        ax.set(xlabel='n points added', ylabel=ylabel)
        ax.grid()
        #ax.legend()

        #ax.label_outer()  #hides x labels and tick labels for top plots and y ticks for right plots.

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')

    title_string = 'Dataset: ' + dataset_str + ', repetitions per keep-delete pair: ' + str(
        reps) + ', minutes run time: ' + str(run_time)
    fig.suptitle(title_string)

    #fig.legend()
    fig.savefig(save_path_name, dpi=200)