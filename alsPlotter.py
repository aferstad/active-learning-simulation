import matplotlib.pyplot as plt
import alsDataManager
import sys  # to get arguments from terminal
import pandas as pd

input_arguments = sys.argv

if len(input_arguments) > 1:
    json_path = 'output/jsons/' + input_arguments[1]
#else:
#    json_path = 'non_bayesian_xgboost_keep10_vary_delete_threshold.txt'

d = alsDataManager.open_dict_from_json(json_path)
d = d['results']  # NOTE: remove this line if json is in old format

"""
@param d : dictionary of experiment result data
    first key: learning_method
    second key:
"""

keys1 = list(d.keys())  # assumed to be learning methods
keys2 = list(d[keys1[0]].keys())  # assumed to be row variation
keys3 = list(d[keys1[0]][keys2[0]].keys())  # assumed to be column variation
keys4 = list(d[keys1[0]][keys2[0]][keys3[0]].keys())  # assumed to be performance metrics

n_rows = len(keys2)
n_cols = len(keys3)
print(n_rows)
print(n_cols)
methods = keys1
#metric = 'consistencies'  # keys4[0] #accuracy
max_x = 625
max_y = 0.92
min_y = 0.88
N_DELETED = None

metrics = ['accuracy', 'consistencies']
y_range_dict = {
    'accuracy' : [0.3, 0.9],
    'consistencies' : [0.5, 1]
}

for metric in metrics:

    min_y = y_range_dict[metric][0]
    max_y = y_range_dict[metric][1]

    TITLE_STR = '? Dataset, n_keep = ?, reps = ?, pct_unlabeled_labeled = ? ,' + metric
    save_path_name = 'output/plots/' + metric + '_plotted_' + input_arguments[1].split('.')[0] + '.png'

    fig, axs = plt.subplots(n_rows, n_cols)  # sharex=True, sharey=True)
    fig.set_size_inches(30, 20)

    grid_element_initialized = False

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_dict = dict(zip(methods, colors[:len(methods)]))

    for i in range(n_rows):
        for j in range(n_cols):
            grid_element_initialized = False
            if n_rows > 1 or n_cols > 1:
                current_ax = axs[i, j]
            else:
                current_ax = axs

            for method in methods:

                if not grid_element_initialized:
                    if 'delete' in keys2[0]:
                        n_deleted = int(keys2[i].split('_')[-1])  # get the number of deleted by splitting str
                    elif 'delete' in keys3[0]:
                        n_deleted = int(keys3[j].split('_')[-1])  # get the number of deleted by splitting str
                    else:
                        n_deleted = N_DELETED # TODO: alter this to be input parameter to plotting function

                    current_ax.set_xlim(0, max_x)
                    current_ax.set_ylim(min_y, max_y)

                    current_ax.axvline(x=n_deleted,
                                      color='maroon',
                                      alpha=0.5,
                                      label='# points deleted',
                                      linestyle='dotted')

                    current_ax.axhline(y=d[method][keys2[i]][keys3[j]][metric][0],
                                      color='green',
                                      alpha=0.5,
                                      label='intitial ' + metric,
                                      linestyle='dotted')

                    grid_element_initialized = True

                s = pd.Series(d[method][keys2[i]][keys3[j]][metric])
                s = s.rolling(10).mean()

                current_ax.plot(d[method][keys2[i]][keys3[j]][metric],
                                label='_'.join(method.split('_')[2:]),
                                alpha=0.25,
                                color=color_dict[method])
                current_ax.plot(s,
                                label='_'.join(method.split('_')[2:]),
                                alpha=0.75,
                                color=color_dict[method])

            title = keys2[i] + ' | ' + keys3[j]

            current_ax.set_title(title)


    if n_cols > 1 or n_rows > 1:
        # iterates over all subplots:
        for ax in axs.flat:
            ax.set(xlabel='n points added', ylabel=metric)
            # ax.grid()
        # ax.label_outer()  # hides x labels and tick labels for top plots and y ticks for right plots.
        handles, labels = axs[0, 0].get_legend_handles_labels()
    else:
        axs.set(xlabel='n points added', ylabel=metric)
        handles, labels = axs.get_legend_handles_labels()


    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='center right')

    fig.suptitle(TITLE_STR, fontsize=40)

    # fig.legend()
    fig.savefig(save_path_name, dpi=200)
