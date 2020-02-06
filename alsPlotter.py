import matplotlib.pyplot as plt
import alsDataManager

json_path = 'non_bayesian_thresholding.txt'
save_path_name = json_path.split('.')[0] + '.png'

d = alsDataManager.open_dict_from_json(json_path)

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
methods = keys1
metric = 'accuracy'  # keys4[0]
max_x = 60
max_y = 0.84
min_y = 0.76

fig, axs = plt.subplots(n_rows, n_cols)  # sharex=True, sharey=True)
fig.set_size_inches(30, 20)

for i in range(n_rows):
    for j in range(n_cols):
        grid_element_initialized = False
        for method in methods:
            if not grid_element_initialized:
                if 'delete' in keys2[0]:
                    n_deleted = int(keys2[i].split('_')[-1])  # get the number of deleted by splitting str
                elif 'delete' in keys3[0]:
                    n_deleted = int(keys3[j].split('_')[-1])  # get the number of deleted by splitting str
                else:
                    n_deleted = 0  # TODO: alter this to be input parameter to plotting function

                axs[i, j].set_xlim(0, max_x)
                axs[i, j].set_ylim(min_y, max_y)

                axs[i, j].axvline(x=n_deleted,
                                  color='maroon',
                                  alpha=0.5,
                                  label='# points deleted',
                                  linestyle='dotted')

                axs[i, j].axhline(y=d[method][keys2[i]][keys3[j]][metric][0],
                                  color='green',
                                  alpha=0.5,
                                  label='intitial ' + metric,
                                  linestyle='dotted')

            axs[i, j].plot(d[method][keys2[i]][keys3[j]][metric],
                           # color=method_colors[method],
                           label='_'.join(method.split('_')[2:]))

        title = keys2[i] + ' | ' + keys3[j]

        axs[i, j].set_title(title)

# iterates over all subplots:
for ax in axs.flat:
    ax.set(xlabel='n points added', ylabel=metric)
    # ax.grid()
# ax.label_outer()  # hides x labels and tick labels for top plots and y ticks for right plots.

handles, labels = axs[0, 0].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(), loc='center right')

# title_string = 'Dataset: ' + dataset_str + ', repetitions per keep-delete pair: ' + str(
#    reps) + ', minutes run time: ' + str(run_time)
# fig.suptitle(title_string)

# fig.legend()
fig.savefig(save_path_name, dpi=200)
