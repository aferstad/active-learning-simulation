import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import pandas as pd


def __define_color_dict(exclusion_thresholds, methods_ts, upper_color=1):
    # create color_dict
    n_cmap_lines = 2 + len(methods_ts) - len(exclusion_thresholds)
    cm_subsection = np.linspace(0, upper_color, n_cmap_lines)

    colors = [cm.winter(x) for x in cm_subsection]

    color_dict = {
        'learning_method_random': (101 / 255, 101 / 255, 101 / 255, 1)
    }
    label_dict = {
        'random': 'random',
        'uncertainty': 'uncertainty, k=1',
        'similar': 'similar, k=∞'
    }

    colors = [
        (230 / 255, 7 / 255, 26 / 255),
        (222 / 255, 122 / 255, 160 / 255, 1),
        (141 / 255, 204 / 255, 235 / 255, 1),
        (27 / 255, 141 / 255, 204 / 255, 1)
    ]

    colors = [
        (243 / 255, 56 / 255, 41 / 255, 1),
        (255 / 255, 119 / 255, 81 / 255, 1),
        (14 / 255, 185 / 255, 203 / 255, 1),
        (2 / 255, 95 / 255, 106 / 255, 1)
    ]

    # Google Colors:
    colors = [
        (219.0 / 255, 68.0 / 255, 55.0 / 255, 1),
        (244 / 255.0, 180 / 255.0, 0 / 255.0, 1),
        (15 / 255.0, 157 / 255.0, 88 / 255.0, 1),
        (66 / 255.0, 133 / 255.0, 244 / 255.0, 1)
    ]

    color_dict['learning_method_uncertainty'] = colors[0]
    color_dict['learning_method_similar'] = colors[-1]

    i = 0
    for method in methods_ts:
        label = '_'.join(method.split('_')[2:])
        if label in exclusion_thresholds:
            continue
        color_dict[method] = colors[i + 1]
        k = method.split('_')[-1]
        label_dict['_'.join(method.split('_')[2:])] = 'thresholding, k=' + k
        i += 1

    return color_dict, label_dict


# xg voice
def __plot_ac(save_path,
              n_deleted,
              rolling_window_size,
              max_x,
              max_y,
              min_y,
              exclusion_thresholds,
              y_pos_change,
              df,
              methods_ts):
    methods_no_ts = ['learning_method_similar', 'learning_method_uncertainty', 'learning_method_random']
    min_x = 0
    lw = 3
    color_dict, label_dict = __define_color_dict(exclusion_thresholds = exclusion_thresholds, methods_ts=methods_ts)

    tick_distance_dict = {
        'consistencies': 1,
        'accuracy': 1
    }
    metrics = ['accuracy', 'consistencies']

    for metric in metrics:
        width = 3 * 2.54
        height = 3 * 2.54
        plt.figure(figsize=(width, height))

        # Remove the plot frame lines
        ax = plt.subplot(111)
        ax.spines["top"].set_visible(False)
        # ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # ax.spines["left"].set_visible(False)

        label = 'points deleted'  # TODO: maybe add this label to the chart?
        plt.axvline(x=n_deleted,
                    color=color_dict['learning_method_random'],
                    alpha=1,
                    lw = 1.5,
                    label=label,
                    linestyle='dotted')

        if metric == 'accuracy':
            initial_y = df[metric][methods_no_ts[0]][0]
            label = 'initial accuracy'
            plt.axhline(y=initial_y,
                        color=color_dict['learning_method_random'],
                        alpha=1,
                        lw = 1.5,
                        label=label,
                        linestyle='dotted')

            plt.text(max_x, initial_y - 0.0005, label, color='gray', fontsize=14, fontstyle='italic')

        for i, method in enumerate(df[metric]):
            label = '_'.join(method.split('_')[2:])
            if label in exclusion_thresholds:
                continue
            s = pd.Series(df[metric][method])
            s_smooth = s.rolling(rolling_window_size, center=True).mean()
            plt.plot(s,
                     label=label,
                     lw=1.5,
                     alpha=0.2,
                     color=color_dict[method])
            plt.plot(list(s_smooth),
                     label=label,
                     lw=lw,
                     alpha=1,
                     color=color_dict[method])
            #y_pos = list(s_smooth)[-rolling_window_size]
            if save_path == 'voice_lr' or save_path == 'voice_xgboost' or save_path == 'heart_lr':
                y_pos = list(s)[max_x]
            else:
                y_pos = list(s_smooth)[-rolling_window_size]

            if save_path == 'heart_xgboost':
                y_pos = list(s_smooth)[max_x]

            change = y_pos_change[metric].get(label, 0)
            y_pos += change

            if metric == 'accuracy':
                plt.text(max_x+3, y_pos, label_dict[label], color=color_dict[method], fontsize=14, weight='bold')
            else:
                plt.text(max_x+3, y_pos, label_dict[label], color=color_dict[method], fontsize=14, weight='bold')

        plt.xlabel('Points Labeled After Data Deletion', fontsize=14, weight='bold')

        plt.ylabel('Model ' + metric, fontsize=14, weight='bold')
        if metric == 'consistencies':
            plt.ylabel('Model ' + 'Consistency', fontsize=14, weight='bold')
        else:
            plt.ylabel('Model ' + 'Accuracy', fontsize=14, weight='bold')


        plt.xlim(min_x, max_x)
        plt.ylim(min_y[metric], max_y[metric])

        # label ='points deleted = points added'
        # if metric == 'accuracy':
        #    plt.text(N_DELETED+3, max_y-0.002, label, color='gray', fontsize = 14)
        #    pass
        # else:
        #    plt.text(N_DELETED+3, max_y-0.001, label, color='gray', fontsize = 14)
        #    pass
        # manipulate
        vals = ax.get_yticks()
        ax.set_yticklabels(['{:,.1%}'.format(x) for x in vals])
        plt.tight_layout()

        tick_distances = tick_distance_dict[metric]
        ytick_positions = np.array(
            (range(int(min_y[metric] * 100), int(max_y[metric] * 100) + 1, tick_distances))) / 100

        plt.yticks(ytick_positions, [str(int(x * 100)) + "%" for x in ytick_positions], fontsize=14)

        start, end = ax.get_xlim()

        if save_path == 'heart_lr' or save_path == 'heart_xgboost':
            ax.xaxis.set_ticks(np.arange(start, end + 25, 25))
        else:
            ax.xaxis.set_ticks(np.arange(start, end + 100, 100))

        plt.xticks(fontsize=14)
        # plt.legend()
        if metric == 'consistencies':
            plt.savefig(save_path + '_c', dpi=200)
        else:
            plt.savefig(save_path + '_a', dpi=200)


def __merge_ts_no_ts(ts, no_ts):
    d = no_ts

    keys1 = list(d.keys())  # assumed to be learning methods
    keys2 = list(d[keys1[0]].keys())  # assumed to be row variation
    keys3 = list(d[keys1[0]][keys2[0]].keys())  # assumed to be column variation

    methods_no_ts = list(no_ts.keys())

    # remove methods_ts here if you have too many
    methods_ts = list(ts['learning_method_similar_uncertainty_optimization'].keys())

    metrics = ['accuracy', 'consistencies']

    df = {}

    for metric in metrics:
        df[metric] = {}
        for method in methods_no_ts:
            df[metric][method] = no_ts[method][keys2[0]][keys3[0]][metric]
        for method in methods_ts:
            df[metric][method] = ts['learning_method_similar_uncertainty_optimization'][method][keys3[0]][metric]

    return df, methods_ts


def create_plot(save_path,
                ts,
                no_ts,
                exclusion_thresholds,
                n_deleted,
                rolling_window_size,
                max_x,
                max_y,
                min_y,
                y_pos_change):
    df, methods_ts = __merge_ts_no_ts(ts=ts, no_ts=no_ts)

    __plot_ac(save_path = save_path,
              n_deleted = n_deleted,
              rolling_window_size = rolling_window_size,
              max_x = max_x,
              max_y = max_y,
              min_y = min_y,
              exclusion_thresholds = exclusion_thresholds,
              y_pos_change = y_pos_change,
              df = df,
              methods_ts = methods_ts)




