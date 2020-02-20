from input.heart_import import get_heart_data
from input.ads_import import get_ads_data
from alsRepeaterLauncher import AlsRepeaterLauncher
import alsDataManager
import sys  # to get arguments from terminal
import multiprocessing

# import matplotlib
# matplotlib.use('Agg')


if __name__ == '__main__':  # to avoid multiprocessor children to begin from start

    input_arguments = sys.argv

    params = alsDataManager.open_dict_from_json(input_arguments[1])
    launcher = AlsRepeaterLauncher(params)

    results = launcher.run_3_dimensional_varied_reps()

    save_path = 'output/jsons/' + input_arguments[1] + '.txt'
    alsDataManager.save_dict_as_json(output_dict=results, output_path=save_path)


"""
{"data_str": "heart", "als_input_dict": {"unsplit_data": null, "learning_method": null, "model_type": "lr", "seed": 0, "n_points_labeled_keep": 25, "n_points_labeled_delete": 25, "use_pca": false, "scale": true, "n_points_to_add_at_a_time": 50, "certainty_ratio_threshold": 2, "pct_unlabeled_to_label": 0.5, "pct_points_test": 0.25, "cores": 4}, "reps": 10, "n_jobs": 5, "argument_value_dict": {"learning_method": ["random", "uncertainty", "similar", "similar_uncertainty_optimization"], "certainty_ratio_threshold": [50], "n_points_labeled_delete": [300]}, "results": []}

"""