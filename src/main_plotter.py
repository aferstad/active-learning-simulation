from final_plotter import create_plot
import alsDataManager

## ADS LR

# create ts and no_ts data for lr ads
no_ts = 'final_experiments/ads_lr/ads_lr_no_ts_final.txt.txt'
no_ts = alsDataManager.open_dict_from_json(no_ts)

ts = 'final_experiments/ads_lr/ads_lr_ts_final.txt.txt'
ts = alsDataManager.open_dict_from_json(ts)

no_ts = no_ts['results']
ts = ts['results']

y_pos_change = {
    'accuracy' : {
        'random': 0.00,
        'similar': -0.002,
        'uncertainty': 0.00,
        'threshold_1000': 0.00,
        'threshold_100': -0.0005
    },
    'consistencies' : {
        'uncertainty': 0.00,
        'threshold_1000': -0.002,
        'threshold_100': 0.00,
        'similar': 0.00,
    }
}
metrics = ['accuracy', 'consistencies']
create_plot(save_path = 'ads_lr',
                ts = ts,
                no_ts = no_ts,
                exclusion_thresholds = ['threshold_5', 'threshold_50', 'threshold_500'],
                n_deleted = 300,
                rolling_window_size = 30,
                max_x = 500,
                max_y = dict(zip(metrics, [0.89, 0.97])),
                min_y = dict(zip(metrics, [0.84, 0.92])),
                y_pos_change = y_pos_change)




### ADS XGBOOST

no_ts_path = 'final_experiments/ads_xgboost/ads_xgboost_r20_no_ts.txt.txt'
ts_path = 'final_experiments/ads_xgboost/ads_xgboost_r20_t100_1000.txt.txt'
no_ts = alsDataManager.open_dict_from_json(no_ts_path)['results']
ts = alsDataManager.open_dict_from_json(ts_path)['results']

save_path = 'ads_xgboost'
n_deleted = 300
rolling_window_size = 30
max_x = 500
metrics = ['accuracy', 'consistencies']
max_y = dict(zip(metrics, [0.88, 0.93]))
min_y = dict(zip(metrics, [0.81, 0.88]))
y_pos_change = {
    'accuracy' : {
        'random': 0.001,
        'similar': 0.00,
        'uncertainty': -0.0005,
        'threshold_1000': 0.00,
        'threshold_100': +0.001
    },
    'consistencies' : {
        'uncertainty': -0.001,
        'threshold_100': -0.001,
        'similar': -0.001,
    }
}
exclusion_thresholds = []
create_plot(save_path = save_path,
                ts = ts,
                no_ts = no_ts,
                exclusion_thresholds = exclusion_thresholds,
                n_deleted = n_deleted,
                rolling_window_size = rolling_window_size,
                max_x = max_x,
                max_y = max_y,
                min_y = min_y,
                y_pos_change = y_pos_change)

### VOICE LR

# create ts and no_ts data for lr voice
no_ts = 'final_experiments/voice_lr/lr_voice_merged.txt.txt'
no_ts = alsDataManager.open_dict_from_json(no_ts)

ts = 'final_experiments/voice_lr/lr_voice_ts.txt.txt'
ts = alsDataManager.open_dict_from_json(ts)

key1 = 'results'
key2 = 'learning_method_similar_uncertainty_optimization'
key3 = 'certainty_ratio_threshold_5'

ts[key1][key2][key3] = no_ts[key1][key2][key3]
del no_ts[key1][key2]

no_ts = no_ts['results']
ts = ts['results']

save_path = 'voice_lr'
n_deleted = 300
rolling_window_size = 30
max_x = 400
metrics = ['accuracy', 'consistencies']
max_y = dict(zip(metrics, [0.91, 0.90]))
min_y = dict(zip(metrics, [0.83, 0.86]))

y_pos_change = {
    'accuracy' : {
        'random': 0.00,
        'similar': 0.00,
        'uncertainty': 0.00,
        'threshold_3': -0.002,
        'threshold_100': 0.000
    },
    'consistencies' : {
        'uncertainty': -0.001,
        'threshold_1000': 0.00,
        'threshold_100': -0.001,
        'similar': 0.00,
    }
}

exclusion_thresholds = ['threshold_1.5', 'threshold_4', 'threshold_5']

create_plot(save_path = save_path,
                ts = ts,
                no_ts = no_ts,
                exclusion_thresholds = exclusion_thresholds,
                n_deleted = n_deleted,
                rolling_window_size = rolling_window_size,
                max_x = max_x,
                max_y = max_y,
                min_y = min_y,
                y_pos_change = y_pos_change)


### VOICE XGBOOST

# create ts and no_ts data for xgboost voice
no_ts = 'final_experiments/voice_xgboost/data/xgboost_voice_merged.txt'
no_ts = alsDataManager.open_dict_from_json(no_ts)

paths = [
    'final_experiments/voice_xgboost/data/xgboost_voice_t2.txt.txt',
    'final_experiments/voice_xgboost/data/xgboost_voice_t3.txt.txt',
    'final_experiments/voice_xgboost/data/xgboost_voice_t10.txt.txt',
    'final_experiments/voice_xgboost/data/xgboost_voice_t20.txt.txt'
]
no_ts = no_ts['results']
ds = {}
ds['learning_method_similar_uncertainty_optimization'] = {}
for path in paths:
    d = alsDataManager.open_dict_from_json(path)['results']['learning_method_similar_uncertainty_optimization']
    key = list(d.keys())[0]
    ds['learning_method_similar_uncertainty_optimization'][key] = d[key]
ts = ds
key2 = 'learning_method_similar_uncertainty_optimization'
key3 = 'certainty_ratio_threshold_5'
ts[key2][key3] = no_ts[key2][key3]
del no_ts[key2]

save_path = 'voice_xgboost'
n_deleted = 300
rolling_window_size = 30
max_x = 400
metrics = ['accuracy', 'consistencies']
max_y = dict(zip(metrics, [0.87, 0.85]))
min_y = dict(zip(metrics, [0.72, 0.77]))
exclusion_thresholds = ['threshold_10', 'threshold_20', 'threshold_5']




y_pos_change = {
    'accuracy' : {
        'random': -0.005,
        'similar': 0.00,
        'uncertainty': -0.006,
        'threshold_2': 0.00,
        'threshold_3': -0.001
    },
    'consistencies' : {
        'uncertainty': 0.00,
        'threshold_2': 0.0005,
        'threshold_3': -0.001,
        'similar': 0.002,
    }
}

create_plot(save_path = save_path,
                ts = ts,
                no_ts = no_ts,
                exclusion_thresholds = exclusion_thresholds,
                n_deleted = n_deleted,
                rolling_window_size = rolling_window_size,
                max_x = max_x,
                max_y = max_y,
                min_y = min_y,
                y_pos_change = y_pos_change)


# HEART LR

# create ts and no_ts data for heart lr
no_ts = 'final_experiments/heart_lr/heart_lr.txt.txt'
no_ts = alsDataManager.open_dict_from_json(no_ts)

ts = 'final_experiments/heart_lr/heart_lr.txt.txt'
ts = alsDataManager.open_dict_from_json(ts)

key1 = 'results'
key2 = 'learning_method_similar_uncertainty_optimization'
#key3 = 'certainty_ratio_threshold_5'

#ts[key1][key2][key3] = no_ts[key1][key2][key3]
del no_ts[key1][key2]

no_ts = no_ts['results']
ts = ts['results']

save_path = 'heart_lr'
n_deleted = 40
rolling_window_size = 10
max_x = 75
metrics = ['accuracy', 'consistencies']
max_y = dict(zip(metrics, [0.82, 0.86]))
min_y = dict(zip(metrics, [0.76, 0.82]))
exclusion_thresholds = ['threshold_75', 'threshold_50', 'threshold_25']



y_pos_change = {
    'accuracy' : {
        'random': -0.00,
        'similar': -0.001,
        'uncertainty': 0.001,
        'threshold_5': -0.001,
        'threshold_100': -0.00
    },
    'consistencies' : {
        'uncertainty': -0.001 ,
        'threshold_2': 0.000,
        'threshold_100': -0.0013,
        'similar': 0.00,
        'random': -0.002
    }
}

create_plot(save_path = save_path,
                ts = ts,
                no_ts = no_ts,
                exclusion_thresholds = exclusion_thresholds,
                n_deleted = n_deleted,
                rolling_window_size = rolling_window_size,
                max_x = max_x,
                max_y = max_y,
                min_y = min_y,
                y_pos_change = y_pos_change)



# HEART XGBOOST


path = 'final_experiments/heart_xgboost/heart_xgboost_final.txt.txt'

d = alsDataManager.open_dict_from_json(path)['results']

ts = {}
ts['learning_method_similar_uncertainty_optimization'] = d['learning_method_similar_uncertainty_optimization']

no_ts = d
del no_ts['learning_method_similar_uncertainty_optimization']

save_path = 'heart_xgboost'
n_deleted = 40
rolling_window_size = 20
lw = 2
max_x = 75
min_x = 0
metrics = ['accuracy', 'consistencies']

max_y = dict(zip(metrics, [0.79, 0.82]))
min_y = dict(zip(metrics, [0.72, 0.78]))

tick_distance_dict = {
    'consistencies' : 1,
    'accuracy' : 1
}


y_pos_change = {
    'accuracy' : {
        'random': -0.0045,
        'similar': 0.0025,
        'uncertainty': -0.0015,
        'threshold_5': -0.00,
        'threshold_100': 0.002
    },
    'consistencies' : {
        'uncertainty': -0.00,
        'threshold_5': 0.002
    }
}

#exclusion_thresholds = ['threshold_5', 'threshold_50', 'threshold_500']

exclusion_thresholds = ['threshold_25', 'threshold_50', 'threshold_75']


create_plot(save_path = save_path,
                ts = ts,
                no_ts = no_ts,
                exclusion_thresholds = exclusion_thresholds,
                n_deleted = n_deleted,
                rolling_window_size = rolling_window_size,
                max_x = max_x,
                max_y = max_y,
                min_y = min_y,
                y_pos_change = y_pos_change)
