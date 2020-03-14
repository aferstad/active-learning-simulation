if __name__ == '__main__':
    from xgboost.sklearn import XGBClassifier
    from sklearn.model_selection import GridSearchCV
    import numpy as np
    from input.voice.voice_import import get_voice_data
    from input.heart.heart_import import get_heart_data

    import alsDataManager


    def cv_grid(param_space, model, X, y, cv_folds = 5):
        try:
            grid_search = GridSearchCV(estimator=model, param_grid=param_space, scoring='f1_macro', n_jobs=4, cv=cv_folds,
                                       refit=True)
            grid_search.fit(X, y)
        except ValueError:
            if cv_folds > 2:
                return cv_grid(param_space, model, X, y, cv_folds=cv_folds-1)
            else:
                print('2-FOLD DATASET CONTAINS TOO FEW POS OR NEG SAMPLES. NO TUNING PERFORMED')
                return model

        print('Parameter Space: ' + str(param_space))
        print('Best Params: ' + str(grid_search.best_params_))
        return grid_search.best_estimator_


    data_str = 'heart'

    if data_str == 'voice':
        data = get_voice_data(n_components=300)

        train = data.sample(n=400, random_state=0)
        X = train.iloc[:,1:]
        y = train.iloc[:,0]

        model = XGBClassifier(learning_rate=0.1,
                    n_estimators=5000,
                    max_depth=5,
                    min_child_weight=1,
                    gamma=0,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='multi:softmax',
                    nthread=4,
                    scale_pos_weight=1,
                    seed=0)

        param_spaces = [
            {
                'n_estimators': list(range(500, 3000, 500)),
            },
            {
                'max_depth': list(range(1, 5, 1)),
                'min_child_weight': list(range(0, 5, 1))
            },
            {
                'gamma': np.linspace(0, 0.5, 5)
            },
            {
                'n_estimators': list(range(500, 3000, 500)),
            },
            {
                'subsample': np.linspace(0, 1, 5),
                'colsample_bytree': np.linspace(0, 1, 5)
            },
            {
                'reg_alpha': [0, 1e-10, 1e-5, 1e-2]
            },
            {
                'learning_rate': [0.01] # set learning rate low
            },
            {
                'n_estimators': list(range(100, 5000, 500))
            }
        ]
    elif data_str == 'heart':
        data = get_heart_data()

        train = data.sample(n=20, random_state=0)
        X = train.iloc[:, 1:]
        y = train.iloc[:, 0]

        model = XGBClassifier(
                learning_rate=0.1,
                n_estimators=1000,
                max_depth=5,
                min_child_weight=1,
                gamma=0,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='binary:logistic',
                nthread=8,
                scale_pos_weight=1,
                seed=0)

        param_spaces = [
            {
                'n_estimators': list(range(500, 3000, 500)),
            },
            {
                'max_depth': list(range(1, 5, 1)),
                'min_child_weight': list(range(0, 5, 1))
            },
            {
                'gamma': np.linspace(0, 0.5, 5)
            },
            {
                'n_estimators': list(range(500, 3000, 500)),
            },
            {
                'subsample': np.linspace(0, 1, 5),
                'colsample_bytree': np.linspace(0, 1, 5)
            },
            {
                'reg_alpha': [0, 1e-10, 1e-5, 1e-2]
            },
            {
                'learning_rate': [0.01]  # set learning rate low
            },
            {
                'n_estimators': list(range(100, 5000, 500))
            }
        ]

    for i, param_space in enumerate(param_spaces):
        print('pct spaces tuned: ' + str(1.*i/len(param_space)))
        model = cv_grid(param_space, model, X, y)

        tuned_params_dict = model.get_xgb_params()
        alsDataManager.save_dict_as_json(output_dict=tuned_params_dict, output_path='xgboost_tuned_params.txt')





