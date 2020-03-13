if __name__ == '__main__':
    from xgboost.sklearn import XGBClassifier
    from sklearn.model_selection import GridSearchCV
    import numpy as np
    from input.voice.voice_import import get_voice_data
    import alsDataManager


    def cv_grid(param_space, model, X, y, cv_folds = 5):
        try:
            grid_search = GridSearchCV(estimator=model, param_grid=param_space, scoring='accuracy', n_jobs=4, cv=cv_folds,
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


    data = get_voice_data(n_components=300)

    train = data.sample(n=200, random_state=1)
    X = train.iloc[:,1:]
    y = train.iloc[:,0]

    n_initial_estimators = 100

    model = XGBClassifier(learning_rate=0.1,
                n_estimators=n_initial_estimators,
                max_depth=5,
                min_child_weight=1,
                gamma=0,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='multi:softmax',
                nthread=4,
                scale_pos_weight=1,
                seed=0)

    # max_depth and min_child_weight tuning
    param_space = {
        'max_depth': list(range(1, 20, 3)),
        'min_child_weight': list(range(1, 20, 3))
    }
    model = cv_grid(param_space, model, X,y)

    tuned_params_dict = model.get_xgb_params()
    alsDataManager.save_dict_as_json(output_dict=tuned_params_dict, output_path='xgboost_tuned_params.txt')

    # gamma tuning
    param_space = {
        'gamma': np.linspace(0, 1, 5)
    }
    model = cv_grid(param_space, model, X,y)

    tuned_params_dict = model.get_xgb_params()
    alsDataManager.save_dict_as_json(output_dict=tuned_params_dict, output_path='xgboost_tuned_params.txt')

    # subsample and colsample_bytree tuning
    param_space = {
        'subsample': np.linspace(0, 1, 5),
        'colsample_bytree': np.linspace(0, 1, 5)
    }
    model = cv_grid(param_space, model, X, y)

    tuned_params_dict = model.get_xgb_params()
    alsDataManager.save_dict_as_json(output_dict=tuned_params_dict, output_path='xgboost_tuned_params.txt')

    # reg_alpha tuning
    param_space = {
        'reg_alpha': [1e-10, 1e-5, 1e-2, 1, 100]
    }
    model = cv_grid(param_space, model, X, y)

    tuned_params_dict = model.get_xgb_params()
    alsDataManager.save_dict_as_json(output_dict=tuned_params_dict, output_path='xgboost_tuned_params.txt')

    # set learning rate low and re-tune n_estimators
    model.set_params(learning_rate=0.01)

    param_space = {
        'n_estimators': list(range(100, 5000, 500)),
    }
    model = cv_grid(param_space, model, X,y)

    tuned_params_dict = model.get_xgb_params()
    alsDataManager.save_dict_as_json(output_dict=tuned_params_dict, output_path='xgboost_tuned_params.txt')













