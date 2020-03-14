
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from xgboost.core import XGBoostError
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import matplotlib.pylab as plt
import numpy as np


def cv_n_estimators(model, X, y, cv_folds=5, early_stopping_rounds=50):
    model_params = model.get_xgb_params()  # returns all parameters of model

    # converts data to xg matrix
    xgtrain = xgb.DMatrix(X.values, label=y.values)


    try:
        cvresult = xgb.cv(model_params,
                          xgtrain,
                          num_boost_round=model.get_params()['n_estimators'],
                          nfold=cv_folds,
                          metrics='auc',
                          early_stopping_rounds=early_stopping_rounds)
    except XGBoostError:
        if cv_folds > 2:
            return cv_n_estimators(model, X, y, cv_folds=cv_folds-1)
        else:
            print('2-FOLD DATASET CONTAINS TOO FEW POS OR NEG SAMPLES. NO TUNING PERFORMED')
            return model

    model.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm on the data
    model.fit(X, y, eval_metric='auc')

    # print('n estimators set to : ' + str(model.get_num_boosting_rounds()))
    return model


def print_performance(model, X, y):
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]

    # Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(y.values, y_pred))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(y.values, y_pred_proba))

    feat_imp = pd.Series(model.get_booster().get_score(importance_type='gain')).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')


def cv_grid(param_space, model, X, y, cv_folds = 5):
    try:
        grid_search = GridSearchCV(estimator=model, param_grid=param_space, scoring='roc_auc', n_jobs=1, cv=cv_folds,
                                   refit=True)
        grid_search.fit(X, y)
    except ValueError:
        if cv_folds > 2:
            return cv_grid(param_space, model, X, y, cv_folds=cv_folds-1)
        else:
            print('2-FOLD DATASET CONTAINS TOO FEW POS OR NEG SAMPLES. NO TUNING PERFORMED')
            return model

    # print(grid_search.best_params_)

    return grid_search.best_estimator_


def fit_and_tune_xgboost(model, X, y):
    # n_estimators tuning
    model = cv_n_estimators(model, X, y)

    # max_depth and min_child_weight tuning
    param_space = {
        'max_depth': list(range(1, 3, 1)),
        'min_child_weight': np.linspace(0, 1, 10)  # list(range(0,10,1))
    }
    model = cv_grid(param_space, model, X,y)

    # gamma tuning
    param_space = {
        'gamma': np.linspace(0, 0.2, 10)
    }
    model = cv_grid(param_space, model, X,y)

    # n_estimator re-tuning
    model = cv_n_estimators(model, X, y)

    # subsample and colsample_bytree tuning
    param_space = {
        'subsample': np.linspace(0.5, 1, 10),
        'colsample_bytree': np.linspace(0, 0.5, 10)
    }
    model = cv_grid(param_space, model, X, y)

    # reg_alpha tuning
    param_space = {
        'reg_alpha': [0, 1e-13, 1e-8, 1e-5, 1e-2]
    }
    model = cv_grid(param_space, model, X, y)

    # set learning rate low and re-tune n_estimators
    model.set_params(learning_rate=0.01, n_estimators=5000)
    model = cv_n_estimators(model, X, y)

    print('fit complete')
    return model


class XGBoostModel:

    def __init__(self, n_classes=2, data_str=None):
        # TODO: decide what nthread should be?
        if n_classes > 2 and data_str == 'voice':
            self.model = XGBClassifier(learning_rate=0.1,
                                  n_estimators=650,
                                  max_depth=5,
                                  min_child_weight=1,
                                  gamma=0,
                                  subsample=0.8,
                                  colsample_bytree=0.8,
                                  objective='multi:softmax',
                                  nthread=8,
                                  scale_pos_weight=1,
                                  seed=0)

            # parameters found by tuning on voice data with 400 points:
            self.model.set_params(**{"base_score": 0.5, "booster": "gbtree", "colsample_bylevel": 1, "colsample_bynode": 1,
                                "colsample_bytree": 0.25, "gamma": 0.0, "learning_rate": 0.01, "max_delta_step": 0,
                                "max_depth": 2, "min_child_weight": 1, "n_estimators": 3600, "nthread": 4,
                                "objective": "multi:softprob", "reg_alpha": 1e-05, "reg_lambda": 1,
                                "scale_pos_weight": 1, "seed": 0, "subsample": 0.75, "verbosity": 1})
        elif data_str == 'heart':
            self.model = XGBClassifier(learning_rate=0.1,
                          n_estimators=20,
                          max_depth=2,
                          min_child_weight=0.3,
                          gamma=0,
                          subsample=0.65,
                          colsample_bytree=0.15,
                          reg_alpha=0,
                          objective='binary:logistic',
                          nthread=8,
                          scale_pos_weight=1,
                          seed=0)
        elif data_str == 'ads':
            print('TODO: tune xgboost on ads! not done yet?')

            # TODO: tune model on heart and ads data too
            self.model = XGBClassifier(
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
        else:
            print('ERROR in xgBoostModel, only data_str in (voice, heart, ads) supported. Please add tuning parameters to file')

    def fit(self, x, y, with_tuning=False):
        self.model.fit(x,y)

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        return self.model.predict_proba(x)