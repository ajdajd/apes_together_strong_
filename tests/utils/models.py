from catboost import CatBoostClassifier, FeaturesData, Pool, CatBoostRegressor
import lightgbm as lgbm
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import utils
import gc; gc.enable()

class CatBoostCV():

    def __init__(self, cv=None, obj='binary', cats=None, nums=None, **kwargs):
        self.cv = cv
        self.cb_params = kwargs
        self.nums = nums
        self.cats = cats
        self.obj = obj
        self.metric = str(kwargs['eval_metric'])

    def fit(self, X, y=None, **kwargs):

        self.models_ = []
        feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else list(range(X.shape[1]))
       # cat_features = X.select_dtypes(include=['category', 'object']).columns.tolist()
        self.feature_importances_ = pd.DataFrame()
        self.model_scores_ = []
        self.model_best_iterations_ = []

        for i, (fit_idx, val_idx) in enumerate(self.cv):

            # Split the dataset according to the fold indexes
            if isinstance(X, pd.DataFrame):
                X_fit = X.iloc[fit_idx]
                X_val = X.iloc[val_idx]
            else:
                X_fit = X[fit_idx]
                X_val = X[val_idx]

            if isinstance(y, pd.Series):
                y_fit = y.iloc[fit_idx]
                y_val = y.iloc[val_idx]
            else:
                y_fit = y[fit_idx]
                y_val = y[val_idx]
            
            # https://lightgbm.readthedocs.io/en/latest/Python-API.html#lightgbm.Dataset
            fit_set = Pool(
                data=FeaturesData(

                    num_feature_data=X_fit[self.nums].values,
                    cat_feature_data=X_fit[self.cats].values,
                    num_feature_names=self.nums,
                    cat_feature_names=self.cats
                ), label=y_fit) 
            
            val_set = Pool(
                data=FeaturesData(

                    num_feature_data=X_val[self.nums].values,
                    cat_feature_data=X_val[self.cats].values,
                    num_feature_names=self.nums,
                    cat_feature_names=self.cats
                ), label=y_val)

            if self.obj == 'binary':    
                model = CatBoostClassifier(
                    **self.cb_params
                )
            else:
                model = CatBoostRegressor(
                    **self.cb_params
                )

            
            model.fit(
                fit_set,
                eval_set=val_set,
                **kwargs
            )
            #print(f'fit set {len(X_fit)} and val_set {len(X_val)}')
            # https://lightgbm.readthedocs.io/en/latest/Python-API.html#lightgbm.train

            # Store the feature importances
            if i == 0:
                self.feature_importances_['feature_names'] = model.feature_names_
            self.feature_importances_['importance_{}'.format(i)] = model.get_feature_importance()
            self.model_scores_.append(model.best_score_['validation'][self.metric])
            self.model_best_iterations_.append(model.best_iteration_)
            
            # Store the model
            self.models_.append(model)
            del fit_set, val_set, X_fit, y_fit ,X_val, y_val, model
            gc.collect()
        

        return self

    def predict(self, X):

        utils.validation.check_is_fitted(self, ['models_'])
        # if pandas
        try:
            y = np.zeros(len(X))
        except:
            y = np.zeros(X.shape[0])

        for model in self.models_:
            if self.obj == 'binary':
                y += model.predict_proba(X)[:,1]
            else:
                y += model.predict(X)

        return y / len(self.models_)

import lightgbm as lgbm
from sklearn import utils
class LGBMCV():
    def __init__(self, cv=None, **kwargs):
        self.cv = cv
        self.lgbm_params = kwargs
        self.metric = kwargs['metric']

    def fit(self, X, y=None, **kwargs):
        self.models_ = []
        feature_names = X.columns if isinstance(X, pd.DataFrame) else list(range(X.shape[1]))
        self.feature_importances_ = pd.DataFrame(index=feature_names)
        self.evals_results_ = {}
        self.model_scores_ = []
        self.model_best_iterations_ = []

        for i, (fit_idx, val_idx) in enumerate(self.cv):

            # Split the dataset according to the fold indexes
            if isinstance(X, pd.DataFrame):
                X_fit = X.iloc[fit_idx]
                X_val = X.iloc[val_idx]
            else:
                X_fit = X[fit_idx]
                X_val = X[val_idx]

            if isinstance(y, pd.Series):
                y_fit = y.iloc[fit_idx]
                y_val = y.iloc[val_idx]
            else:
                y_fit = y[fit_idx]
                y_val = y[val_idx]

            # https://lightgbm.readthedocs.io/en/latest/Python-API.html#lightgbm.Dataset
            
            fit_set = lgbm.Dataset(X_fit, y_fit)
            val_set = lgbm.Dataset(X_val, y_val)
                
            #print(f'fit set {len(X_fit)} and val_set {len(X_val)}')
            # https://lightgbm.readthedocs.io/en/latest/Python-API.html#lightgbm.train
            self.evals_results_[i] = {}
            model = lgbm.train(
                params=self.lgbm_params,
                train_set=fit_set,
                valid_sets=(fit_set, val_set),
                valid_names=('train', 'eval'),
                evals_result=self.evals_results_[i],
                **kwargs
            )
            self.model_scores_.append(model.best_score['eval'][str(self.metric)])
            # Store the feature importances
            self.feature_importances_['gain_{}'.format(i)] = model.feature_importance('gain')
            self.feature_importances_['split_{}'.format(i)] = model.feature_importance('split')
            self.model_best_iterations_.append(model.best_iteration)
            # Store the model
            self.models_.append(model)

        return self

    def predict(self, X):

        utils.validation.check_is_fitted(self, ['models_'])

        y = np.zeros(len(X))

        for model in self.models_:
            y += model.predict(X, num_iteration=model.best_iteration)
        
        return y / len(self.models_)
    
from fastai import *
from fastai.tabular import *
from fastai.callbacks import SaveModelCallback

class FastAICV():
    
    def __init__(self, folds, cat_names, cont_names, procs, metric, metric_mode, bs, obj='binary'):
        self.folds = folds
        self.cat_names = cat_names
        self.cont_names = cont_names
        self.procs = procs
        self.metric = metric
        self.obj = obj
        self.metric_mode = metric_mode
        self.bs = bs
        
    def fit_predict(self, X, X_test, epochs, lr, wd=None, y=None, **kwargs):
        self.epochs = epochs
        self.lr = lr
        self.wd = wd
        self.models_ = []
        self.model_scores_ = []
        self.oof_preds_ = []
        self.seed = 42
        try:
            self.sub_preds = np.zeros(len(X_test))
        except:
            self.sub_preds = np.zeros(X_test.shape[0])
        for i, (fit_idx, val_idx) in enumerate(self.folds):
            if self.obj == 'binary':
                data = (
                    TabularList.from_df( 
                        X, path='.', cat_names=self.cat_names, \
                        cont_names=self.cont_names, procs=self.procs     
                            ).split_by_idx(val_idx
                        ).label_from_df(cols=y
                       ).add_test(TabularList.from_df(X_test, path=".", cat_names=self.cat_names, \
                        cont_names=self.cont_names, procs=self.procs)   
                     ).databunch(bs=self.bs)
                )
                
                learn = tabular_learner(data, metrics=self.metric, **kwargs)
                
                learn.fit_one_cycle(self.epochs, self.lr, wd=self.wd,\
                   callbacks=[SaveModelCallback(learn, every='improvement', mode=self.metric_mode, \
                                monitor=self.metric.__name__, name=f'fold{i}_{self.seed}')]
                                   )
#                 pred, _ = learn.get_preds()
#                 pred = pred[:,1]
                self.models_.append(learn)
                self.model_scores_.append(learn.validate(metrics=[self.metric])[1])
                pred_test, _ = learn.get_preds(DatasetType.Test)
                self.sub_preds += np.array(pred_test[:,1])
            else:
                data = (
                    TabularList.from_df( 
                        X, path='.', cat_names=self.cat_names, \
                        cont_names=self.cont_names, procs=self.procs     
                            ).split_by_idx(val_idx
                        ).label_from_df(cols=y, label_cls=FloatList, log=True
                       ).add_test(TabularList.from_df(X_test, path=".", cat_names=self.cat_names, \
                        cont_names=self.cont_names, procs=self.procs)   
                     ).databunch(bs=self.bs)
                )
                
                learn = tabular_learner(data, metrics=self.metric, **kwargs)
                
                learn.fit_one_cycle(self.epochs, self.lr, wd=self.wd,\
                   callbacks=[SaveModelCallback(learn, every='improvement', mode=self.metric_mode, \
                                monitor=self.metric.__name__, name=f'fold_{i}_{self.seed}')]
                                   )
#                 pred, _ = learn.get_preds()
#                 pred = pred[:,1]
                self.models_.append(learn)
                self.model_scores_.append(learn.validate(metrics=[self.metric])[1])
                pred_test, _ = learn.get_preds(DatasetType.Test)
                self.sub_preds += np.array(pred_test.reshape(-1))
        return self
    def predict(self):
        
        return self.sub_preds / len(self.models_)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

class RandomForestCV:
    def __init__(self, cv=None, obj="binary", cats=None, nums=None, **kwargs):
        self.cv = cv
        self.rf_params = kwargs
        self.nums = nums
        self.cats = cats
        self.obj = obj
        # self.metric = str(kwargs["metric"])

    def fit(self, X, y=None, **kwargs):

        self.models_ = []
        self.feature_importances_ = pd.DataFrame()
        self.model_scores_ = []
        self.model_best_iterations_ = []
        self.model_classes_ = []

        for i, (fit_idx, val_idx) in enumerate(self.cv):
            # Split the dataset according to the fold indexes
            if isinstance(X, pd.DataFrame):
                X_fit = X.iloc[fit_idx]
                X_val = X.iloc[val_idx]
            else:
                X_fit = X[fit_idx]
                X_val = X[val_idx]

            if isinstance(y, pd.Series):
                y_fit = y.iloc[fit_idx]
                y_val = y.iloc[val_idx]
            else:
                y_fit = y[fit_idx]
                y_val = y[val_idx]
            
            if self.obj in ["binary", "multiclass"]:
                model = RandomForestClassifier(**self.rf_params)
            else:
                raise Exception(f"{self.obj} not supported.")

            model.fit(X=X_fit, y=y_fit, **kwargs)

            # Store the feature importances
            if i == 0:
                self.feature_importances_["feature_names"] = X.columns
            self.feature_importances_["importance_{}".format(i)] = model.feature_importances_

            # Store model score
            self.model_scores_.append(f1_score(y_val, model.predict(X_val), average='macro'))

            # Store the model
            self.models_.append(model)

            # Store the classes
            self.model_classes_.append(model.classes_)

            del X_fit, y_fit, X_val, y_val, model
            gc.collect()

        return self

    def predict(self, X):

        utils.validation.check_is_fitted(self, ["models_"])
        # if pandas
        # try:
        #     y = np.zeros(len(X))
        # except:
        #     y = np.zeros(X.shape[0])
        
        all_preds = []
        
        for model in self.models_:
            if self.obj in ["binary", "multiclass"]:
                model_preds = model.predict_proba(X)
                all_preds.append(model_preds)
        
        all_preds = np.array(all_preds).mean(axis=0).argmax(axis=1)

        return all_preds