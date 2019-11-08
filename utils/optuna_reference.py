trials=20
cv='group'
def objective(trial):
    param = {
      'num_leaves': trial.suggest_int('num_leaves', 500, 2000),
      'max_bin': trial.suggest_int('max_bin', 50, 255),
      'min_child_weight': trial.suggest_loguniform('min_child_weight', 0.001, 0.5),
      'feature_fraction': trial.suggest_loguniform('feature_fraction', 0.1, 1.0),
      'bagging_fraction': trial.suggest_loguniform('bagging_fraction', 0.1, 1.0),
      #'min_data_in_leaf': 105,
      'objective': 'binary',
      'max_depth': -1,
      'learning_rate': 0.03,
      "boosting_type": "gbdt",
      "bagging_seed": 11,
      "metric": 'auc',
      "verbosity": -1,
      'reg_alpha': trial.suggest_loguniform('reg_alpha', 0.001, 1.0),
      'reg_lambda': trial.suggest_loguniform('reg_lambda', 0.001, 1.0),
      'random_state': 21
    }
   # param.update({'min_data_in_leaf': int(train.shape[0] * 0.005)})
    model = LGBMCV(cv=cv ,**param)
    model = model.fit(train[feats], train['isFraud'], group=group,
             num_boost_round=10000, early_stopping_rounds=250, verbose_eval=False,
             categorical_feature=cats)
   # cb_best_iteration.extend(model.model_best_iterations_)
    return np.mean(model.model_scores_)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=trials)
cb_study = study.trials_dataframe()