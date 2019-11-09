# apes_together_strong

MLHackFest 2019 Repo

## How to use
1. create your own `conda` or `virtual environment`
2. run `pip install -r requirements.txt` 
3. get wild.

## Structure
`test` folder contains all experiment notebooks from kaggle competitions with the pipeline created in utils/models.

`src` folder is reserved for competition purposes, need to copy tests/utils in the folder to use the modules.

`data` folder is used for downloading dataset in kaggle using `KAGGLE API`

src and test folder must have it's own `submission` folder when generating predictions in competition/challenges for easier tracking of the submission files.

## Notes:
* When using `CatBoostCV` specify in *__init__* the `obj`  i.e `regression` or `binary` to use the correct algorithm to use. `LGBMCV` works for both, no need to specify the `obj`

* When dealing with Regression Problem transform the target to `np.log` for easier training then transform again back to original state using `np.exp`. if `negative values` are encountered in the prediction values, just use `pd.Series.clip` function to clip the values to its min, and max.

* When dealing with Classification Problem with large dataset 500K ~ 1M+ Instances, consider to downsample the majority class for easier feedback loop iteration, `don't` use SMOTE or other stuff, that doesn't work!

* When everything doesn't work, use target encoding under `utils/cat_encoding.py` that will `automagically` make the model better, but ofcourse make sure you have solid CV and **DO NOT OVERFIT**

* RandomForest is the only model that `Rafael Trusts` in sklearn that can be used in competitions, unless you ensemble/stack predictions, use LogisticRegression.

* Submission files makes it easier for us to check if our `Cross Validation` correlates with the Public Leaderboard in Kaggle by formatting the name of the submission file using `{model_used}_{challenge}_(my_cv_score}.csv`



