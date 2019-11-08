from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import clone
from sklearn.model_selection import check_cv

from category_encoders import CatBoostEncoder

import pandas as pd
import numpy as np


class TargetEncoderCV(BaseEstimator, TransformerMixin):

    def __init__(self, cv, **cbe_params):
        #self.encoder = encoder
        self.cv = cv
        self.cbe_params = cbe_params

    @property
    def _n_splits(self):
        return check_cv(self.cv).n_splits

    def fit_transform(self, X: pd.DataFrame, y) -> pd.DataFrame:
        self.cbe_ = []
        cv = check_cv(self.cv)

        cbe = CatBoostEncoder(
            cols=X.columns.tolist(),
            return_df=False,
            **self.cbe_params
        )

        X_transformed = np.zeros_like(X, dtype=np.float64)
        for train_idx, valid_idx in cv.split(X, y):
            self.cbe_.append(
                clone(cbe).fit(X.loc[train_idx], y[train_idx])
            )
            X_transformed[valid_idx] = self.cbe_[-1].transform(
                X.loc[valid_idx]
            )

        return pd.DataFrame(X_transformed, columns=X.columns)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_transformed = np.zeros_like(X, dtype=np.float64)
        for cbe in self.cbe_:
            X_transformed += cbe.transform(X) / self._n_splits
        return pd.DataFrame(X_transformed, columns=X.columns)
        
### usage
'''
te_cv = TargetEncoderCV(KFold(n_splits=3))
X_train_encoded = te_cv.fit_transform(X_train, y_train)
X_test_encoded = te_cv.transform(X_test)
'''

def frequency_encoding(feature):
    t = df[feature].value_counts().reset_index()
    t = t.reset_index()
    t.loc[t[feature] == 1, 'level_0'] = np.nan
    t.set_index('index', inplace=True)
    max_label = t['level_0'].max() + 1
    t.fillna(max_label, inplace=True)
    return t.to_dict()['level_0']

### usage
'''
    for feature in tqdm(cats):
        freq_enc_dict = frequency_encoding(feature)
        df[f"freq_{feature}"] = df[feature].map(lambda x: freq_enc_dict.get(x, np.nan))
        df[f"freq_{feature}"] = df[f"freq_{feature}"].astype(int) 
'''