
import scipy
import numpy as np
import pandas as pd

class DataFramePreprocessor():
    '''
        Creates an encoder/normalizer/preprocessor for each feature column
        Then fits the prepreocessor and returns a transformed matrix
    '''
    
    def __init__(self, preprocessor_map):
        self.preprocessor_map = preprocessor_map
    
    def fit_transform(self, df):
        return self._convert_df_to_ndarray(df, self.preprocessor_map, method='fit_transform')
    
    def transform(self, df):
        return self._convert_df_to_ndarray(df, self.preprocessor_map, method='transform')
        
    def _convert_df_to_ndarray(self, df, preprocessor_map, method='fit_transform'):    
        print(df.shape)
        data = []
        for column_name, preprocessor in preprocessor_map.items():

            if preprocessor is not None:
                if method == 'fit_transform':
                    transformed_data = preprocessor.fit_transform(df[[column_name]])
                else:
                    transformed_data = preprocessor.transform(df[[column_name]])


                if isinstance(transformed_data, scipy.sparse.csr.csr_matrix):
                    transformed_data = transformed_data.todense()
            else:
                # Just get the values
                transformed_data = df[[column_name]].values
            print(transformed_data.shape)
            data.append(transformed_data)
        return np.hstack(data)
    
# Define how each column will be preprocessed, if no preprocessing, just map the column to None
# preprocessor_map = {
#     'sex': OneHotEncoder(),
#     'embarked': OneHotEncoder(),
#     'pclass': OneHotEncoder(),
#     'age': MinMaxScaler((0,1)),
#     'sibsp': MinMaxScaler((0,1)),
#     'parch': MinMaxScaler((0,1)),
#     'fare': MinMaxScaler((0,1))    
# }
