import pandas as pd
import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class DropUvaerosolLayerHeightTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, pattern='uvaerosollayerheight'):
        if isinstance(pattern, re.Pattern):
            self.pattern = pattern
        else:
            self.pattern = re.compile(pattern, re.IGNORECASE)

    def fit(self, X, y=None):
        # No fitting necessary for this transformer
        return self

    def transform(self, X):
        # Implement your transformation logic here
        # For example, dropping columns that match the pattern
        columns_to_drop = [col for col in X.columns if self.pattern.search(col)]
        return X.drop(columns=columns_to_drop)

class ConvertDateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, date_column='date', date_format='%Y-%m-%d'):
        self.date_column = date_column
        self.date_format = date_format
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        missing_before = X[self.date_column].isnull().sum()
        # print(f"Missing values in '{self.date_column}' before conversion: {missing_before}")
        
        X[self.date_column] = pd.to_datetime(X[self.date_column], format=self.date_format, errors='coerce')
        missing_after = X[self.date_column].isnull().sum()
        # if missing_after > 0:
        #     print(f"Missing or invalid dates after conversion: {missing_after}")
        # else:
        #     print(f"All dates in '{self.date_column}' successfully converted to datetime format.")
        
        return X

class SortByDateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, date_column='date'):
        self.date_column = date_column
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_sorted = X.sort_values(by=self.date_column).reset_index(drop=True)
        # print(f"DataFrame sorted by '{self.date_column}' in ascending order.\n")
        return X_sorted

class ExtractDateComponentsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, date_column='date'):
        self.date_column = date_column
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['day'] = X[self.date_column].dt.day
        X['day_of_week'] = X[self.date_column].dt.dayofweek
        # print("Extracted 'day' and 'day_of_week' from 'date' column.\n")
        return X

class DropDateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, pattern='date'):
        if isinstance(pattern, re.Pattern):
            self.pattern = pattern
        else:
            self.pattern = re.compile(pattern, re.IGNORECASE)
    
    def fit(self, X, y=None):
        # No fitting necessary
        return self
    
    def transform(self, X):
        # Example transformation: Drop columns matching the pattern
        columns_to_drop = [col for col in X.columns if self.pattern.search(col)]
        return X.drop(columns=columns_to_drop)