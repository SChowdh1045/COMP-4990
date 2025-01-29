from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

# Custom transformer for binary encoding
class BinaryEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, binary_features):
        self.binary_features = binary_features
        self.binary_mappings = {}
    
    def fit(self, X, y=None):
        # Create mappings dynamically for each binary feature
        for feature in self.binary_features:
            unique_values = list(X[feature].unique())
            # Create mapping with first unique value as 0, second as 1
            self.binary_mappings[feature] = {unique_values[0]: 0, unique_values[1]: 1}
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        # Apply mappings
        for feature, mapping in self.binary_mappings.items():
            X_copy[feature] = X_copy[feature].map(mapping)
        return X_copy


# Create the pipeline
def create_model_pipeline(binary_features, multi_cat_features, numerical_features):
    # Create transformers
    binary_cat_transformer = BinaryEncoder(binary_features)
    multi_cat_transformer = OneHotEncoder(drop='first', sparse_output=False)
    numeric_transformer = MinMaxScaler()

    # Create column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('bin', binary_cat_transformer, binary_features),
            ('cat', multi_cat_transformer, multi_cat_features),
            ('num', numeric_transformer, numerical_features)
        ])

    # Create pipeline
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    return model_pipeline