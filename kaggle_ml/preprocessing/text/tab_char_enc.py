from sklearn.base import TransformerMixin
import numpy as np


class TabularCharEncoder(TransformerMixin):

    def __init__(self):
        def ch_to_int(ch):
            return ord(ch) - 65

        self.ch_to_int_v = np.vectorize(ch_to_int)
        self.feature_names = []

    def fit(self, X, y=None):
        length = len(X.iloc[0])
        self.feature_names = [f"pos_{i}" for i in range(length)]
        return self

    def transform(self, X, y=None):
        X = [list(s) for s in X]
        X_np = self.ch_to_int_v(X)
        return X_np

    def get_params(self, deep=None):
        return {}

    def get_feature_names(self):
        return self.feature_names
