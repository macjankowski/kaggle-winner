from sklearn.base import TransformerMixin


class TabularCustomTransformer(TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        X['i_02_21'] = (X["f_21"] + X["f_02"] > 5.2).astype(int) - (X["f_21"] + X["f_02"] < -5.3).astype(int)
        X['i_05_22'] = (X["f_22"] + X["f_05"] > 5.1).astype(int) - (X["f_22"] + X["f_05"] < -5.4).astype(int)
        i_00_01_26 = X["f_00"] + X["f_01"] + X["f_26"]
        X['i_00_01_26'] = (i_00_01_26 > 5.0).astype(int) - (i_00_01_26 < -5.0).astype(int)

        return X

    def get_params(self, deep=None):
        return {}

    def get_feature_names(self):
        return self.feature_names