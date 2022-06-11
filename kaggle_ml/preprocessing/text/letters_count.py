from sklearn.base import TransformerMixin


class LettersCountTransformer(TransformerMixin):

    def __init__(self, feature_name, drop_feature=False):
        self.feature_name = feature_name
        self.drop_feature = drop_feature

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for char in letters:
            X[char] = X[self.feature_name].str.count(char)

        if self.drop_feature:
            X = X.drop([self.feature_name], axis=1)

        return X
