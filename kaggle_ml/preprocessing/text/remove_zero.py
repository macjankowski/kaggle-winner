from sklearn.base import TransformerMixin


class RemoveZeroTransformer(TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for char in letters:
            if X[char].sum() == 0:
                X = X.drop([char], axis=1)

        return X