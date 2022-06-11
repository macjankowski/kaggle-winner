from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd


class NGramsCountTransformer(TransformerMixin):

    def __init__(self, feature_name, ngram_range=(1, 1), binary=False, drop_feature=False):
        self.feature_name = feature_name
        self.drop_feature = drop_feature
        self.vectorizer = CountVectorizer(analyzer="char", ngram_range=ngram_range, binary=binary)

    def fit(self, X, y=None):
        corpus = X[self.feature_name].values
        self.vectorizer.fit(corpus)
        return self

    def transform(self, X, y=None):

        corpus = X[self.feature_name].values
        X_np = self.vectorizer.transform(corpus)

        new_columns = [col.upper() for col in self.vectorizer.get_feature_names()]
        X_counts = pd.DataFrame(data=X_np.toarray(), columns=new_columns, index=X.index)

        X_final = pd.concat([X, X_counts], axis=1)

        if self.drop_feature:
            X_final = X_final.drop([self.feature_name], axis=1)

        return X_final
