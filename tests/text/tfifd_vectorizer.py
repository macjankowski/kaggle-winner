

import unittest

import pandas as pd
from pandas._testing import assert_frame_equal
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

from kaggle_ml.preprocessing.text.tab_char_enc import AddPositionTransformer
from kaggle_ml.preprocessing.text.count_vectorizer import NGramsCountTransformer
from kaggle_ml.preprocessing.text.letters_count import LettersCountTransformer
from kaggle_ml.preprocessing.text.remove_zero import RemoveZeroTransformer
from kaggle_ml.preprocessing.text.tfidf_vectorizer import TfidfVectorizerTransformer


class TestTfIdfVectorizer(unittest.TestCase):


    def test_count_vectorizer(self):

        self.common_test_ngram(False)

    def test_count_vectorizer_drop(self):

        self.common_test_ngram(True)

    def common_test_ngram(self, drop_feature):
        ngam_transformer = TfidfVectorizerTransformer("f_27", drop_feature=drop_feature)
        d = {
            'col1': [2, 5],
            'f_27': pd.Series(["ABACABD", "ACDFBAA"], index=[2, 3])
        }
        X = pd.DataFrame(data=d)
        res_ngram = ngam_transformer.fit_transform(X)
        pipe = Pipeline(
            steps=[
                ("letters_count", LettersCountTransformer("f_27", drop_feature=drop_feature)),
                ("remove_zero", RemoveZeroTransformer()),
            ]
        )
        d = {
            'col1': [2, 5],
            'f_27': pd.Series(["ABACABD", "ACDFBAA"], index=[2, 3])
        }
        X = pd.DataFrame(data=d)
        res = pipe.fit_transform(X)
        assert_frame_equal(res_ngram, res)


    def test_count_vectorizer_bigrams(self):
        ngam_transformer = TfidfVectorizerTransformer("f_27", ngram_range=(1,2))
        d = {
            'col1': [2, 5],
            'f_27': pd.Series(["ABACABD", "ACDFBAA"], index=[2, 3])
        }
        X = pd.DataFrame(data=d)
        res_ngram = ngam_transformer.fit_transform(X)

    def test_count_vectorizer_binary(self):
        ngam_transformer = TfidfVectorizerTransformer("f_27", binary=True)
        d = {
            'col1': [2, 5],
            'f_27': pd.Series(["ABACABD", "ACDFBAA"], index=[2, 3])
        }
        X = pd.DataFrame(data=d)
        res_ngram = ngam_transformer.fit_transform(X)

        for col in ["A", "B", "C", "D", "F"]:
            assert res_ngram[col].max() == 1