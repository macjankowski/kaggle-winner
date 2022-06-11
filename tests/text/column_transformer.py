import unittest
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer

from kaggle_ml.preprocessing.text.tab_char_enc import TabularCharEncoder


class TestCountVectorizer(unittest.TestCase):

    def test_count_vectorizer(self):
        pipe = ColumnTransformer(
            [
                ('categories', CountVectorizer(analyzer="char"), 'f_27'),
                ("add_position", TabularCharEncoder(), 'f_27')
            ],
            remainder='passthrough', verbose_feature_names_out=False)

        d = {
            'col1': [2, 5],
            'f_27': pd.Series(["ABACABD", "ACDFBAA"], index=[2, 3])
        }
        X = pd.DataFrame(data=d)

        res = pipe.fit_transform(X)
        res_df = pd.DataFrame(res, columns=pipe.get_feature_names())
        print(res_df)
