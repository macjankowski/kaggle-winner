import unittest
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer

from kaggle_ml.preprocessing.text.tab_char_enc import TabularCharEncoder
from kaggle_ml.preprocessing.text.string_to_columns import StringToColumnsTransformer


class TestLabelEncoder(unittest.TestCase):

    def test_label_encoder(self):
        pipe = ColumnTransformer(
            [
                ('str_to_col', StringToColumnsTransformer(), 'f_27'),
                # ("add_position", AddPositionTransformer2(), 'f_27')
                ("label_enc", TestLabelEncoder(), [['str_to_col__pos_0', 'str_to_col__pos_1', 'str_to_col__pos_2', 'str_to_col__pos_3', 'str_to_col__pos_4', 'str_to_col__pos_5', 'str_to_col__pos_6']])
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
