import unittest

import pandas as pd
from sklearn.pipeline import Pipeline

from kaggle_ml.preprocessing.text.tab_char_enc import AddPositionTransformer
from kaggle_ml.preprocessing.text.letters_count import LettersCountTransformer
from kaggle_ml.preprocessing.text.remove_zero import RemoveZeroTransformer


class TestPipeline(unittest.TestCase):

    def test_pipe(self):

        pipe = Pipeline(
            steps=[
                ("letters_count", LettersCountTransformer("f_27")),
                ("remove_zero", RemoveZeroTransformer()),
                ("add_position", AddPositionTransformer("f_27", drop_feature=True))
            ]
        )

        d = {
            'col1': [2, 5],
            'f_27': pd.Series(["ABACABD", "ACDFBAA"], index=[2, 3])
        }
        X = pd.DataFrame(data=d)

        res = pipe.fit_transform(X)
        print(res)

        assert 1 == 2