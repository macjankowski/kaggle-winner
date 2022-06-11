import unittest

import pandas as pd
import numpy as np
from numpy.ma.testutils import assert_array_equal
from pandas._testing import assert_frame_equal
from sklearn.pipeline import Pipeline

from kaggle_ml.preprocessing.common.utils import split_to_features_and_target
from kaggle_ml.preprocessing.text.tab_char_enc import AddPositionTransformer
from kaggle_ml.preprocessing.text.letters_count import LettersCountTransformer
from kaggle_ml.preprocessing.text.remove_zero import RemoveZeroTransformer


class TestCommon(unittest.TestCase):

    def test_split(self):
        d = {
            'target': [2, 5],
            'f_27': pd.Series(["ABACABD", "ACDFBAA"], index=[2, 3])
        }
        df = pd.DataFrame(data=d)

        X, y = split_to_features_and_target(df, target_col_name="target")

        assert_frame_equal(X, pd.DataFrame(data={
            'f_27': pd.Series(["ABACABD", "ACDFBAA"], index=[2, 3])
        }))

        assert_array_equal(y, np.array([2, 5]))

