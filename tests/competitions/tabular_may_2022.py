import pandas as pd
import xgboost as xgb

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from kaggle_ml.preprocessing.common.utils import split_to_features_and_target
from kaggle_ml.preprocessing.tabular.tabular_transformer import TabularCustomTransformer
from kaggle_ml.preprocessing.text.tab_char_enc import TabularCharEncoder


def create_pipeline(df):
    ct = ColumnTransformer(
        [
            ('categories', CountVectorizer(analyzer="char"), 'f_27'),
            ("add_position", TabularCharEncoder(), 'f_27')
        ],
        remainder='passthrough')

    pipe = Pipeline(
        steps=[
            ("tabular", TabularCustomTransformer()),
            ("column_transformer", ct),
            ("scaler", StandardScaler()),
        ]
    )

    return pipe.fit(df)


if __name__ == "__main__":
    SEED = 123

    train_df = pd.read_csv("../data/train.csv")

    data, label = split_to_features_and_target(train_df, target_col_name="target")
    data = data.drop(columns=["id"])

    data_train, data_eval, labels_train, labels_eval = train_test_split(data, label, test_size=0.10, random_state=42)

    pipe = create_pipeline(data_train)

    X_train = pipe.transform(data_train)
    X_test = pipe.transform(data_eval)

    dtrain = xgb.DMatrix(X_train, label=labels_train)
    deval = xgb.DMatrix(X_test, label=labels_eval)

    param = {
        'max_depth': 14,
        'eta': 0.25,
        'gamma': 0.06915198287661628,
        'lambda': 11.9924333574069,
        'max_bin': 768,
        'objective': 'binary:logistic'
    }

    # param['tree_method'] = 'gpu_hist'
    # param['gpu_id'] = 0
    param['eval_metric'] = 'auc'
    evallist = [(deval, 'deval'), (dtrain, 'train')]

    num_round = 3
    bst = xgb.train(param, dtrain, num_round, evallist)

    # test_df[["id", "target"]].to_csv("sample_submission_28.05.csv", index=False)
