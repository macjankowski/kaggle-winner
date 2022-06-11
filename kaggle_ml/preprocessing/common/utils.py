

def split_to_features_and_target(df, target_col_name):

    target = df[[target_col_name]].values.reshape(-1)
    features_df = df.drop(columns=[target_col_name])

    return features_df, target
