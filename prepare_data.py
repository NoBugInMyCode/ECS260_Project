import pandas as pd
from sklearn.model_selection import train_test_split


def convert_columns_to_float32(df, columns):
    for col in columns:
        df[col] = df[col].astype('float32')
    return df


def prepare_data():
    df_cleaned = pd.read_csv("data_cleaned.csv")

    # Specify the input and target columns
    input_columns = ["forks", "watchers", "releases_freq", "pull_requests", "readme_size", "lines_of_codes"]
    target_column = "stars"

    # Convert int64 to float32
    df_cleaned = convert_columns_to_float32(df_cleaned, input_columns+[target_column])


    # Extract the input and target data
    X = df_cleaned[input_columns]
    y = df_cleaned[target_column]

    dtypes = list(zip(X.dtypes.index, map(str, X.dtypes)))
    for k, dtype in dtypes:
        if dtype == "float32":
            X.loc[:, k] -= X[k].mean()
            X.loc[:, k] /= X[k].std()

    # Split the data into training and testing sets with a fixed random state
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("X_train: {}".format(X_train.shape))
    print("y_train: {}".format(y_train.shape))
    print("X_test: {}".format(X_test.shape))
    print("y_test: {}".format(y_test.shape))

    return X_train, X_test, y_train, y_test, dtypes