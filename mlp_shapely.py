import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler

import tensorflow as tf
from keras.regularizers import l2
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

import shap


analysis_csv = "AI_repos.csv"
checkpoint_name = "checkpoint_AI_top3000.h5"


def calculate_correlation(analysis_csv):
    df_cleaned = pd.read_csv(analysis_csv)
    column_of_interests = ["releases_freq", "pull_requests", "readme_size", "lines_of_codes", "number_of_languages", "forks", "watchers", "stars"]

    # Compute the correlation matrix
    correlation_matrix = df_cleaned[column_of_interests].corr()

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")

    # Save the plot
    plt.title("Correlation Matrix")
    plt.savefig("correlation_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()


def convert_columns_to_float32(df, columns):
    for col in columns:
        df[col] = df[col].astype('float32')
    return df


def prepare_data(analysis_csv):
    df_cleaned = pd.read_csv(analysis_csv)

    # Specify the input and target columns
    input_columns = ["forks", "watchers", "releases_freq", "pull_requests", "readme_size", "lines_of_codes", "number_of_languages"]
    target_column = "stars"

    # Convert int64 to float32
    df_cleaned = convert_columns_to_float32(df_cleaned, input_columns+[target_column])
    df_cleaned = df_cleaned.sort_values(by="stars", ascending=False).head(3000)


    # Extract the input and target data
    X = df_cleaned[input_columns]
    y = df_cleaned[target_column]

    # Split the data into training and testing sets with a fixed random state
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize using only training set statistics
    dtypes = list(zip(X.dtypes.index, map(str, X.dtypes)))
    for k, dtype in dtypes:
        if dtype == "float32":
            X_train[k] = (X_train[k]+1).apply(np.log)
            X_test[k] = (X_test[k]+1).apply(np.log)

            x_mean = X_train[k].mean()
            x_std = X_train[k].std()

            # Apply normalization
            X_train.loc[:, k] = (X_train[k] - x_mean) / x_std
            X_test.loc[:, k] = (X_test[k] - x_mean) / x_std

    y_train = (y_train+1).apply(np.log)
    y_test = (y_test+1).apply(np.log)

    y_mean = y_train.mean()
    y_std = y_train.std()

    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    print("X_train: {}".format(X_train.shape))
    print("y_train: {}".format(y_train.shape))
    print("X_test: {}".format(X_test.shape))
    print("y_test: {}\n".format(y_test.shape))

    return X_train, X_test, y_train, y_test, dtypes, y_mean, y_std


def build_model(dtypes):
    # build model
    input_els = []
    encoded_els = []
    for k, dtype in dtypes:
        input_els.append(Input(shape=(1,)))
        # if dtype == "int8":
        #     e = Flatten()(Embedding(X_train[k].max() + 1, 1)(input_els[-1]))
        # else:
        e = input_els[-1]
        encoded_els.append(e)

    encoded_els = concatenate(encoded_els)

    # Add more layers with Batch Normalization
    layer1 = Dense(32, activation="relu")(encoded_els)
    layer2 = Dense(32, activation="relu")(layer1)
    layer3 = Dense(32, activation="relu")(layer2)
    out = Dense(1)(layer3)

    # train model
    regressor = Model(inputs=input_els, outputs=[out])
    optimizer = Adam(learning_rate=0.0001)
    regressor.compile(optimizer=optimizer, loss="mean_squared_error")

    return regressor


def run_fit(epochs, regressor, X_train, y_train, dtypes, checkpoint_name):
    # Define checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        checkpoint_name,            # Filepath to save the model
        save_best_only=True,              # Save only the best model based on val_loss
        monitor="loss",                   # Metric to monitor
        mode="min",                       # Save model with the minimum val_loss
        save_weights_only=False,          # Save the entire model (not just weights)
        verbose=1                         # Print messages when saving
    )

    regressor.fit(
        [X_train[k].values for k, t in dtypes],
        y_train,
        epochs=epochs,
        batch_size=128,
        shuffle=True,
        callbacks=[checkpoint_callback]
    )


def run_test(regressor, X_test, y_test, dtypes, y_mean, y_std):
    # Make predictions
    y_pred = regressor.predict([X_test[k].values for k, t in dtypes])

    y_pred = y_pred * y_std + y_mean
    y_test = y_test * y_std + y_mean

    y_pred = np.exp(y_pred) + 1
    y_test = np.exp(y_test) + 1

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Plot ground-truth vs predicted values
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.xlabel("Ground-Truth Values")
    plt.ylabel("Predicted Values")
    plt.title("Ground-Truth vs Predicted Values")
    plt.grid()
    plt.savefig("./prediction_result.png")

    # Print evaluation metrics
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")


def f(X):
    return regressor.predict([X[:, i] for i in range(X.shape[1])]).flatten()




if __name__ == "__main__":
    # calculate_correlation(analysis_csv)

    X_train, X_test, y_train, y_test, dtypes, y_mean, y_std = prepare_data(analysis_csv)
    regressor = build_model(dtypes)
    run_fit(150, regressor, X_train, y_train, dtypes, checkpoint_name)
    regressor = load_model(checkpoint_name)
    run_test(regressor, X_test, y_test, dtypes, y_mean, y_std)

    sampled_test = shap.sample(X_test, 500)
    X_train_summary = shap.kmeans(X_train, 500)
    explainer = shap.KernelExplainer(f, X_train_summary)
    shap_values = explainer.shap_values(sampled_test, nsamples=500)


    # Beeswarm plot
    shap.summary_plot(shap_values, sampled_test, plot_type="dot", show=False)
    plt.savefig("shap_beeswarm_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
