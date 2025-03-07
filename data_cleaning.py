import json
import os
import numpy as np
import pandas as pd
# from sentence_transformers import SentenceTransformer, util

import seaborn as sns
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt


def calculate_readme_size(readme):
    if isinstance(readme, str):
        return len(readme)
    else:
        return 0
    

def calculate_commits_freq(commits):
    if isinstance(commits, list):
        return len(commits)
    else:
        return 0
    

def calculate_releases_freq(releases):
    if isinstance(releases, list):
        return len(releases)
    else:
        return 0


def calculate_lines_of_codes(languages):
    if isinstance(languages, dict):
        return sum(languages.values())
    return 0


def calculate_popularity_score_1(stars, forks, pull_requests):
    return stars + forks + pull_requests * pull_requests


def calculate_popularity_score_2(watchers, pull_requests):
    return watchers + pull_requests * pull_requests


def calculate_popularity_score_3(stars, forks):
    return stars + forks





def detect_and_remove_outliers(df, feature_columns, ratio=1.5):
    df = df.copy()  # Avoid modifying the original DataFrame
    
    for col in feature_columns:
        print(col)
        # Drop NaN values for this column
        df = df.dropna(subset=[col])
        
        # Calculate IQR bounds
        Q1 = np.percentile(df[col], 25)  # First quartile (25%)
        Q3 = np.percentile(df[col], 75)  # Third quartile (75%)
        IQR = Q3 - Q1  # Interquartile range
        lower_bound = Q1 - ratio * IQR  # Lower bound
        upper_bound = Q3 + ratio * IQR  # Upper bound
        
        # Create a flag column indicating whether a sample is an outlier
        df[f"{col}_outlier_flag"] = ((df[col] >= lower_bound) & (df[col] <= upper_bound)).astype(int)
        
        # Plot distributions before and after cleaning
        # plot_distribution(df[col], f"Original {col} Distribution", f"original_{col}_distribution.png")
        # plot_distribution(df[df[f"{col}_outlier_flag"] == 1][col], 
        #                        f"Cleaned {col} Distribution", 
        #                        f"cleaned_{col}_distribution.png")
    
    # Remove rows where at least one outlier flag is 0 (outlier detected in any feature)
    outlier_flags = [f"{col}_outlier_flag" for col in feature_columns]
    df_cleaned = df[df[outlier_flags].all(axis=1)].copy()

    print(f"Original data size: {len(df)}")
    print(f"Cleaned data size: {len(df_cleaned)}")

    return df_cleaned, df


def plot_distribution(data, title, filename):
    """
    Plots the KDE distribution of a feature and saves it as a PNG file.
    
    Args:
    - data (pd.Series): The feature data.
    - title (str): Title of the plot.
    - filename (str): File path to save the image.
    """
    kde = gaussian_kde(data)
    x_values = np.linspace(data.min(), data.max(), 200)
    y_values = kde(x_values) * len(data)  # Scale KDE to count

    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values, label=title, color="blue")
    plt.fill_between(x_values, y_values, alpha=0.3, color="blue")

    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    
    # Save the plot
    plt.savefig(filename, dpi=300)
    plt.show()




if __name__ == "__main__":
    file_paths = []

    for f_name in os.listdir('data'):
        if f_name.endswith('json'):
            file_paths.append("data/" + f_name)

    print(f"Total files detected: {len(file_paths)}\n")


    error_count = 0
    rows = []

    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                pull_requests = data.get("pull_requests", 0)  # 提取 pull_requests

                for url, details in data.items():
                    if url != "pull_requests":  # 排除 pull_requests 键
                        row = {"url": url, "name": url.split('/')[-1], "owner": url.split('/')[-2]}
                        row.update(details)
                        row["pull_requests"] = pull_requests  # 添加 pull_requests 到每行数据中
                        rows.append(row)
        except Exception as e:
            error_count += 1

    df = pd.DataFrame(rows)
    print(f"Total errors: {error_count}")
    print(f"Raw data size {len(df)}\n")


    df["readme_size"] = df["readme"].apply(calculate_readme_size)
    df["commits_freq"] = df["commits"].apply(calculate_commits_freq)
    df["releases_freq"] = df["releases"].apply(calculate_releases_freq)
    df["lines_of_codes"] = df["languages"].apply(calculate_lines_of_codes)
    df["popularity_score_1"] = df.apply(lambda row: calculate_popularity_score_1(row["stars"], row["forks"], row["pull_requests"]), axis=1)
    df["popularity_score_2"] = df.apply(lambda row: calculate_popularity_score_2(row["watchers"], row["pull_requests"]), axis=1)
    df["popularity_score_3"] = df.apply(lambda row: calculate_popularity_score_3(row["stars"], row["forks"]), axis=1)


    languages = []
    for repo_languages in df["languages"]:
        if isinstance(repo_languages, dict):  # 确保 repo_languages 是字典类型
            for repo_language in repo_languages:
                if repo_language not in languages:
                    languages.append(repo_language)
    print(f"There are total {len(languages)} languages \n")

    df.to_csv('data_original.csv', index=False)



    df_original = pd.read_csv("data_original.csv")

    df_original.drop(columns=['watchers'], inplace=True)
    df_original.rename(columns={'subscribers': 'watchers'}, inplace=True)

    df_original.to_csv('data_original_v2.csv', index=False)

    df_cleaned, df = detect_and_remove_outliers(df_original, feature_columns=["forks", "watchers", "stars",  "releases_freq",  "pull_requests",  "readme_size", "lines_of_codes"], ratio=1.5)
    df_cleaned.to_csv('data_cleaned.csv', index=False)


    df_cleaned = pd.read_csv("data_cleaned.csv")


    # List of feature columns
    feature_columns = ["forks", "watchers", "stars", "releases_freq", "pull_requests", "readme_size", "lines_of_codes"]

    # Compute statistics for original and cleaned dataframes
    stats_original = df_original[feature_columns].describe().loc[["min", "mean", "max"]]
    stats_cleaned = df_cleaned[feature_columns].describe().loc[["min", "mean", "max"]]

    # Combine results for comparison
    print("Original size: {}".format(len(df_original)))
    print(stats_original)
    print("\n")
    print("Cleaned size: {}".format(len(df_cleaned)))
    print(stats_cleaned)
    print("\n")
