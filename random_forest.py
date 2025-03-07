import pandas as pd
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer, util
import numpy as np
from sympy.physics.units import length
from tqdm.auto import tqdm
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from prepare_data import prepare_data


df = pd.read_csv('nonAI_repos.csv')
df = df.drop(columns=['watchers'])
df = df.rename(columns={'subscribers': 'watchers'})
features = ['forks', 'watchers', 'releases_count', 'pull_requests', 'readme_size', 'lines_of_codes']
X_train, X_test, y_train, y_test, _ = prepare_data(df)

rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"R² score: {r2}")

feature_importance = rf.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance in Predicting GitHub Stars Non AI Category (Random Forest)")
plt.gca().invert_yaxis()
plt.show()

print(feature_importance_df)