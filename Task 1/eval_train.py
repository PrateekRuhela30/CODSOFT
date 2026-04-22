import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score, recall_score, precision_score
import sys
import os
sys.path.append('src')
from data_utils import load_and_preprocess_data

print("Loading data...")
df_train, _ = load_and_preprocess_data('data/fraudTrain.csv', sample_fraction=0.1)
X_train = df_train.drop(columns=['is_fraud'])
y_train = df_train['is_fraud']

print("Loading model...")
model = joblib.load('models/best_model.joblib')
y_train_pred = model.predict(X_train)

print(f"Train Recall: {recall_score(y_train, y_train_pred):.4f}")
print(f"Train Precision: {precision_score(y_train, y_train_pred):.4f}")
print(f"Train ROC-AUC: {roc_auc_score(y_train, y_train_pred):.4f}")
