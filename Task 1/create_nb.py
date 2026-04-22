import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

title_cell = nbf.v4.new_markdown_cell("# Credit Card Fraud Detection EDA & Modeling\nThis notebook analyzes the `fraudTrain.csv` dataset, handles highly imbalanced data, and trains multiple classification models to optimize for high recall.")

code_setup = nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

plt.style.use('ggplot')
""")

md_load = nbf.v4.new_markdown_cell("## 1. Data Loading and Exploration")

code_load = nbf.v4.new_code_cell("""# Using 10% sample of training data within Notebook for faster EDA
df_train_full = pd.read_csv('../data/fraudTrain.csv')
df = df_train_full.sample(frac=0.1, random_state=42).copy()
df.head()
""")

md_eda = nbf.v4.new_markdown_cell("## 2. Exploratory Data Analysis\nLet's check the class distributions and basic feature patterns.")

code_eda1 = nbf.v4.new_code_cell("""# Class distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='is_fraud')
plt.title('Fraud Class Distribution')
plt.show()

print(f"Fraud Ratio: {df['is_fraud'].mean():.4f}")
""")

code_eda2 = nbf.v4.new_code_cell("""# Transaction amount by class
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x='is_fraud', y='amt')
plt.title('Transaction Amount by Fraud Status')
plt.yscale('log')
plt.show()
""")

code_eda3 = nbf.v4.new_code_cell("""# Fraud by Category
plt.figure(figsize=(12, 6))
sns.countplot(data=df[df['is_fraud']==1], y='category', order=df[df['is_fraud']==1]['category'].value_counts().index)
plt.title('Fraudulent Transactions by Category')
plt.show()
""")

md_preprocess = nbf.v4.new_markdown_cell("## 3. Preprocessing & SMOTE\nUsing the `data_utils.py` logic to standard scale numerical features and balance classes via SMOTE.")

code_preprocess = nbf.v4.new_code_cell("""import sys
sys.path.append('../src')
from data_utils import get_preprocessor

from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, roc_auc_score, recall_score, confusion_matrix

df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['trans_hour'] = df['trans_date_trans_time'].dt.hour
df['trans_dayofweek'] = df['trans_date_trans_time'].dt.dayofweek
df['trans_month'] = df['trans_date_trans_time'].dt.month

df['dob'] = pd.to_datetime(df['dob'])
df['age'] = (df['trans_date_trans_time'] - df['dob']).dt.days // 365

features_to_keep = ['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long', 
                    'trans_hour', 'trans_dayofweek', 'trans_month', 'age', 'category', 'gender', 'is_fraud']

df = df[features_to_keep].dropna()

X = df.drop(columns=['is_fraud'])
y = df['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Training on:", X_train.shape)
""")

md_model = nbf.v4.new_markdown_cell("## 4. Modeling & Evaluation\nTraining a base logistic regression model prioritizing recall.")

code_model = nbf.v4.new_code_cell("""preprocessor, _, _ = get_preprocessor()

# We only SMOTE inside the pipeline to prevent data leakage during CV
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42, sampling_strategy=0.1)),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print(classification_report(y_test, y_pred))

print("ROC-AUC:", roc_auc_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
""")

md_tuning = nbf.v4.new_markdown_cell("## 5. Hyperparameter Tuning\nRefer to `src/train.py` for full tuning setup using `RandomizedSearchCV` on Random Forests and Decision Trees. For real-world deployments on 350MB+ datasets, moving processing to dedicated scripts rather than Jupyter is standard practice.")

nb['cells'] = [title_cell, code_setup, md_load, code_load, md_eda, code_eda1, code_eda2, code_eda3, md_preprocess, code_preprocess, md_model, code_model, md_tuning]

os.makedirs('notebooks', exist_ok=True)
nbf.write(nb, 'notebooks/EDA_and_Modeling.ipynb')
print("Notebook created successfully!")
