import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(train_path, test_path=None, sample_fraction=1.0):
    """
    Loads data and performs initial cleaning and feature engineering.
    sample_fraction allows using a smaller subset of training data for faster runs.
    """
    df_train = pd.read_csv(train_path)
    
    if test_path:
        df_test = pd.read_csv(test_path)
    else:
        df_test = None

    if sample_fraction < 1.0:
        df_train = df_train.sample(frac=sample_fraction, random_state=42)
    
    def process_df(df):
        # Convert date to datetime
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
        df['trans_hour'] = df['trans_date_trans_time'].dt.hour
        df['trans_dayofweek'] = df['trans_date_trans_time'].dt.dayofweek
        df['trans_month'] = df['trans_date_trans_time'].dt.month
        
        # Calculate age if we use dob
        df['dob'] = pd.to_datetime(df['dob'])
        df['age'] = (df['trans_date_trans_time'] - df['dob']).dt.days // 365
        
        # Drop redundant and high cardinality columns
        cols_to_drop = ['Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'merchant', 
                        'first', 'last', 'street', 'city', 'state', 'zip', 'job', 
                        'dob', 'trans_num', 'unix_time']
        
        df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
        return df

    df_train = process_df(df_train)
    if df_test is not None:
        df_test = process_df(df_test)
        
    return df_train, df_test

def get_preprocessor():
    """
    Return a scikit-learn ColumnTransformer for preprocessing.
    """
    numeric_features = ['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long', 
                        'trans_hour', 'trans_dayofweek', 'trans_month', 'age']
    categorical_features = ['category', 'gender']
    
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor, numeric_features, categorical_features
