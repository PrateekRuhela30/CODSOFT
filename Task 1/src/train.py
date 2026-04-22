import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score, recall_score, precision_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RandomizedSearchCV
import warnings
warnings.filterwarnings('ignore')

from data_utils import load_and_preprocess_data, get_preprocessor

def train_and_evaluate():
    print("Loading data...")
    # Using 10% of training data for speed during testing
    train_path = os.path.join('..', 'data', 'fraudTrain.csv')
    test_path = os.path.join('..', 'data', 'fraudTest.csv')
    
    # Check if run from src or root
    if not os.path.exists(train_path):
        train_path = os.path.join('data', 'fraudTrain.csv')
        test_path = os.path.join('data', 'fraudTest.csv')
        
    df_train, df_test = load_and_preprocess_data(train_path, test_path, sample_fraction=0.1)
    
    print(f"Train set size: {df_train.shape}")
    print(f"Test set size: {df_test.shape}")
    print(f"Train fraud ratio: {df_train['is_fraud'].mean():.4f}")
    
    X_train = df_train.drop(columns=['is_fraud'])
    y_train = df_train['is_fraud']
    
    X_test = df_test.drop(columns=['is_fraud'])
    y_test = df_test['is_fraud']

    preprocessor, _, _ = get_preprocessor()
    
    # Models to compare
    models = {
        'Logistic Regression': LogisticRegression(max_iter=500, random_state=42, class_weight='balanced'),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=30, max_depth=8, random_state=42, n_jobs=-1, class_weight='balanced')
    }
    
    print("\n--- Training Basic Models ---")
    best_model = None
    best_score = 0
    best_name = ""
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        # Optional: Use SMOTE. 
        # Using class_weight='balanced' in the model might be enough and faster, 
        # but let's include SMOTE as per requirements
        pipeline = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42, sampling_strategy=0.1)), # Upsample minority to 10% of majority to speed up SMOTE
            ('classifier', model)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        rec = recall_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_pred)
        
        print(f"{name} Results:")
        print(f"Recall: {rec:.4f} | Precision: {prec:.4f} | F1: {f1:.4f} | ROC-AUC: {roc:.4f}")
        
        # We prioritize balanced performance favoring recall
        if roc > best_score:
            best_score = roc
            best_model = pipeline
            best_name = name
            
    print(f"\nBest Model selected: {best_name} (ROC-AUC: {best_score:.4f})")
    
    # Save best model
    models_dir = os.path.join(os.path.dirname(train_path), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'best_model.joblib')
    joblib.dump(best_model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_and_evaluate()
