import os
import joblib
import pandas as pd

class FraudPredictor:
    def __init__(self, model_path=None):
        if model_path is None:
            # Try to resolve path
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, 'models', 'best_model.joblib')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
            
        self.model = joblib.load(model_path)
        
    def predict(self, input_data: dict) -> dict:
        """
        Takes raw input dict, converts to DataFrame, extracts features in the expected way,
        and returns prediction and probability.
        """
        df = pd.DataFrame([input_data])
        
        # Similar logic to data_utils process_df for serving
        # Note: In a real prod environment, serving & training prep logic should be tightly unified.
        
        # Datetime processing
        if 'trans_date_trans_time' in df.columns:
            dt = pd.to_datetime(df['trans_date_trans_time'])
            df['trans_hour'] = dt.dt.hour
            df['trans_dayofweek'] = dt.dt.dayofweek
            df['trans_month'] = dt.dt.month
            
        if 'dob' in df.columns and 'trans_date_trans_time' in df.columns:
            dob = pd.to_datetime(df['dob'])
            df['age'] = (dt - dob).dt.days // 365
            
        prediction = self.model.predict(df)[0]
        probabilities = self.model.predict_proba(df)[0]
        
        return {
            'is_fraud': bool(prediction),
            'fraud_probability': float(probabilities[1]),
            'legitimate_probability': float(probabilities[0])
        }

if __name__ == "__main__":
    predictor = FraudPredictor()
    
    # Sample input (dummy data based on required features by pipeline)
    sample = {
        'amt': 105.2,
        'lat': 42.18,
        'long': -112.26,
        'city_pop': 4154,
        'merch_lat': 43.15,
        'merch_long': -112.15,
        'trans_date_trans_time': '2023-10-15 14:30:00',
        'dob': '1985-05-20',
        'category': 'grocery_pos',
        'gender': 'F'
    }
    
    result = predictor.predict(sample)
    print("Prediction Result:")
    print(result)
