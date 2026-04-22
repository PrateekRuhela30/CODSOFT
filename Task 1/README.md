# Credit Card Fraud Detection

This project demonstrates an end-to-end Machine Learning pipeline for detecting credit card fraud using Python, `scikit-learn`, `imbalanced-learn`, and `Streamlit`.

## Setup

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Preparation:**
   The training and testing datasets should be located in `data/fraudTrain.csv` and `data/fraudTest.csv` respectively.

3. **Run the application:**
   You can run the web app using:
   ```bash
   streamlit run app.py
   ```

4. **Run Training:**
   To train the model manually:
   ```bash
   python src/train.py
   ```

## Folder Structure
- `data/`: Contains datasets
- `notebooks/`: Jupyter notebooks for EDA and modeling iterations.
- `src/`: Core Python modules for data loading, preprocessing, and training.
- `models/`: Saved models (.pkl or .joblib)
- `app.py`: Streamlit application.
