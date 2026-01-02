# File: src/car_price_train.py

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def train_models():
    print(">>> Starting model training pipeline...")

    # Define paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(BASE_DIR, "data", "clean_car_data.csv")
    output_dir = os.path.join(BASE_DIR, "outputs")
    
    if not os.path.exists(data_path):
        print("Error: Clean data not found. Run 'car_price_prepare.py' first.")
        return

    # 1. Load Data
    df = pd.read_csv(data_path)
    
    # 2. Split Features and Target
    X = df.drop(columns=['fiyat', 'fiyat_log'])
    y = df['fiyat_log'] # Target is Log-Transformed Price

    # 3. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Ensure boolean columns are float for XGBoost compatibility
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)
    
    print(f"Train Set: {X_train.shape} | Test Set: {X_test.shape}")

    # 4. Define Models
    models = {
        "OLS (Linear Regression)": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
    }

    results = []
    best_model = None
    best_r2 = -np.inf
    
    print("\nTraining models... (This might take a moment)")

    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        
        # Predict (Log Scale)
        y_pred_log = model.predict(X_test)
        
        # Inverse Transform (Log -> Real TL Price)
        # This is crucial for interpretable metrics
        y_pred_real = np.expm1(y_pred_log)
        y_test_real = np.expm1(y_test)
        
        # Calculate Metrics on REAL prices
        rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
        mae = mean_absolute_error(y_test_real, y_pred_real)
        r2 = r2_score(y_test_real, y_pred_real)
        
        results.append({
            "Model": name,
            "RMSE (TL)": rmse,
            "MAE (TL)": mae,
            "R2 Score": r2
        })
        
        print(f"✅ {name} completed. R2: {r2:.4f}")

        # Keep track of the best model to save later
        if r2 > best_r2:
            best_r2 = r2
            best_model = model

    # 5. Show Comparison Table
    results_df = pd.DataFrame(results).sort_values(by="R2 Score", ascending=False)
    print("\n🏆 MODEL COMPARISON TABLE 🏆")
    print(results_df.to_string(index=False))

    # 6. Save Best Model
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(best_model, os.path.join(output_dir, "best_model.pkl"))
    joblib.dump(X_train.columns, os.path.join(output_dir, "model_columns.pkl"))
    
    print(f"\nBest model ({results_df.iloc[0]['Model']}) saved to: {output_dir}")

if __name__ == "__main__":
    train_models()