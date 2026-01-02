# File: src/car_price_shap.py

import pandas as pd
import numpy as np
import os
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

def generate_plots():
    print(">>> Starting visualization pipeline...")

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(BASE_DIR, "data", "clean_car_data.csv")
    model_path = os.path.join(BASE_DIR, "outputs", "best_model.pkl")
    columns_path = os.path.join(BASE_DIR, "outputs", "model_columns.pkl")
    output_dir = os.path.join(BASE_DIR, "outputs")

    if not os.path.exists(model_path):
        print("Error: Model not found. Run 'car_price_train.py' first.")
        return

    # Load resources
    df = pd.read_csv(data_path)
    model = joblib.load(model_path)
    model_cols = joblib.load(columns_path)

    # Prepare X (Features only)
    X = df.drop(columns=['fiyat', 'fiyat_log'], errors='ignore')
    X = X[model_cols] # Ensure strict column order
    X = X.astype(float) # Ensure float for XGBoost compatibility

    # Sample for SHAP (100 samples as in your original code)
    X_sample = X.sample(n=100, random_state=42)
    
    print("1. Generating SHAP Plots...")
    # Initialize Explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Plot 1: SHAP Summary
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("SHAP Summary: Feature Impact on Price", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary.png"))
    plt.close()

    # Plot 2: Feature Importance (Bar)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.title("Global Feature Importance", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance.png"))
    plt.close()

    # Plot 3: Partial Dependence Plot (RQ1 Analysis)
    # This addresses the Research Question: How do Year and Km affect price?
    print("2. Generating Partial Dependence Plots (RQ1)...")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    features_to_plot = ['yil', 'kilometre']
    
    # We use common_norm=False to visualize trends clearly
    PartialDependenceDisplay.from_estimator(
        model, 
        X, 
        features_to_plot, 
        ax=ax,
        line_kw={"color": "red", "linewidth": 2}
    )
    plt.suptitle('RQ1: Effect of Year and Mileage on Price', fontsize=16)
    plt.subplots_adjust(top=0.9)
    plt.savefig(os.path.join(output_dir, "pdp_analysis_rq1.png"))
    plt.close()

    print(f">>> All plots saved to: {output_dir}")

if __name__ == "__main__":
    generate_plots()