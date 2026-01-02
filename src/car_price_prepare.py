# File: src/car_price_prepare.py

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

def load_and_clean_data():
    print(">>> Starting data preparation pipeline...")
    
    # Define paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    INPUT_FILE = os.path.join(BASE_DIR, "data", "vehicle_data.csv")
    OUTPUT_FILE = os.path.join(BASE_DIR, "data", "clean_car_data.csv")
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found at {INPUT_FILE}")
        return

    # 1. Load Data
    df = pd.read_csv(INPUT_FILE)
    print(f"Original Data Shape: {df.shape}")

    # 2. Drop irrelevant columns
    if 'id' in df.columns:
        df.drop(columns=['id'], inplace=True)

    # 3. Clean critical missing values
    df.dropna(subset=['fiyat', 'yil', 'marka', 'seri'], inplace=True)

    # 4. Impute missing values (Median & Mode)
    if 'motor_hacmi' in df.columns:
        df['motor_hacmi'] = df['motor_hacmi'].fillna(df['motor_hacmi'].median())
    if 'motor_gucu' in df.columns:
        df['motor_gucu'] = df['motor_gucu'].fillna(df['motor_gucu'].median())

    df['degisen_sayisi'] = df['degisen_sayisi'].fillna(0)
    df['boyali_sayisi'] = df['boyali_sayisi'].fillna(0)
    df['model'] = df['model'].fillna('Bilinmiyor')

    # Filling categorical gaps with Mode (from your snippets)
    for col in ['vites_tipi', 'kasa_tipi', 'renk']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # 5. Outlier Removal (Your strict filters)
    # Price between 50k and 20M TL
    df = df[(df['fiyat'] > 50000) & (df['fiyat'] < 20000000)]
    # Km under 1M
    df = df[df['kilometre'] < 1000000]

    # 6. Type Conversion
    numeric_cols = ['yil', 'kilometre', 'motor_hacmi', 'motor_gucu', 'degisen_sayisi', 'boyali_sayisi']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # 7. LOG TRANSFORMATION (Crucial step)
    df['fiyat_log'] = np.log1p(df['fiyat'])
    print("Applied Log-Transformation to price.")

    # 8. ENCODING (Your logic: LabelEncoder + OneHot)
    
    # Label Encoding for high cardinality
    le = LabelEncoder()
    high_card_cols = ['marka', 'seri', 'model', 'renk']
    for col in high_card_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
            print(f"Applied Label Encoding to: {col}")

    # One-Hot Encoding for low cardinality
    # We apply this now and save the fully numeric dataframe
    dummy_cols = ['vites_tipi', 'yakit_tipi', 'kasa_tipi', 'kimden']
    # Check which columns exist before encoding
    existing_dummy_cols = [c for c in dummy_cols if c in df.columns]
    
    df = pd.get_dummies(df, columns=existing_dummy_cols, drop_first=True)
    print("Applied One-Hot Encoding.")

    # 9. Save Processed Data
    df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"Final Processed Data Shape: {df.shape}")
    print(f"Data saved to: {OUTPUT_FILE}")
    print(">>> Data preparation complete.")

if __name__ == "__main__":
    load_and_clean_data()