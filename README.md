# 🚗 Used Car Price Prediction (Turkey)

## 📖 Project Description
Predicting used car prices in Turkey is challenging due to **economic fluctuations** and **high inflation**.  
In this project, we built a **regression-based machine learning model** to estimate car prices using features such as:

- **Brand**
- **Model**
- **Production Year**
- **Mileage (km)**
- **Engine Power**

The main objectives are:
- 🎯 Estimating the **fair market value** of used cars  
- 🔍 Understanding **which features affect prices the most** using explainability tools like **SHAP**

---

## 📂 Dataset & Preprocessing
The dataset consists of approximately **50,000 used car listings** collected from the Turkish market.

### 🛠️ Preprocessing Steps

#### 🔹 Data Cleaning
- Removed price outliers:
  - Prices `< 50,000 TL`
  - Prices `> 20,000,000 TL`
- Removed vehicles with mileage `> 1,000,000 km`

#### 🔹 Missing Value Handling
- **Numerical features** (`motor_hacmi`, `motor_gucu`) filled with **Median**
- **Categorical features** filled with **Mode**

#### 🔹 Feature Engineering
- **Log Transformation:**  
  Applied `np.log1p()` to the target variable (**Price**) to reduce skewness
- **Encoding:**
  - Label Encoding → Brand, Model
  - One-Hot Encoding → Gear Type, Fuel Type

---

## 🤖 Methodology
We experimented with three different regression models:

1. **Linear Regression (OLS)**  
   - Used as a baseline model
2. **XGBoost Regressor**  
   - Gradient boosting-based ensemble method
3. **Random Forest Regressor** ⭐  
   - Selected as the **final model** due to best performance

---

## 📈 Model Performance (Test Set)

All models were evaluated on **real TL prices** (after inverse log transformation).

| Model | R² Score | MAE (TL) | RMSE (TL) | Status |
|------|----------|----------|-----------|--------|
| **Random Forest** | **0.8984** | **88,540 ₺** | **325,017 ₺** | 🏆 **Best Model** |
| XGBoost | 0.8875 | 101,719 ₺ | 341,924 ₺ | Competitive |
| Linear Regression | 0.6561 | 195,877 ₺ | 597,841 ₺ | Baseline |

> **Insight:**  
> Tree-based models reduced prediction error by approximately **50%** compared to Linear Regression.

---

## 🔍 Visualizations & Explainability

### 📌 SHAP Summary Plot
Shows how each feature impacts the prediction:
- 🔴 Red → increases price
- 🔵 Blue → decreases price

![SHAP Summary](outputs/shap_summary.png)

---

### 📌 Feature Importance
Displays the most influential features in the Random Forest model.

![Feature Importance](outputs/feature_importance.png)

---

### 📌 Year & Mileage Analysis (RQ1)
Partial Dependence Plots showing how:
- Vehicle age  
- Mileage  

affect car prices.

![PDP Analysis](outputs/pdp_analysis_rq1.png)

---

## 🚀 How to Run the Project

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 2️⃣ Prepare Dataset
```bash
python src/car_price_prepare.py
```
### 3️⃣ Train the Model
```bash
python src/car_price_train.py
```
### 4️⃣ Generate SHAP & Visualizations
```bash
python src/car_price_shap.py
```
### 👥 Authors
```bash
[ANIL AYDIN] - [220717047]

[HAKAN ENES ERİŞEN] - [220717605]
```
