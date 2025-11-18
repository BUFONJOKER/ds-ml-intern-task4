# 🛒 Walmart Sales Forecasting Dashboard
[Live app](https://abdulrehman-ds-ml-task4.streamlit.app/)

## 📌 Overview
Interactive Streamlit app to forecast weekly sales for Walmart stores/departments using a pre-trained LightGBM model. Predictions use economic indicators (CPI, Unemployment, Fuel Price) and temporal features. The dashboard includes KPIs, interactive visualizations, and SHAP-based explainability.

## ✨ Key Features
- Interactive filtering by Store, Department, Year, Month, and Holiday status
- Instant forecasts from a serialized LightGBM model
- Visualizations:
    - Interactive Actual vs. Predicted line charts (Plotly)
    - Confidence interval visualization
    - SHAP analysis to show feature contributions
- Business KPIs: Total Forecast, Peak Demand Week, Holiday Sales impact
- Export: download forecasted results as CSV

## 🛠️ Tech Stack
- Frontend: Streamlit
- Data manipulation: pandas, NumPy
- Visualization: Plotly Express
- ML: LightGBM, scikit-learn (scaler), joblib
- Explainability: SHAP

## 📂 Project Structure
```
walmart-sales-forecasting/
├── app.py                       # Main Streamlit app
├── model_lightgbm.pkl           # Pre-trained LightGBM model
├── scaler.pkl                   # Pre-trained Standard Scaler
├── walmart_sales_forecasting_cleaned.csv  # Historical dataset
├── pyproject.toml             # Python dependencies
└── README.md                    # Project documentation
```

## 🚀 Installation & Setup
1. Clone the repository:
     ```bash
     git clone https://github.com/BUFONJOKER/ds-ml-intern-task4.git
     cd walmart-sales-forecasting
     ```
2. Create uv project:
     ```bash
     uv init walmart-sales-forecasting
     ```
3. Install dependencies:
     ```bash
     uv sync
     ```
4. Run the app:
     ```bash
     streamlit run app.py
     ```

## 📊 Methodology
- Data preprocessing:
    - Handle missing values and categorical encodings
    - Temporal features encoded cyclically (sine/cosine for day/month) to capture seasonality
- Modeling:
    - LightGBM regressor trained on historical features for efficiency on tabular data
- Scaling:
    - Numerical inputs (CPI, Temperature, Fuel Price, etc.) scaled using a saved StandardScaler to match training distribution
- Explainability:
    - SHAP values used to attribute predictions to individual features

