import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import joblib
import shap

# Load dataset, model, and scaler
df = pd.read_csv('wallmart_sales_forecasting_cleaned.csv')
model = joblib.load("model_lightgbm.pkl")
scaler = joblib.load("scaler.pkl")
st.title("Sales Forecasting Dashboard")

# -------------------------------
# Sidebar Filters
# -------------------------------
st.sidebar.title('1. Filter based on store and date')

with st.sidebar.expander('Store'):
    store_selected = st.selectbox('Select Store', df['Store'].unique())
    department_selected = st.selectbox(
        'Select Department', 
        df[df['Store'] == store_selected]['Dept'].unique()
    )

with st.sidebar.expander('Date'):
    year_selected = st.selectbox('Select Year', df['Year'].unique())
    month_selected = st.selectbox(
        'Select Month',
        df[df['Year'] == year_selected]['Month'].unique()
    )

st.sidebar.title('2. Filter based on Holiday')
isHoliday = st.sidebar.radio('Pick One', ["Holiday", "No Holiday"])
isHoliday = 1 if isHoliday == "Holiday" else 0

# -------------------------------
# Filter weekly sales
# -------------------------------
weekly_sales = df[
    (df['Store'] == store_selected) &
    (df['Dept'] == department_selected) &
    (df['Year'] == year_selected) &
    (df['Month'] == month_selected) &
    (df['IsHoliday'] == isHoliday)
]

weekly_sales = weekly_sales.copy()

total_weekly_sales = weekly_sales.groupby('Week')['Weekly_Sales'].sum()


if not total_weekly_sales.empty:
    graph_choice = st.radio(
        "Select a graph to display:",
        ('Original', 'Predicted', 'Both'),
        index=2,
        horizontal=True
    )

    # -------------------------------
    # Prepare input for prediction
    # -------------------------------
    weeks = list(total_weekly_sales.index)
    input_list = []
    for week in weeks:
        row = weekly_sales[weekly_sales['Week'] == week].iloc[0]
        day = row['Day']
        day_sin = np.sin(2 * np.pi * day / 31)
        day_cos = np.cos(2 * np.pi * day / 31)
        month_sin = np.sin(2 * np.pi * month_selected / 12)
        month_cos = np.cos(2 * np.pi * month_selected / 12)

        input_list.append({
            'Store': store_selected,
            'Dept': department_selected,
            'Temperature': float(row['Temperature']),
            'Fuel_Price': float(row['Fuel_Price']),
            'CPI': float(row['CPI']),
            'Unemployment': float(row['Unemployment']),
            'IsHoliday': isHoliday,
            'Year': year_selected,
            'Week': week,
            'Day_sin': day_sin,
            'Day_cos': day_cos,
            'Month_sin': month_sin,
            'Month_cos': month_cos
        })

    feature_cols = ['Store', 'Dept', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
                    'IsHoliday', 'Year', 'Week', 'Day_sin', 'Day_cos', 'Month_sin', 'Month_cos']
    input_data = pd.DataFrame(input_list)[feature_cols]

    # Scale and predict
    input_scaled = scaler.transform(input_data)
    predicted_sales = model.predict(input_scaled)
    weekly_sales['Predicted_Sales'] = predicted_sales

    # -------------------------------
    # Confidence intervals (simple)
    # -------------------------------
    ci = np.std(predicted_sales) * 1.96 / np.sqrt(len(predicted_sales))
    weekly_sales['CI_Lower'] = weekly_sales['Predicted_Sales'] - ci
    weekly_sales['CI_Upper'] = weekly_sales['Predicted_Sales'] + ci

    # -------------------------------
    # KPI Cards
    # -------------------------------
    st.subheader("Key Performance Indicators (KPIs)")
    total_forecast = weekly_sales['Predicted_Sales'].sum()
    peak_week = weekly_sales.loc[weekly_sales['Predicted_Sales'].idxmax(), 'Week']
    holiday_diff = weekly_sales['Weekly_Sales'].sum() - weekly_sales['Predicted_Sales'].sum()
    change_pct = ((weekly_sales['Predicted_Sales'].sum() - weekly_sales['Weekly_Sales'].sum()) /
                  weekly_sales['Weekly_Sales'].sum()) * 100

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Forecasted Sales", f"${total_forecast:,.0f}")
    kpi2.metric("Peak Demand Week", peak_week)
    kpi3.metric("Holiday Week Sales Diff", f"${holiday_diff:,.0f}")
    kpi4.metric("Expected Change %", f"{change_pct:.2f}%")

    # -------------------------------
    # Main Plot: Actual vs Predicted + CI
    # -------------------------------
    if graph_choice in ('Original', 'Both'):
        fig = px.line(total_weekly_sales, x=total_weekly_sales.index, y=total_weekly_sales.values,
                      markers=True, title="Original Sales across Weeks")
        fig.update_layout(title=dict(x=0.5, xanchor='center', font=dict(size=24)),
                          xaxis_title="Week", yaxis_title="Total Sales",
                          xaxis=dict(showgrid=True), yaxis=dict(showgrid=True))
        st.plotly_chart(fig, width='stretch')

    if graph_choice in ('Predicted', 'Both'):
        fig = px.line(weekly_sales, x='Week', y='Predicted_Sales', markers=True,
                      title="Predicted Sales across Weeks")
        fig.add_traces(px.scatter(weekly_sales, x='Week', y='CI_Lower', opacity=0.2).data)
        fig.add_traces(px.scatter(weekly_sales, x='Week', y='CI_Upper', opacity=0.2).data)
        fig.update_layout(title=dict(x=0.5, xanchor='center', font=dict(size=24)),
                          xaxis_title="Week", yaxis_title="Predicted Sales",
                          xaxis=dict(showgrid=True), yaxis=dict(showgrid=True))
        st.plotly_chart(fig, width='stretch')

    # -------------------------------
    # Optional Feature Impact (SHAP)
    # -------------------------------
    st.subheader("Feature Impact (SHAP Values)")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_scaled)
    mean_shap = np.abs(shap_values).mean(axis=0)
    feature_impact = pd.DataFrame({'Feature': feature_cols, 'Impact': mean_shap})
    feature_impact = feature_impact.sort_values(by='Impact', ascending=False)
    fig_feat = px.bar(feature_impact, x='Feature', y='Impact', title="Feature Impact on Sales")
    st.plotly_chart(fig_feat, width='stretch')

    # -------------------------------
    # Data Table
    # -------------------------------
    st.subheader("Forecast Data Table")
    st.dataframe(weekly_sales[['Store','Dept','Year','Month','Week','Weekly_Sales',
                               'Predicted_Sales','CI_Lower','CI_Upper']])

    # -------------------------------
    # Download Button
    # -------------------------------
    csv = weekly_sales.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Forecast CSV",
        data=csv,
        file_name=f'forecast_store{store_selected}_dept{department_selected}.csv',
        mime='text/csv'
    )
