# student_performance_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn import linear_model as lm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_squared_error, r2_score

# --- Streamlit UI ---
st.title("🎓 Student Performance Prediction App")

# Upload CSV
uploaded_file = st.file_uploader("Upload your Student Performance CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("✅ Data Loaded Successfully!")
    
    st.subheader("Preview of Data")
    st.dataframe(data.head(20))

    st.subheader("Dataset Info")
    st.write(data.describe())

    st.subheader("Checking Null Values")
    st.write(data.isnull().sum())

    st.write(f"Shape of the dataset: {data.shape}")

    # Preprocessing
    data['Extracurricular Activities'].replace(['Yes', 'No'], [True, False], inplace=True)

    st.subheader("Correlation Heatmap")
    fig = px.imshow(data.corr(), text_auto=True, color_continuous_scale='RdBu')
    st.plotly_chart(fig)

    # Feature Selection
    y = data['Performance Index']
    X = data.drop('Performance Index', axis=1)

    # Split Data
    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.15, random_state=42)

    st.subheader("Training Info")
    st.write(f"Training Set Shape: {X_train.shape}")
    st.write(f"Testing Set Shape: {X_test.shape}")

    # Linear Regression Model
    model = lm.LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    st.subheader("Model Performance")
    st.write(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.2f}")
    st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")

    # Plot Actual vs Predicted
    st.subheader("📈 Actual vs Predicted Scatter Plot")
    plot_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred
    })

    scatter_plot = px.scatter(
        plot_df,
        x='Actual',
        y=plot_df.index,
        title="Actual vs Predicted",
        labels={"Actual": "Actual Values", "index": "Index"},
    )

    scatter_plot.add_scatter(
        x=plot_df['Predicted'],
        y=plot_df.index,
        mode='markers',
        hovertemplate="Predicted: %{x}<br>Index: %{y}",
        marker=dict(opacity=0.5),
        name="Predicted"
    )

    st.plotly_chart(scatter_plot)

else:
    st.info("👈 Please upload a CSV file to start.")

