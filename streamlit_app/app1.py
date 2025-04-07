import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import datetime
import pickle
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Load the pickled model
with open('models/model(XGB).pikle', 'rb') as f:
    pipeline = pickle.load(f)

# Save using joblib and reload
joblib.dump(pipeline, "xgb_pipeline.joblib")
pipeline = joblib.load('xgb_pipeline.joblib')

# Load preprocessed dataframe (optional for EDA)
with open('data/processed/df', 'rb') as file:
    df = pickle.load(file)

# Sidebar Navigation
st.sidebar.title("Flight Price Prediction System")
selection = st.sidebar.radio("Go to", ["Home", "Price Prediction", "Evaluation"])

# Home Page
if selection == "Home":
    st.title("Welcome to the Flight Price Prediction System")
    st.write("This application predicts flight prices based on various features and provides insights through data analysis and visualization.")
    
    st.markdown("## Features:")
    st.markdown("- **Price Prediction:** Predict flight prices based on user-provided details like airline, route, and travel class.")
    st.markdown("- **Data Cleaning & Preprocessing:** Handles missing values, transforms date and duration fields, and encodes categorical features.")
    st.markdown("- **Exploratory Data Analysis (EDA):** Analyze trends such as price distribution across airlines, stops, and duration.")
    st.markdown("- **Model Training & Evaluation:** Uses machine learning models like Random Forest and XGBoost with proper cross-validation.")
    st.markdown("- **Interactive Deployment:** Powered by Streamlit, allowing real-time prediction and evaluation.")

# Price Prediction Page
if selection == "Price Prediction":
    st.title("Price Prediction 📈")
    st.write("Enter your flight details to predict the estimated flight price.")
    st.header('Enter your inputs:')

    # Inputs for price prediction
    Departure_city = st.selectbox('Departure City', ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai'])
    Arrival_city = st.selectbox('Arrival City', ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai'])
    Class = st.selectbox('Class', ['economy', 'business'])
    Airline = st.selectbox('Airline', ['SpiceJet', 'AirAsia', 'Vistara', 'GO FIRST', 'Indigo', 'Air India', 'Trujet', 'StarAir'])
    flight_month = st.number_input('Flight Month', min_value=1, max_value=12, step=1)
    Day_of_week = st.number_input('Day of Week', min_value=0, max_value=6, step=1)

    user_time = st.time_input("Departure Time", value=datetime.time(0, 0))
    Departure_Time = user_time.hour * 60 + user_time.minute

    arrival_time_input = st.time_input("Arrival Time", value=datetime.time(0, 0))
    Arrival_Time = arrival_time_input.hour * 60 + arrival_time_input.minute

    # Calculate duration in minutes, handling overnight flights
    Duration = (Arrival_Time - Departure_Time) % (24 * 60)

    # Prediction Button
    if st.button('Predict'):
      input_data = {
        'departure_city': Departure_city,
        'arrival_city': Arrival_city,
        'class': Class,
        'airline': Airline,
        'flight_month': flight_month,
        'day_of_week': Day_of_week,
        'departure_time': Departure_Time,
        'arrival_time': Arrival_Time,
        'duration': Duration}
      input_df = pd.DataFrame([input_data])
      prediction = pipeline.predict(input_df)[0]
      st.write(f"Estimated Flight Price: ₹{prediction:.2f}")
# Evaluation Page
if selection == "Evaluation":
    st.title("📊 Model Performance Evaluation")

    if 'price' not in df.columns:
        st.error("Dataset does not contain actual prices for evaluation.")
    else:
        st.subheader("🔍 Evaluation Metrics")
        st.write("The model has been evaluated using standard regression metrics to assess its prediction performance on unseen data.")

        X = df.drop(columns=['price'])
        y_true = df['price']
        y_pred = pipeline.predict(X)

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        st.metric("📌 Mean Absolute Error (MAE)", f"{mae:.2f}", help="Average of the absolute errors between actual and predicted prices.")
        st.metric("📌 Mean Squared Error (MSE)", f"{mse:.2f}", help="Average of squared differences between actual and predicted prices.")
        st.metric("📌 Root Mean Squared Error (RMSE)", f"{rmse:.2f}", help="Square root of MSE. Lower values indicate better model performance.")
        st.metric("📌 R² Score", f"{r2:.4f}", help="Proportion of the variance in the dependent variable that is predictable from the features.")

        st.subheader("📈 Actual vs Predicted Price Comparison")
        fig = px.scatter(
            x=y_true, 
            y=y_pred, 
            labels={'x': 'Actual Price', 'y': 'Predicted Price'},
            title="Scatter Plot: Actual vs Predicted Flight Prices",
            template="plotly_white",
            color_discrete_sequence=["#636EFA"]
        )
        fig.update_traces(marker=dict(size=5, opacity=0.6))
        fig.update_layout(
            height=500,
            margin=dict(l=20, r=20, t=50, b=20),
            title_x=0.5
        )
        st.plotly_chart(fig, use_container_width=True)

        st.success("Model evaluation completed successfully. The metrics and plot above provide insights into how well the model is performing.")
