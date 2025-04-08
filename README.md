# Flight-Price-Prediction-System
A machine learning model to predict flight ticket prices based on airline, route, and other factors.
# ✈️ Flight Price Prediction System

![Flight App Screenshot](Images/Screenshot%20%28445%29.png)


## 📍 Project Overview

The **Flight Price Prediction System** is a full-stack data science project that predicts flight ticket prices based on key parameters like airline, class, source, destination, and travel times. The project includes data cleaning, feature engineering, exploratory data analysis (EDA), model training using XGBoost, and interactive deployment using **Streamlit**.

This project is designed for real-world applications and demonstrates end-to-end deployment of a machine learning model through an intuitive web interface.

---

## 🚀 Deployment

🌐 The app is deployed using **Streamlit** and includes three main sections:
- **Home**: Project introduction, features.
- **Price Prediction**: Real-time prediction form using XGBoost.
- **Evaluation**: Displays model metrics (MAE, MSE, RMSE, R²) and visual comparison.

---

## 🔍 Features

- 🎯 **Price Prediction**: Predicts flight fare using trained ML model based on user input.
- 🧹 **Data Cleaning**: Handled nulls, converted time formats, encoded categorical variables.
- 📊 **EDA**: Visualization of price distribution across cities, airlines, and flight times.
- 🧠 **ML Model**: Trained using **XGBoost** with hyperparameter tuning.
- 📈 **Evaluation Metrics**: MAE, MSE, RMSE, and R² Score.
- 🧪 **Interactive Deployment**: Streamlit interface for non-technical users.

---

## 🧰 Tools and Technologies

| Area | Libraries / Frameworks |
|------|------------------------|
| Data Cleaning & EDA | `pandas`, `numpy`, `plotly` |
| Model Building | `xgboost`, `sklearn`, `pickle`, `joblib` |
| App Interface | `streamlit` |
| Visualization | `plotly.express` |

---
## 🧪 How to Run the App Locally

1. Clone the repository:
git clone https://github.com/yourusername/flight-price-prediction-app.git
cd flight-price-prediction-app

