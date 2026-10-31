# 🚗 Car Price Prediction

An end-to-end machine learning project for estimating the selling price of used cars from vehicle characteristics. The project includes exploratory analysis, a trained regression model, and an interactive Streamlit application.

## 🎯 Project Objective

Build a regression workflow that helps estimate used-car selling prices from available vehicle information.

## 📊 Dataset

The included CarDekho dataset contains **301 rows and 9 columns**. The saved training information records **299 processed samples**, **7 model features**, a **239/60 train-test split**, and the following features:

- Present Price
- Driven Kilometers
- Fuel Type
- Selling Type
- Transmission
- Owner
- Car Age

## 🤖 Verified Model Results

The saved project artifacts identify the deployed model as a **Decision Tree Regressor**.

| Metric | Verified value |
|---|---:|
| R² Score | 0.8252 |
| MAE | 1.1478 |
| RMSE | 2.1227 |
| Training Samples | 239 |
| Testing Samples | 60 |

> These values come from the project's saved `model_metrics.pkl` and `training_info.pkl` artifacts.

## ✨ Features

- Interactive car price prediction
- Dataset exploration
- Exploratory data analysis
- Actual vs. predicted visualization using saved test results
- Residual error analysis
- Feature-importance visualization
- Downloadable project resources

## 🛠️ Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Plotly
- Streamlit
- Joblib

## 📁 Project Structure

```text
Car-Price-Prediction-CodeAlpha/
├── app.py
├── data/
├── models/
├── views/
├── assets/
├── requirements.txt
└── README.md
```

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📌 Important Implementation Note

The application documentation and performance page are aligned with the **saved Decision Tree model** and its stored evaluation metrics. Avoid claiming a different model or placeholder performance values unless the model is retrained and the saved artifacts are updated.

