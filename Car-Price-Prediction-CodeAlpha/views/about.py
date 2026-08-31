"""
👨‍💻 About page for the Car Price Prediction Streamlit app.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

from utils import (
    df, model, fuel_encoder, seller_encoder, transmission_encoder,
    asset_path, model_resource_path,
    PROJECT_NAME, AUTHOR, MODEL_NAME, DATASET_NAME, ROWS, COLS,
    BASE_DIR, DATA_PATH, MODEL_PATH,
)


def render_about():

    # ------------------------------------------------------
    # HERO SECTION
    # ------------------------------------------------------

    st.markdown("""
    <div class="hero-container">

    <h1>🚗 Car Price Prediction using Machine Learning</h1>

    <p>

    An end-to-end Machine Learning project that predicts the
    resale value of used cars using historical market data,
    advanced analytics and an interactive Streamlit dashboard.

    </p>

    </div>

    """, unsafe_allow_html=True)

    st.write("")

    # ------------------------------------------------------
    # HERO IMAGE
    # ------------------------------------------------------

    col1, col2 = st.columns([2,1])

    with col1:

        st.markdown("""
### 🌟 Welcome

This application demonstrates the complete Machine Learning
workflow—from data preprocessing and exploratory analysis
to model training, evaluation, and deployment.

It is designed as a portfolio-quality project showcasing
Data Science, Machine Learning and Dashboard Development
skills.
""")

    with col2:

        st.image(
            asset_path("hero.png"),
            use_container_width=True
        )

    st.divider()

    # ------------------------------------------------------
    # PROJECT OVERVIEW
    # ------------------------------------------------------

    st.subheader("📌 Project Overview")

    st.write("""

The objective of this project is to estimate the selling
price of a used car based on various vehicle attributes
such as:

• Present Price

• Kilometers Driven

• Fuel Type

• Seller Type

• Transmission

• Previous Owners

• Vehicle Age

A Random Forest Regression model is trained on historical
CarDekho data to generate accurate predictions.

The application also includes interactive visualizations,
model evaluation dashboards and downloadable reports.

""")

    st.divider()

    # ------------------------------------------------------
    # PROJECT OBJECTIVES
    # ------------------------------------------------------

    st.subheader("🎯 Project Objectives")

    c1, c2 = st.columns(2)

    with c1:

        st.success("""

### 📊 Data Analysis

✔ Data Cleaning

✔ Exploratory Data Analysis

✔ Feature Engineering

✔ Visualization

✔ Statistical Summary

""")

    with c2:

        st.success("""

### 🤖 Machine Learning

✔ Train Regression Model

✔ Evaluate Performance

✔ Feature Importance

✔ Price Prediction

✔ Dashboard Deployment

""")

    st.divider()

    # ------------------------------------------------------
    # MACHINE LEARNING WORKFLOW
    # ------------------------------------------------------

    st.subheader("🔄 Machine Learning Workflow")

    workflow = pd.DataFrame({

        "Step":[

            "1",

            "2",

            "3",

            "4",

            "5",

            "6",

            "7"

        ],

        "Process":[

            "Collect Dataset",

            "Data Cleaning",

            "Feature Engineering",

            "EDA",

            "Train Random Forest",

            "Evaluate Model",

            "Deploy using Streamlit"
        ]

    })

    st.dataframe(

        workflow,

        hide_index=True,

        use_container_width=True

    )

    st.info("""

Machine Learning Pipeline

Dataset
⬇

Cleaning
⬇

EDA
⬇

Feature Engineering
⬇

Model Training
⬇

Evaluation
⬇

Deployment

""")

    st.divider()

    # ------------------------------------------------------
    # TECH STACK
    # ------------------------------------------------------

    st.subheader("🛠 Technology Stack")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.info("""

## 💻 Programming

🐍 Python

📦 NumPy

🐼 Pandas

""")

    with c2:

        st.info("""

## 🤖 Machine Learning

Scikit-Learn

Random Forest

Joblib

""")

    with c3:

        st.info("""

## 📊 Visualization

Plotly

Streamlit

Matplotlib

""")

    st.divider()

    # ------------------------------------------------------
    # DATASET INFORMATION
    # ------------------------------------------------------

    st.subheader("📊 Dataset Information")

    total_rows = df.shape[0]
    total_columns = df.shape[1]

    numeric_columns = len(
        df.select_dtypes(include=np.number).columns
    )

    categorical_columns = len(
        df.select_dtypes(exclude=np.number).columns
    )

    d1, d2, d3, d4 = st.columns(4)

    d1.metric(
        "Rows",
        total_rows
    )

    d2.metric(
        "Columns",
        total_columns
    )

    d3.metric(
        "Numeric Features",
        numeric_columns
    )

    d4.metric(
        "Categorical Features",
        categorical_columns
    )

    st.write("")

    st.dataframe(

        pd.DataFrame({

            "Column":df.columns,

            "Data Type":df.dtypes.astype(str)

        }),

        use_container_width=True,

        hide_index=True

    )

    st.divider()

    # ------------------------------------------------------
    # DATASET FEATURES
    # ------------------------------------------------------

    st.subheader("📋 Input Features")

    features = pd.DataFrame({

        "Feature":[

            "Present Price",

            "Kms Driven",

            "Fuel Type",

            "Seller Type",

            "Transmission",

            "Owner",

            "Year"

        ],

        "Description":[

            "Current showroom price",

            "Total kilometers driven",

            "Petrol/Diesel/CNG",

            "Dealer or Individual",

            "Manual or Automatic",

            "Number of previous owners",

            "Manufacturing year"

        ]

    })

    st.dataframe(

        features,

        hide_index=True,

        use_container_width=True

    )

    st.divider()

    st.success("✅ Dataset successfully loaded and ready for Machine Learning.")

    # ======================================================
    # 👨‍💻 PART 8B
    # Premium About Section
    # ======================================================

    st.markdown("## 👨‍💻 Meet the Developer")

    col1, col2 = st.columns([1,2])

    with col1:

        try:
            st.image(
                asset_path("gaurav.png"),
                use_container_width=True
            )
        except:
            st.info("📷 Add your image to assets/gaurav.png")

    with col2:

        st.markdown("""
### Gaurav Eknath Kumbhar

🎓 MCA Student

📍 Maharashtra, India

💡 Aspiring Data Scientist | Machine Learning Engineer










Passionate about building intelligent machine learning
applications, interactive dashboards, and end-to-end
data science projects using Python and modern AI tools.
""")

        st.success("""
🎯 Career Goal

To become a professional AI & Machine Learning Engineer
while developing real-world solutions using Data Science,
Cloud Computing, and Artificial Intelligence.
""")

    st.divider()

    # ------------------------------------------------------
    # SKILLS
    # ------------------------------------------------------

    st.subheader("🛠 Technical Skills")

    skill1, skill2, skill3 = st.columns(3)

    with skill1:

        st.info("""
### 💻 Programming

🐍 Python

🗄 SQL

📓 Jupyter Notebook

🧩 OOP
""")

    with skill2:

        st.info("""
### 📊 Data Science

Pandas

NumPy

Matplotlib

Plotly

Seaborn
""")

    with skill3:

        st.info("""
### 🤖 Machine Learning

Scikit-Learn

Regression

Classification

Model Evaluation

Feature Engineering
""")

    st.divider()

    # ------------------------------------------------------
    # PROJECT HIGHLIGHTS
    # ------------------------------------------------------

    st.subheader("🏆 Project Highlights")

    h1, h2, h3 = st.columns(3)

    h1.metric("Dataset Records", len(df))
    h2.metric("ML Algorithm", "Random Forest")
    h3.metric("Deployment", "Streamlit")

    st.write("")

    st.success("""
✔ Interactive Machine Learning Dashboard

✔ End-to-End Data Science Workflow

✔ Real-Time Price Prediction

✔ Exploratory Data Analysis

✔ Model Performance Dashboard

✔ Downloadable Reports

✔ Responsive User Interface
""")

    st.divider()

    # ------------------------------------------------------
    # EDUCATION
    
    # ------------------------------------------------------
    # PROJECT FEATURES
    # ------------------------------------------------------

    st.subheader("🚀 Project Features")

    left, right = st.columns(2)

    with left:

        st.success("""
### 📊 Analytics

✔ Dataset Explorer

✔ Interactive Charts

✔ Statistical Analysis

✔ Correlation Heatmap

✔ Business Insights
""")

    with right:

        st.success("""
### 🤖 AI Features

✔ Price Prediction

✔ Feature Importance

✔ Vehicle Health Score

✔ Smart Recommendations

✔ Model Evaluation
""")

    st.divider()

    # ------------------------------------------------------
    # CONNECT
    # ------------------------------------------------------

    st.subheader("🌐 Connect With Me")

    github = st.text_input(
        "GitHub Profile",
        value="https://github.com/yourusername"
    )

    linkedin = st.text_input(
        "LinkedIn Profile",
        value="https://linkedin.com/in/yourusername"
    )

    email = st.text_input(
        "Email",
        value="your.email@example.com"
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.link_button("💻 GitHub", github)

    with c2:
        st.link_button("💼 LinkedIn", linkedin)

    with c3:
        st.link_button("📧 Email", f"mailto:{email}")

    st.divider()

    # ------------------------------------------------------
    # CERTIFICATIONS
    # ------------------------------------------------------

 
    # ------------------------------------------------------
    # THANK YOU
    # ------------------------------------------------------

    st.markdown("""

<div class="prediction-card">

<h2>🙏 Thank You!</h2>

<p>

Thank you for exploring this project.

If you found this application useful,
please consider giving the repository a ⭐ on GitHub.

</p>

</div>

""", unsafe_allow_html=True)

    st.markdown("""

<div class="footer">

<h3>🚗 Car Price Prediction using Machine Learning</h3>

<p>

Designed & Developed by

<b>Gaurav Eknath Kumbhar</b>

<br><br>

Python • Streamlit • Scikit-Learn • Plotly

</p>

</div>

""", unsafe_allow_html=True)

# ==========================================================
# 📬 CONTACT
# ==========================================================

