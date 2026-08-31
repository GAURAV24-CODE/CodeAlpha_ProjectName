"""
🏠 Home page for the Car Price Prediction Streamlit app.
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


def render_home():

    # ------------------------------------------------------
    # HERO SECTION
    # ------------------------------------------------------

    st.markdown("""
    <div class="hero-container">
        <h1>🚗 Car Price Prediction using Machine Learning</h1>
        <p>
        Estimate the resale value of used cars using a trained
        Random Forest Regression model. Explore the dataset,
        visualize insights, and make real-time predictions
        through an interactive dashboard.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # Hero Banner
    if os.path.exists(asset_path("home_banner.png")):
        st.image(asset_path("home_banner.png"), use_container_width=True)

    st.write("")

    # ------------------------------------------------------
    # KPI CARDS
    # ------------------------------------------------------

    st.subheader("📊 Project Statistics")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Dataset Rows",
        f"{df.shape[0]}"
    )

    c2.metric(
        "Features",
        f"{df.shape[1]-1}"
    )

    c3.metric(
        "Algorithm",
        "Random Forest"
    )

    c4.metric(
        "Predictions",
        "Real-Time"
    )

    st.divider()

    # ------------------------------------------------------
    # ABOUT PROJECT
    # ------------------------------------------------------

    left, right = st.columns([2,1])

    with left:

        st.subheader("📌 About This Project")

        st.write("""
This project predicts the **selling price of used cars**
using Machine Learning.

The model has been trained using the **CarDekho Used Car Dataset**
and leverages several important features such as:

- Present Price
- Kilometers Driven
- Fuel Type
- Seller Type
- Transmission
- Number of Owners
- Car Age

The application allows users to:

- Predict resale prices
- Explore the dataset
- Visualize trends
- Evaluate model performance
- Download project resources
""")

    with right:

        st.subheader("📦 Dataset")

        st.info(f"""
Dataset Name

{DATASET_NAME}

Rows

{ROWS}

Columns

{COLS}
""")

        st.success("""
Machine Learning

✔ Random Forest

✔ Regression

✔ Feature Engineering

✔ Data Visualization
""")

    st.divider()

    # ------------------------------------------------------
    # TECHNOLOGY STACK
    # ------------------------------------------------------

    st.subheader("🛠 Technology Stack")

    t1, t2, t3, t4 = st.columns(4)

    with t1:
        st.markdown("""
### 🐍 Python

- Pandas
- NumPy
""")

    with t2:
        st.markdown("""
### 📊 Visualization

- Plotly
- Matplotlib
- Seaborn
""")

    with t3:
        st.markdown("""
### 🤖 Machine Learning

- Scikit-Learn
- Random Forest
- Joblib
""")

    with t4:
        st.markdown("""
### 🌐 Deployment

- Streamlit
- GitHub
""")

    st.divider()

    # ------------------------------------------------------
    # MACHINE LEARNING PIPELINE
    # ------------------------------------------------------

    st.subheader("⚙ Machine Learning Workflow")

    st.code("""
Raw Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Feature Engineering
      │
      ▼
Encoding
      │
      ▼
Train-Test Split
      │
      ▼
Random Forest Model
      │
      ▼
Prediction
""")

    st.divider()

    # ------------------------------------------------------
    # PROJECT FEATURES
    # ------------------------------------------------------

    st.subheader("✨ Application Features")

    f1, f2 = st.columns(2)

    with f1:

        st.success("""
✅ Real-Time Price Prediction

✅ Interactive Dashboard

✅ Plotly Charts

✅ Data Exploration

✅ Responsive UI
""")

    with f2:

        st.success("""
✅ Model Performance

✅ Download Dataset

✅ Feature Importance

✅ Machine Learning

✅ Professional Layout
""")

    st.divider()

    # ------------------------------------------------------
    # QUICK DATA OVERVIEW
    # ------------------------------------------------------

    st.subheader("📋 Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True,
        height=350
    )

    st.divider()

    # ------------------------------------------------------
    # TARGET DISTRIBUTION
    # ------------------------------------------------------

    st.subheader("💰 Selling Price Distribution")

    fig = px.histogram(
        df,
        x="Selling_Price",
        nbins=30,
        color_discrete_sequence=["#0d6efd"],
        template="plotly_white"
    )

    fig.update_layout(
        height=450,
        title="Selling Price Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # ------------------------------------------------------
    # WHY THIS PROJECT
    # ------------------------------------------------------

    st.subheader("🎯 Project Objectives")

    st.write("""
The primary goal of this project is to estimate the resale value
of used cars using historical vehicle information.

This application demonstrates a complete Machine Learning workflow,
including:

- Data preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Training
- Model Evaluation
- Streamlit Deployment
""")

    st.divider()

    # ------------------------------------------------------
    # AUTHOR
    # ------------------------------------------------------
    # ==========================================
    # AUTHOR SECTION
    # ==========================================

    st.markdown("<hr>", unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div style="
        background: linear-gradient(90deg,#0F2027,#203A43,#2C5364);
        padding:20px;
        border-radius:15px;
        text-align:center;
        color:white;
        margin-bottom:25px;
    ">
        <h2 style="margin-bottom:5px;"> ABOUT THE AUTHOR</h2>
        <h3 style="margin-top:0;">Gaurav Eknath Kumbhar</h3>
        <p style="font-size:18px;">
            Data Scientist | Machine Learning Engineer | MCA Student
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Two Columns
    col1, col2 = st.columns([1, 2], gap="large")

    # ==========================================
    # LEFT COLUMN
    # ==========================================

    with col1:

        if os.path.exists(asset_path("gaurav.png")):
            st.image(
                asset_path("gaurav.png"),
                use_container_width=True
            )
        else:
            st.warning("Profile Image Not Found!")

    # ==========================================
    # RIGHT COLUMN
    # ==========================================

    with col2:

        st.markdown("## 👨‍💻 Developer")

        st.markdown(f"""
### **{AUTHOR}**

🎓 **MCA Student**

🚀 **Aspiring Data Scientist**

💻 **Python | Machine Learning | Data Analytics**

---

✅ Passionate about solving real-world problems using Machine Learning.

✅ Skilled in Python, SQL, Power BI, Streamlit and Data Visualization.

✅ Currently building AI & Data Science projects.

---

📧 **Email:** kumbhargaurav24.com

🌐 **GitHub:** https://github.com/GAURAV24-CODE

🔗 **LinkedIn:**https://www.linkedin.com/in/gaurav-kumbhar-0b4a39293?utm_source=share_via&utm_content=profile&utm_medium=member_android
""")

    st.markdown("<hr>", unsafe_allow_html=True)
    # ------------------------------------------------------
    # FOOTER
    # ------------------------------------------------------

 
# ==========================================================
# 🚗 PRICE PREDICTION PAGE (PART 3A.1)
# Layout + Input UI + Validation
# ==========================================================

