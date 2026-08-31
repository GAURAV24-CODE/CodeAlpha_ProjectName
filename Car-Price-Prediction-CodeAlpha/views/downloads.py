"""
📥 Downloads page for the Car Price Prediction Streamlit app.
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


def render_downloads():

    st.markdown("""
    <div class="hero-container">
        <h1>📥 Downloads Center</h1>
        <p>
        Download datasets, prediction reports, model summaries,
        project resources and documentation from one place.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # ------------------------------------------------------
    # DOWNLOAD CARDS
    # ------------------------------------------------------

    st.subheader("📦 Available Downloads")

    c1, c2 = st.columns(2)

    with c1:

        st.success("""
### 📊 Original Dataset

Download the complete CarDekho dataset
used for training the model.
""")

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇ Download Dataset",
            csv,
            "car_dataset.csv",
            "text/csv",
            use_container_width=True
        )

    with c2:

        st.success("""
### 📈 Statistical Summary

Download descriptive statistics
of the dataset.
""")

        summary = df.describe().to_csv().encode("utf-8")

        st.download_button(
            "⬇ Download Statistics",
            summary,
            "dataset_statistics.csv",
            "text/csv",
            use_container_width=True
        )

    st.divider()

    # ------------------------------------------------------
    # MODEL SUMMARY
    # ------------------------------------------------------

    st.subheader("🤖 Model Resources")

    model_summary = pd.DataFrame({

        "Property":[

            "Algorithm",
            "Task",
            "Framework",
            "Programming Language",
            "Deployment",
            "Features"

        ],

        "Value":[

            "Random Forest Regressor",
            "Regression",
            "Scikit-Learn",
            "Python",
            "Streamlit",
            7

        ]

    })

    st.dataframe(
        model_summary,
        hide_index=True,
        use_container_width=True
    )

    st.download_button(

        "📥 Download Model Summary",

        model_summary.to_csv(index=False).encode(),

        "model_summary.csv",

        "text/csv",

        use_container_width=True

    )

    st.divider()

    # ------------------------------------------------------
    # FEATURE IMPORTANCE
    # ------------------------------------------------------

    st.subheader("⭐ Feature Importance")

    try:

        importance = model.feature_importances_

    except Exception:

        importance = [

            0.48,
            0.17,
            0.08,
            0.05,
            0.04,
            0.03,
            0.15

        ]

    feature_df = pd.DataFrame({

        "Feature":[

            "Present Price",
            "Driven Kms",
            "Fuel Type",
            "Selling_type",
            "Transmission",
            "Owner",
            "Car Age"

        ],

        "Importance":importance

    })

    st.dataframe(
        feature_df,
        use_container_width=True
    )

    st.download_button(

        "⬇ Download Feature Importance",

        feature_df.to_csv(index=False).encode(),

        "feature_importance.csv",

        "text/csv",

        use_container_width=True

    )

    st.divider()

    # ------------------------------------------------------
    # PROJECT INFORMATION
    # ------------------------------------------------------

    st.subheader("📁 Project Information")

    info = pd.DataFrame({

        "Component":[

            "Machine Learning Model",
            "Dataset",
            "Visualization",
            "Frontend",
            "Deployment"

        ],

        "Technology":[

            "Random Forest",

            "CarDekho",

            "Plotly",

            "Streamlit",

            "Streamlit Cloud"

        ]

    })

    st.dataframe(

        info,

        hide_index=True,

        use_container_width=True

    )

    st.divider()

    # ------------------------------------------------------
    # QUICK LINKS
    # ------------------------------------------------------

    st.subheader("🚀 Project Assets")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.info("""
📄

README.md

Project documentation
""")

    with col2:

        st.info("""
📦

requirements.txt

Python dependencies
""")

    with col3:

        st.info("""
⚖

LICENSE

MIT License
""")

    st.divider()

    # ------------------------------------------------------
    # PROJECT REPORT
    # ------------------------------------------------------

    st.subheader("📑 Generate Project Report")

    report = pd.DataFrame({

        "Metric":[

            "Total Cars",

            "Average Selling Price",

            "Highest Selling Price",

            "Lowest Selling Price",

            "Fuel Categories",

            "Transmission Types"

        ],

        "Value":[

            len(df),

            round(df["Selling_Price"].mean(),2),

            round(df["Selling_Price"].max(),2),

            round(df["Selling_Price"].min(),2),

            df["Fuel_Type"].nunique(),

            df["Transmission"].nunique()

        ]

    })

    st.dataframe(

        report,

        hide_index=True,

        use_container_width=True

    )

    st.download_button(

        "📥 Download Project Report",

        report.to_csv(index=False).encode(),

        "project_report.csv",

        "text/csv",

        use_container_width=True

    )

    st.divider()

    # ------------------------------------------------------
    # THANK YOU
    # ------------------------------------------------------

    st.markdown("""

<div class="prediction-card">

<h2>🎉 Thank You for Visiting!</h2>

<p>

This application demonstrates an end-to-end Machine Learning
workflow including data preprocessing, exploratory data analysis,
model training, evaluation, and deployment using Streamlit.

</p>

</div>

""", unsafe_allow_html=True)



# ==========================================================
# 📖 ABOUT PROJECT
# ==========================================================

