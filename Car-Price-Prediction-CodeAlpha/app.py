# ==========================================================
# 🚗 CAR PRICE PREDICTION USING MACHINE LEARNING
# Author : Gaurav Eknath Kumbhar
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image


# ==========================================================
# ✅ ROBUST PROJECT PATHS
# Works on LOCAL + STREAMLIT CLOUD
# ==========================================================

# Get the folder where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset
DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "car data.csv"
)

# Main ML model
MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "car_price_model.pkl"
)

# CSS
STYLE_PATH = os.path.join(
    BASE_DIR,
    "style.css"
)


# ==========================================================
# 📁 HELPER FUNCTIONS
# ==========================================================

def asset_path(filename):
    """
    Returns the correct absolute path for files
    inside the assets folder.
    """
    return os.path.join(
        BASE_DIR,
        "assets",
        filename
    )


def model_resource_path(filename):
    """
    Returns the correct absolute path for files
    inside the models folder.
    """
    return os.path.join(
        BASE_DIR,
        "models",
        filename
    )


# ==========================================================
# ⚙ PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# 🎨 LOAD CUSTOM CSS
# ==========================================================

def load_css():

    if os.path.exists(STYLE_PATH):

        try:

            with open(
                STYLE_PATH,
                "r",
                encoding="utf-8"
            ) as f:

                css = f.read()

            st.markdown(
                f"<style>{css}</style>",
                unsafe_allow_html=True
            )

        except Exception as e:

            st.warning(
                f"Could not load CSS: {e}"
            )


load_css()


# ==========================================================
# 📊 LOAD DATASET
# ==========================================================

@st.cache_data
def load_data():

    file_path = DATA_PATH

    if not os.path.exists(file_path):

        st.error(
            "❌ Dataset not found!"
        )

        st.info(
            f"""
Expected dataset location:

{file_path}

Make sure your project structure is:

Car-Price-Prediction-CodeAlpha/
│
├── app.py
│
└── data/
    └── car data.csv
"""
        )

        st.stop()

    try:

        return pd.read_csv(file_path)

    except Exception as e:

        st.error(
            f"❌ Error loading dataset: {e}"
        )

        st.stop()


df = load_data()


# ==========================================================
# 🤖 LOAD MACHINE LEARNING MODEL
# ==========================================================

@st.cache_resource
def load_model():

    model_path = MODEL_PATH

    if not os.path.exists(model_path):

        st.error(
            "❌ Model file not found!"
        )

        st.info(
            f"""
Expected model location:

{model_path}

Make sure your project structure is:

Car-Price-Prediction-CodeAlpha/
│
├── app.py
│
└── models/
    └── car_price_model.pkl
"""
        )

        st.stop()

    try:

        return joblib.load(model_path)

    except Exception as e:

        st.error(
            f"❌ Error loading model: {e}"
        )

        st.stop()


model = load_model()


# ==========================================================
# 🔤 OPTIONAL ENCODERS
# ==========================================================

fuel_encoder = None
seller_encoder = None
transmission_encoder = None


try:

    fuel_encoder = joblib.load(
        model_resource_path(
            "fuel_encoder.pkl"
        )
    )

except Exception:
    fuel_encoder = None


try:

    seller_encoder = joblib.load(
        model_resource_path(
            "seller_encoder.pkl"
        )
    )

except Exception:
    seller_encoder = None


try:

    transmission_encoder = joblib.load(
        model_resource_path(
            "transmission_encoder.pkl"
        )
    )

except Exception:
    transmission_encoder = None


# ==========================================================
# 🖼️ SAFE ASSET CHECK
# ==========================================================

def asset_exists(filename):

    return os.path.exists(
        asset_path(filename)
    )


# ==========================================================
# 📂 PROJECT INFORMATION
# ==========================================================

PROJECT_NAME = "Car Price Prediction"

AUTHOR = "Gaurav Eknath Kumbhar"

MODEL_NAME = "Random Forest Regressor"

DATASET_NAME = "CarDekho Used Cars"

ROWS = df.shape[0]

COLS = df.shape[1]


# ==========================================================
# 🧭 SIDEBAR
# ==========================================================

with st.sidebar:

    # Logo
    if asset_exists("car_logo.png"):

        st.image(
            asset_path("car_logo.png"),
            width=130
        )

    st.title(
        "Car Price Prediction"
    )

    st.caption(
        "Machine Learning Project"
    )

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "🚗 Price Prediction",
            "📊 Dataset Explorer",
            "📈 EDA Dashboard",
            "🤖 Model Performance",
            "📥 Downloads",
            "👨‍💻 About",
            "📬 Contact"
        ]
    )

    st.markdown("---")

    st.info(
        """
        **Model**

        Random Forest Regressor

        **Dataset**

        CarDekho Used Car Dataset
        """
    )


# ==========================================================
# 🏠 HOME
# ==========================================================

if page == "🏠 Home":

    st.markdown(
        """
        <div class="hero-container">

        <h1>
        🚗 Car Price Prediction using Machine Learning
        </h1>

        <p>
        Estimate the resale value of used cars using a trained
        Random Forest Regression model.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    # Hero banner
    if asset_exists("home_banner.png"):

        st.image(
            asset_path("home_banner.png"),
            use_container_width=True
        )

    st.write("")

    # ======================================================
    # PROJECT STATISTICS
    # ======================================================

    st.subheader(
        "📊 Project Statistics"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Dataset Rows",
        f"{df.shape[0]}"
    )

    c2.metric(
        "Features",
        f"{df.shape[1] - 1}"
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


# ==========================================================
# 🔑 IMPORTANT:
# EVERY IMAGE MUST USE asset_path()
# EVERY MODEL FILE MUST USE model_resource_path()
# ==========================================================

# Example:

# WRONG:
# st.image("assets/gaurav.png")

# CORRECT:
# st.image(
#     asset_path("gaurav.png")
# )


# Example:

# WRONG:
# joblib.load("models/file.pkl")

# CORRECT:
# joblib.load(
#     model_resource_path("file.pkl")
# )


# Example:

# WRONG:
# open("assets/Gaurav_Kumbhar_Resume.pdf", "rb")

# CORRECT:
# open(
#     asset_path("Gaurav_Kumbhar_Resume.pdf"),
#     "rb"
# )


# ==========================================================
# 📌 FOR YOUR EXISTING PAGES
# ==========================================================
#
# Replace:
#
# "assets/gaurav.png"
#
# with:
#
# asset_path("gaurav.png")
#
#
# Replace:
#
# "assets/hero.png"
#
# with:
#
# asset_path("hero.png")
#
#
# Replace:
#
# "assets/qr_code.png"
#
# with:
#
# asset_path("qr_code.png")
#
#
# Replace:
#
# "assets/prediction_banner.png"
#
# with:
#
# asset_path("prediction_banner.png")
#
#
# Replace:
#
# "assets/home_banner.png"
#
# with:
#
# asset_path("home_banner.png")
#
#
# Replace:
#
# "assets/Gaurav_Kumbhar_Resume.pdf"
#
# with:
#
# asset_path("Gaurav_Kumbhar_Resume.pdf")
#
#
# Replace:
#
# "models/xxxx.pkl"
#
# with:
#
# model_resource_path("xxxx.pkl")
#
# ==========================================================
