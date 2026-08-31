# ==========================================================
# Car Price Prediction using Machine Learning
# Author : Gaurav Eknath Kumbhar
# Internship : CodeAlpha Data Science
#
# utils.py
# Shared paths, data/model loaders, and constants used by
# every page in the app.
# ==========================================================

import os
import joblib
import pandas as pd
import streamlit as st

# ==========================================================
# ROBUST PROJECT PATHS (LOCAL + STREAMLIT CLOUD)
# ==========================================================
# Always resolve files relative to this file's folder.
# This prevents "Dataset not found" errors when Streamlit Cloud
# runs the app from the repository root.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "car data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "car_price_model.pkl")
STYLE_PATH = os.path.join(BASE_DIR, "style.css")


def asset_path(filename):
    return os.path.join(BASE_DIR, "assets", filename)


def model_resource_path(filename):
    return os.path.join(BASE_DIR, "models", filename)


# ==========================================================
# LOAD CUSTOM CSS
# ==========================================================

def load_css():

    if os.path.exists(STYLE_PATH):

        with open(STYLE_PATH, encoding="utf-8") as f:

            st.markdown(
                f"<style>{f.read()}</style>",
                unsafe_allow_html=True,
            )


# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_data
def load_data():

    file_path = DATA_PATH

    if os.path.exists(file_path):

        return pd.read_csv(file_path)

    st.error("Dataset -----not found!")

    st.stop()


df = load_data()

# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_model():

    model_path = MODEL_PATH

    if os.path.exists(model_path):

        return joblib.load(model_path)

    st.error("Model file not found!")

    st.stop()


model = load_model()

# ==========================================================
# OPTIONAL ENCODERS
# ==========================================================

fuel_encoder = None
seller_encoder = None
transmission_encoder = None

try:
    fuel_encoder = joblib.load(model_resource_path("fuel_encoder.pkl"))
except Exception:
    pass

try:
    seller_encoder = joblib.load(model_resource_path("seller_encoder.pkl"))
except Exception:
    pass

try:
    transmission_encoder = joblib.load(model_resource_path("transmission_encoder.pkl"))
except Exception:
    pass

# ==========================================================
# PROJECT INFORMATION
# ==========================================================

PROJECT_NAME = "Car Price Prediction"

AUTHOR = "Gaurav Eknath Kumbhar"

MODEL_NAME = "Random Forest Regressor"

DATASET_NAME = "CarDekho Used Cars"

ROWS = df.shape[0]

COLS = df.shape[1]
