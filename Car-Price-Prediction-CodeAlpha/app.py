# ==========================================================
# Car Price Prediction using Machine Learning
# Author : Gaurav Eknath Kumbhar
# Internship : CodeAlpha Data Science
#
# app.py
# Main entry point: page config, sidebar navigation, and
# dispatch to the individual page modules under views/.
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import streamlit as st

from utils import asset_path, load_css

from views.home import render_home
from views.price_prediction import render_price_prediction
from views.dataset_explorer import render_dataset_explorer
from views.eda_dashboard import render_eda_dashboard
from views.model_performance import render_model_performance
from views.downloads import render_downloads
from views.about import render_about
from views.contact import render_contact

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# LOAD CUSTOM CSS
# ==========================================================

load_css()

# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.image(
        asset_path("car_logo.png"),
        width=130
    )

    st.title("Car Price Prediction")

    st.caption("Machine Learning Project")

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
# PAGE ROUTER
# ==========================================================

PAGES = {
    "🏠 Home": render_home,
    "🚗 Price Prediction": render_price_prediction,
    "📊 Dataset Explorer": render_dataset_explorer,
    "📈 EDA Dashboard": render_eda_dashboard,
    "🤖 Model Performance": render_model_performance,
    "📥 Downloads": render_downloads,
    "👨‍💻 About": render_about,
    "📬 Contact": render_contact,
}

PAGES[page]()
