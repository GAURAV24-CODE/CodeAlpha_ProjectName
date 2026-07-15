# ============================================================
# 📊 Unemployment Analysis Dashboard
# CodeAlpha Data Science Internship Project
# Developed by: Gaurav Eknath Kumbhar
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Unemployment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Home",
        "📊 Dashboard",
        "📂 Dataset Explorer",
        "📈 Trend Analysis",
        "🗺️ State Analysis",
        "🦠 COVID Analysis",
        "📌 Insights",
        "ℹ️ About"
    ]
)

# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":

    st.title("📊 Unemployment Analysis Dashboard")

    st.markdown("""
Welcome to the **Unemployment Analysis Dashboard**.

This project was developed as part of the **CodeAlpha Data Science Internship**.

The dashboard analyzes unemployment trends across India using interactive visualizations built with **Python, Plotly, and Streamlit**.

### 🎯 Project Objectives

- Analyze unemployment trends
- Study COVID-19 impact
- Discover seasonal patterns
- Compare state-wise unemployment
- Generate business insights
""")

    st.markdown("---")

    # ========================================================
    # Information Cards
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "📁 Datasets",
            "2"
        )

    with col2:
        st.metric(
            "📊 Charts",
            "20+"
        )

    with col3:
        st.metric(
            "🗺️ States",
            "28+"
        )

    with col4:
        st.metric(
            "🦠 COVID Analysis",
            "Included"
        )

    st.markdown("---")

    # ========================================================
    # Technologies
    # ========================================================

    st.subheader("🛠️ Technologies Used")

    tech1, tech2, tech3, tech4 = st.columns(4)

    tech1.success("🐍 Python")
    tech2.info("📄 Pandas")
    tech3.warning("📊 Plotly")
    tech4.success("🚀 Streamlit")

    st.markdown("---")

    # ========================================================
    # Dashboard Features
    # ========================================================

    st.subheader("✨ Dashboard Features")

    feature1, feature2 = st.columns(2)

    with feature1:

        st.markdown("""
✅ Interactive Dashboard

✅ Dataset Explorer

✅ Trend Analysis

✅ State-wise Analysis

✅ COVID-19 Analysis

✅ Correlation Analysis
""")

    with feature2:

        st.markdown("""
✅ Plotly Interactive Charts

✅ KPI Cards

✅ Business Insights

✅ Download Ready

✅ Recruiter Friendly

✅ Streamlit Based
""")

    st.markdown("---")

    st.success("👈 Use the sidebar to explore the dashboard.")

    st.caption("Developed by **Gaurav Eknath Kumbhar**")

# ============================================================
# PLACEHOLDER PAGES
# ============================================================

elif page == "📊 Dashboard":
    st.title("📊 Dashboard")
    st.info("Coming in the next step...")

elif page == "📂 Dataset Explorer":
    st.title("📂 Dataset Explorer")
    st.info("Coming in the next step...")

elif page == "📈 Trend Analysis":
    st.title("📈 Trend Analysis")
    st.info("Coming in the next step...")

elif page == "🗺️ State Analysis":
    st.title("🗺️ State Analysis")
    st.info("Coming in the next step...")

elif page == "🦠 COVID Analysis":
    st.title("🦠 COVID Analysis")
    st.info("Coming in the next step...")

elif page == "📌 Insights":
    st.title("📌 Insights")
    st.info("Coming in the next step...")

elif page == "ℹ️ About":
    st.title("ℹ️ About")
    st.info("Coming in the next step...")
