import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import base64
import os

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Iris Flower Classification",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Global Styling
# -------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.6rem;
        font-weight: 700;
        color: #C2185B;
        margin-bottom: 0rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6b6b6b;
        margin-top: 0rem;
    }
    .stat-card {
        background-color: #FDF2F8;
        border: 1px solid #F3D6E4;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .feature-card {
        background-color: #FAFAFA;
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #eee;
        height: 100%;
    }
    div[data-testid="stMetricValue"] {
        color: #C2185B;
    }
    hr {
        margin: 1.2rem 0;
    }
</style>
""", unsafe_allow_html=True)

FLOWER_INFO = {
    "setosa": {
        "emoji": "🌸",
        "display": "Iris Setosa",
        "fact": "Easily separable species — smallest petals of the three, native to cooler climates."
    },
    "versicolor": {
        "emoji": "🌺",
        "display": "Iris Versicolor",
        "fact": "Also called the Blue Flag Iris, it favors wetlands and marshes."
    },
    "virginica": {
        "emoji": "🌼",
        "display": "Iris Virginica",
        "fact": "The largest of the three species, common in the southeastern United States."
    }
}

# -------------------------------
# Cached Loaders
# -------------------------------
# Cached Loaders
# -------------------------------
@st.cache_resource(show_spinner=False)
def load_model_and_scaler():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(BASE_DIR, "models", "random_forest_model.pkl")
    scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")

    with open(model_path, "rb") as file:
        model = pickle.load(file)

    with open(scaler_path, "rb") as file:
        scaler = pickle.load(file)

    return model, scaler

@st.cache_data(show_spinner=False)
def load_dataset():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(BASE_DIR, "data", "Iris.csv")
    return pd.read_csv(data_path)


try:
    model, scaler = load_model_and_scaler()
    model_loaded = True
except FileNotFoundError as e:
    st.error(f"Missing file: {e}")
    model, scaler = None, None
    model_loaded = False

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.markdown("## 🌸 Iris Flower Classification")
st.sidebar.caption("A Random Forest powered classifier")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🌸 Prediction",
        "📊 Dataset",
        "ℹ️ About"
    ]
)

st.sidebar.markdown("---")
if model_loaded:
    st.sidebar.success("Model loaded ✅")
else:
    st.sidebar.error("Model files not found")
st.sidebar.caption("Built with Streamlit · Scikit-learn · Plotly")

# ==========================================================
# HOME PAGE
# ==========================================================

if page == "🏠 Home":

    st.markdown('<p class="main-header">🌸 Iris Flower Classification</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict Iris species instantly from flower measurements using Machine Learning.</p>', unsafe_allow_html=True)

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.metric("Species", "3")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.metric("Features Used", "4")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.metric("Model", "Random Forest")
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        try:
            df_preview = load_dataset()
            st.metric("Training Samples", df_preview.shape[0])
        except FileNotFoundError:
            st.metric("Training Samples", "150")
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    st.write("")

    left, right = st.columns([1.3, 1])

    with left:
        st.subheader("What this app does")
        st.write("""
        This app takes four simple flower measurements — sepal length, sepal width,
        petal length, and petal width — and predicts which of three Iris species
        the flower belongs to. It's built on the classic Iris dataset, a staple
        benchmark in machine learning.
        """)

        st.subheader("How to use it")
        st.markdown("""
        1. Go to the **🌸 Prediction** page from the sidebar.
        2. Enter or adjust the flower measurements.
        3. Click **Predict Flower** to see the species and confidence score.
        4. Explore the **📊 Dataset** page for charts and statistics.
        """)

        cta1, cta2 = st.columns(2)
        with cta1:
            if st.button("🌸 Try a Prediction", use_container_width=True):
                st.session_state["_go_to_prediction"] = True
        with cta2:
            if st.button("📊 Explore the Dataset", use_container_width=True):
                st.session_state["_go_to_dataset"] = True

        if st.session_state.get("_go_to_prediction"):
            st.info("Use the sidebar and select **🌸 Prediction** to continue.")
        if st.session_state.get("_go_to_dataset"):
            st.info("Use the sidebar and select **📊 Dataset** to continue.")

    with right:
        st.subheader("The Three Species")
        for key, info in FLOWER_INFO.items():
            with st.container(border=True):
                st.markdown(f"**{info['emoji']} {info['display']}**")
                st.caption(info["fact"])

    st.markdown("---")
    st.subheader("Technologies Used")
    tcols = st.columns(5)
    tech = ["Python", "Pandas", "NumPy", "Scikit-learn", "Streamlit"]
    icons = ["🐍", "🐼", "🔢", "🤖", "🎈"]
    for c, t, i in zip(tcols, tech, icons):
        with c:
            st.markdown(f'<div class="feature-card" style="text-align:center;">{i}<br><b>{t}</b></div>', unsafe_allow_html=True)

# ==========================================================
# PREDICTION PAGE
# ==========================================================

elif page == "🌸 Prediction":

    st.markdown('<p class="main-header">🌸 Predict Iris Flower</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Adjust the sliders to match your flower\'s measurements.</p>', unsafe_allow_html=True)
    st.markdown("---")

    if not model_loaded:
        st.error("Model files were not found. Please make sure 'random_forest_model.pkl' and 'scaler.pkl' are present.")
    else:
        input_col, viz_col = st.columns([1, 1.1])

        with input_col:
            st.subheader("Flower Measurements")

            sepal_length = st.slider("Sepal Length (cm)", 0.0, 10.0, 5.1, 0.1)
            sepal_width = st.slider("Sepal Width (cm)", 0.0, 10.0, 3.5, 0.1)
            petal_length = st.slider("Petal Length (cm)", 0.0, 10.0, 1.4, 0.1)
            petal_width = st.slider("Petal Width (cm)", 0.0, 10.0, 0.2, 0.1)

            predict_clicked = st.button("🌸 Predict Flower", type="primary", use_container_width=True)

        with viz_col:
            st.subheader("Your Input, Visualized")
            radar_fig = go.Figure()
            categories = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
            values = [sepal_length, sepal_width, petal_length, petal_width]
            radar_fig.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill='toself',
                name='Input',
                line_color="#C2185B"
            ))
            radar_fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
                showlegend=False,
                height=350,
                margin=dict(l=30, r=30, t=30, b=30)
            )
            st.plotly_chart(radar_fig, use_container_width=True)

        if predict_clicked:
            sample = [[sepal_length, sepal_width, petal_length, petal_width]]
            sample_scaled = scaler.transform(sample)
            prediction = model.predict(sample_scaled)[0]
            pred_key = prediction.lower()
            info = FLOWER_INFO.get(pred_key, {"emoji": "🌷", "display": prediction.title(), "fact": ""})

            st.markdown("---")
            result_col, prob_col = st.columns([1, 1.3])

            with result_col:
                st.success(f"### Predicted Species: {info['emoji']} {info['display']}")
                if info.get("fact"):
                    st.caption(info["fact"])
                st.metric("Model", "Random Forest")

            with prob_col:
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(sample_scaled)[0]
                    classes = [c.title() for c in model.classes_]
                    prob_df = pd.DataFrame({"Species": classes, "Confidence": proba})
                    fig = px.bar(
                        prob_df, x="Species", y="Confidence",
                        color="Species", text_auto=".1%",
                        range_y=[0, 1],
                        color_discrete_sequence=["#C2185B", "#F06292", "#F8BBD0"]
                    )
                    fig.update_layout(showlegend=False, height=320, margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# DATASET PAGE
# ==========================================================
elif page == "📊 Dataset":

    st.markdown('<p class="main-header">📊 Iris Dataset</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Explore the data behind the model.</p>', unsafe_allow_html=True)
    st.markdown("---")

    try:
        df = load_dataset()

        species_col = None
        for candidate in ["species", "Species", "class", "target"]:
            if candidate in df.columns:
                species_col = candidate
                break

        tab1, tab2, tab3 = st.tabs(["🔍 Overview", "📈 Charts", "📌 Data Quality"])

        # ===========================
        # Tab 1
        # ===========================
        with tab1:
            c1, c2, c3 = st.columns(3)

            c1.metric("Rows", df.shape[0])
            c2.metric("Columns", df.shape[1])
            c3.metric("Species", df[species_col].nunique() if species_col else "—")

            st.subheader("Dataset Preview")
            st.dataframe(df, use_container_width=True, height=300)

            st.subheader("Statistical Summary")
            st.dataframe(df.describe(), use_container_width=True)

        # ===========================
        # Tab 2
        # ===========================
        with tab2:

            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

            if species_col:

                st.subheader("Feature Distribution by Species")

                feature = st.selectbox(
                    "Choose a feature",
                    numeric_cols,
                    key="hist_feature"
                )

                hist_fig = px.histogram(
                    df,
                    x=feature,
                    color=species_col,
                    barmode="overlay",
                    opacity=0.7,
                    nbins=25,
                    color_discrete_sequence=["#C2185B", "#F06292", "#F8BBD0"]
                )

                hist_fig.update_layout(height=380)
                st.plotly_chart(hist_fig, use_container_width=True)

                st.subheader("Feature Relationships")

                sc1, sc2 = st.columns(2)

                with sc1:
                    x_axis = st.selectbox("X-axis", numeric_cols)

                with sc2:
                    y_axis = st.selectbox(
                        "Y-axis",
                        numeric_cols,
                        index=min(2, len(numeric_cols)-1)
                    )

                scatter_fig = px.scatter(
                    df,
                    x=x_axis,
                    y=y_axis,
                    color=species_col,
                    hover_data=numeric_cols,
                    color_discrete_sequence=["#C2185B", "#F06292", "#F8BBD0"]
                )

                scatter_fig.update_layout(height=420)
                st.plotly_chart(scatter_fig, use_container_width=True)

                st.subheader("All Features at a Glance")

                matrix_fig = px.scatter_matrix(
                    df,
                    dimensions=numeric_cols,
                    color=species_col,
                    color_discrete_sequence=["#C2185B", "#F06292", "#F8BBD0"]
                )

                matrix_fig.update_layout(height=650)
                st.plotly_chart(matrix_fig, use_container_width=True)

            else:
                st.info("No species/class column detected.")

            st.subheader("Correlation Heatmap")

            corr = df[numeric_cols].corr()

            heat_fig = px.imshow(
                corr,
                text_auto=".2f",
                color_continuous_scale="RdPu",
                aspect="auto"
            )

            heat_fig.update_layout(height=420)

            st.plotly_chart(heat_fig, use_container_width=True)

        # ===========================
        # Tab 3
        # ===========================
        with tab3:

            st.subheader("Column Names")
            st.write(list(df.columns))

            st.subheader("Missing Values")
            st.dataframe(df.isnull().sum().to_frame("Missing Values"))

            st.subheader("Data Types")
            st.dataframe(df.dtypes.astype(str).to_frame("Data Type"))

    except FileNotFoundError as e:
        st.error(f"❌ {e}")

    except Exception as e:
        st.error(f"❌ Unexpected Error: {e}")

# ==========================================================
# ==========================================================

elif page == "ℹ️ About":

    from pathlib import Path

    st.markdown("# ℹ️ About This Project")
    st.markdown("---")

    # ======================================================
    # Top Section
    # ======================================================

    left, right = st.columns([3, 1], gap="large")

    with left:

        st.subheader("🌸 Iris Flower Classification")

        st.write("""
This project predicts the species of an Iris flower using Machine Learning.

The application demonstrates a complete Machine Learning workflow including
data preprocessing, exploratory data analysis, model training, evaluation,
and real-time prediction using a Random Forest Classifier.
""")

        st.subheader("🤖 Machine Learning Model")

        st.success("Random Forest Classifier")

        st.subheader("✨ Key Features")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("""
✅ Real-time Iris Prediction

✅ Dataset Explorer

✅ Interactive Charts

✅ Feature Analysis

✅ Correlation Heatmap
""")

        with c2:
            st.markdown("""
✅ Random Forest Model

✅ High Accuracy

✅ Fast Prediction

✅ Modern Streamlit UI

✅ Responsive Design
""")

    with right:

        image_path = Path(__file__).parent / "assets" / "gaurav.png"

        if image_path.exists():
            st.image(str(image_path), width=230)

        st.markdown("## Gaurav Eknath Kumbhar")

        st.caption("🚀 Data Science • Machine Learning • AI")

        st.markdown("---")

        st.subheader("🛠️ Tech Stack")

        st.markdown("""
🐍 **Python**

📊 **Pandas**

🔢 **NumPy**

🤖 **Scikit-learn**

📈 **Plotly**

🎈 **Streamlit**

🧠 **Machine Learning**
""")

    # ======================================================
    # About Me
    # ======================================================

    st.markdown("---")

    st.subheader("👨‍💻 About Me")

    st.write("""
Hello! I'm **Gaurav Eknath Kumbhar**, an MCA student passionate about
**Data Science, Machine Learning, Artificial Intelligence, Data Analytics,
Python Development, SQL, and Power BI**.

I enjoy building real-world Machine Learning applications,
creating interactive dashboards, and deploying professional
Streamlit projects.

My goal is to become a skilled Data Scientist by continuously
learning, building practical projects, and solving real-world problems.
""")

    st.markdown("### 🌐 Connect With Me")

    col1, col2 = st.columns(2)

    with col1:
        st.link_button(
            "💼 LinkedIn",
            "https://www.linkedin.com/in/gaurav-kumbhar-0b4a39293"
        )

    with col2:
        st.link_button(
            "🐙 GitHub",
            "https://github.com/GAURAV24-CODE"
        )

    st.markdown("---")

    st.success("⭐ Thank you for visiting this project!")

    st.caption("Built with ❤️ using Streamlit by Gaurav Eknath Kumbhar")
# ==========================================================
