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

        with tab1:
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows", df.shape[0])
            c2.metric("Columns", df.shape[1])
            c3.metric("Species", df[species_col].nunique() if species_col else "—")

            st.subheader("Dataset Preview")
            st.dataframe(df, use_container_width=True, height=300)

            st.subheader("Statistical Summary")
            st.dataframe(df.describe(), use_container_width=True)

        with tab2:
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

            if species_col:
                st.subheader("Feature Distribution by Species")
                feature = st.selectbox("Choose a feature", numeric_cols, key="hist_feature")
                hist_fig = px.histogram(
                    df, x=feature, color=species_col, barmode="overlay",
                    opacity=0.7, nbins=25,
                    color_discrete_sequence=["#C2185B", "#F06292", "#F8BBD0"]
                )
                hist_fig.update_layout(height=380)
                st.plotly_chart(hist_fig, use_container_width=True)

                st.subheader("Feature Relationships")
                sc1, sc2 = st.columns(2)
                with sc1:
                    x_axis = st.selectbox("X-axis", numeric_cols, index=0)
                with sc2:
                    y_axis = st.selectbox("Y-axis", numeric_cols, index=min(2, len(numeric_cols) - 1))
                scatter_fig = px.scatter(
                    df, x=x_axis, y=y_axis, color=species_col,
                    color_discrete_sequence=["#C2185B", "#F06292", "#F8BBD0"],
                    hover_data=numeric_cols
                )
                scatter_fig.update_layout(height=420)
                st.plotly_chart(scatter_fig, use_container_width=True)

                st.subheader("All Features at a Glance")
                matrix_fig = px.scatter_matrix(
                    df, dimensions=numeric_cols, color=species_col,
                    color_discrete_sequence=["#C2185B", "#F06292", "#F8BBD0"]
                )
                matrix_fig.update_layout(height=650)
                st.plotly_chart(matrix_fig, use_container_width=True)
            else:
                st.info("No species/class column detected for grouped charts.")

            st.subheader("Correlation Heatmap")
            corr = df[numeric_cols].corr()
            heat_fig = px.imshow(
                corr, text_auto=".2f", color_continuous_scale="RdPu",
                aspect="auto"
            )
            heat_fig.update_layout(height=420)
            st.plotly_chart(heat_fig, use_container_width=True)

        with tab3:
            st.subheader("Column Names")
            st.write(list(df.columns))

            st.subheader("Missing Values")
            st.dataframe(df.isnull().sum().to_frame("Missing Values"))

            st.subheader("Data Types")
            st.dataframe(df.dtypes.astype(str).to_frame("Data Type"))

    except FileNotFoundError:
        st.error("❌ 'data/iris.csv' not found.")

# ==========================================================
# ==========================================================
# ABOUT PAGE
# ==========================================================

elif page == "ℹ️ About":

    st.markdown('<p class="main-header">ℹ️ About This Project</p>', unsafe_allow_html=True)
    st.markdown("---")

    left, right = st.columns([1.4, 1])

    with left:
        st.subheader("🌸 Iris Flower Classification")

        st.write("""
        This project predicts the species of an Iris flower using Machine Learning.
        It is trained on the classic Iris dataset, first introduced by biologist
        Ronald Fisher in 1936, which remains one of the most widely used datasets
        for classification tasks in data science education.
        """)

        st.subheader("🤖 Algorithm")

        st.markdown("""
        **Random Forest Classifier**

        Random Forest is an ensemble learning algorithm that combines multiple
        decision trees to improve prediction accuracy and reduce overfitting.
        """)

        st.subheader("✨ Features")

        st.markdown("""
        - 🌸 Real-time Iris flower prediction
        - 📊 Interactive dataset visualization
        - 📈 Prediction confidence chart
        - 🎨 Modern and responsive Streamlit UI
        - 🤖 Random Forest Machine Learning Model
        - ⚡ Fast predictions
        """)

    with right:

        st.subheader("🛠️ Tech Stack")

        st.markdown("""
        - 🐍 Python
        - 📊 Pandas
        - 🔢 NumPy
        - 🤖 Scikit-learn
        - 🎈 Streamlit
        - 📈 Plotly
        """)

        st.markdown("---")
   
 
    st.markdown("---")

    st.success("⭐ Thank you for visiting this project!")

    st.caption("Built with ❤️ using Streamlit by Gaurav Eknath Kumbhar")



import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

image_path = os.path.join(BASE_DIR, "assets", "gaurav.png")   # or "GAURAV.PNG" if that's the exact filename

if os.path.exists(image_path):
    with open(image_path, "rb") as image_file:
        img = base64.b64encode(image_file.read()).decode()
    image_type = os.path.splitext(image_path)[1][1:]
else:
    st.error(f"Image not found: {image_path}")
    st.stop()
    
    # ------------------------------
    # Premium Hero Section
    # ------------------------------
st.markdown(f"""
    <style>
    
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Poppins', sans-serif;
    }}
    
    @keyframes float {{
        0% {{transform: translateY(0px);}}
        50% {{transform: translateY(-12px);}}
        100% {{transform: translateY(0px);}}
    }}
    
    @keyframes glow {{
        0% {{
            box-shadow:0 0 25px rgba(59,130,246,.35);
        }}
        50% {{
            box-shadow:0 0 55px rgba(96,165,250,.95);
        }}
        100% {{
            box-shadow:0 0 25px rgba(59,130,246,.35);
        }}
    }}
    
    @keyframes gradient {{
        0% {{background-position:0% 50%;}}
        50% {{background-position:100% 50%;}}
        100% {{background-position:0% 50%;}}
    }}
    
    @keyframes fade {{
        from {{
            opacity:0;
            transform:translateY(40px);
        }}
        to {{
            opacity:1;
            transform:translateY(0px);
        }}
    }}
    
    .hero-card{{
        max-width:1800px;
        margin:auto;
        padding:80px 80px;
        border-radius:30px;
        text-align:center;
    
        background:linear-gradient(-45deg,
        #020617,
        #0F172A,
        #111827,
        #1E293B);
    
        background-size:400% 400%;
    
        animation:
            gradient 12s ease infinite,
            fade 1s ease;
    
        border:1px solid rgba(96,165,250,.18);
    
        box-shadow:
            0px 30px 70px rgba(0,0,0,.55);
    }}
    
    .profile{{
        width:180px;
        height:180px;
        border-radius:50%;
        object-fit:cover;
    
        border:6px solid #3B82F6;
    
        animation:
            float 4s ease-in-out infinite,
            glow 3s infinite;
    
        transition:.4s;
    }}
    
    .profile:hover{{
        transform:scale(1.08);
    }}
    
    .name{{
        margin-top:28px;
        margin-bottom:12px;
    
        font-size:48px;
        font-weight:800;
    
        background:linear-gradient(
            90deg,
            #FFFFFF,
            #60A5FA,
            #38BDF8,
            #FFFFFF
        );
    
        background-size:300%;
    
        -webkit-background-clip:text;
        -webkit-text-fill-color:transparent;
    
        animation:gradient 7s linear infinite;
    }}
    
    .role{{
        color:#38BDF8;
        font-size:24px;
        font-weight:600;
    }}
    
    .desc{{
        color:#CBD5E1;
        font-size:18px;
        line-height:1.9;
        max-width:760px;
        margin:auto;
        margin-top:25px;
    }}
    
    .btn{{
        display:inline-block;
    
        margin:12px;
    
        padding:16px 34px;
    
        border-radius:14px;
    
        text-decoration:none;
    
        color:white;
    
        font-weight:700;
    
        font-size:18px;
    
        transition:.35s;
    }}
    
    .linkedin{{
      background:linear-gradient(90deg,#2563EB,#7C3AED);
       
    }}
    
    .github{{
      background:linear-gradient(90deg,#DC2626,#EF4444);
    }}
    
    .btn:hover{{
        transform:translateY(-6px) scale(1.05);
    
        box-shadow:
            0px 15px 35px rgba(37,99,235,.55);
    }}
    
    .skills{{
        display:flex;
        justify-content:center;
        gap:35px;
        flex-wrap:wrap;
        margin-top:40px;
    }}
    
    .skill{{
        padding:15px 22px;
    
        border-radius:14px;
    
        background:rgba(255,255,255,.05);
    
        color:white;
    
        border:1px solid rgba(255,255,255,.08);
    
        font-size:17px;
    
        transition:.3s;
    }}
    
    .skill:hover{{
        background:#2563EB;
        transform:translateY(-5px);
    }}
    
    .mail{{
        margin-top:35px;
        color:#E2E8F0;
        font-size:18px;
    }}
    
    .mail a{{
        color:#93C5FD;
        text-decoration:none;
    }}
    
    </style>
    
    <div class="hero-card">
    
    <img class="profile"
    src="data:image/{image_type};base64,{img}">
    
    <div class="name">
    Gaurav Eknath Kumbhar
    </div>
    
    <div class="role">
    🚀 Data Science • Machine Learning • AI Enthusiast
    </div>
    
    <div class="desc">
    Passionate MCA student focused on building intelligent applications using
    <b>Python</b>,
    <b>SQL</b>,
    <b>Machine Learning</b>,
    <b>Power BI</b>,
    <b>Data Analytics</b>,
    and
    <b>Artificial Intelligence</b>.
    Always learning, always building.
    </div>
    
    <div style="margin-top:35px;">
    
    <a class="btn linkedin"
    href="https://www.linkedin.com/in/gaurav-kumbhar-0b4a39293"
    target="_blank">
    💼 LinkedIn
    </a>
    
    <a class="btn github"
    href="https://github.com/GAURAV24-CODE"
    target="_blank">
    🐙 GitHub
    </a>
    
    </div>
    
    <div class="skills">
    
    <div class="skill">🐍 Python</div>
    
    <div class="skill">📊 Power BI</div>
    
    <div class="skill">🗄 SQL</div>
    
    <div class="skill">🤖 Machine Learning</div>
    
    <div class="skill">📈 Data Analytics</div>
    
    <div class="skill">🧠 AI</div>
    
    </div>
    
    <div class="mail">
    
    📧
    <a href="mailto:kumbhargaurav24@gmail.com">
    kumbhargaurav24@gmail.com
    </a>
    
    </div>
    
    </div>
    
    """, unsafe_allow_html=True)
    
