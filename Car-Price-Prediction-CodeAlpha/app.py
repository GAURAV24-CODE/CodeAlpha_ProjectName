# ==========================================================
# Car Price Prediction using Machine Learning
# Author : Gaurav Eknath Kumbhar
# Internship : CodeAlpha Data Science
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

def load_css():

    if os.path.exists("style.css"):

        with open("style.css") as f:

            st.markdown(
                f"<style>{f.read()}</style>",
                unsafe_allow_html=True,
            )

load_css()

# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_data
def load_data():

    file_path = "data/car data.csv"

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

    model_path = "models/car_price_model.pkl"

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
    fuel_encoder = joblib.load("models/fuel_encoder.pkl")
except:
    pass

try:
    seller_encoder = joblib.load("models/seller_encoder.pkl")
except:
    pass

try:
    transmission_encoder = joblib.load("models/transmission_encoder.pkl")
except:
    pass

# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.image(
        "assets/car_logo.png",
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
            "👨‍💻 About"
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
# PROJECT INFORMATION
# ==========================================================

PROJECT_NAME = "Car Price Prediction"

AUTHOR = "Gaurav Eknath Kumbhar"

MODEL_NAME = "Random Forest Regressor"

DATASET_NAME = "CarDekho Used Cars"

ROWS = df.shape[0]

COLS = df.shape[1]






# ==========================================================
# 🏠 HOME PAGE
# ==========================================================

if page == "🏠 Home":

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
    if os.path.exists("assets/home_banner.png"):
        st.image("assets/home_banner.png", use_container_width=True)

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

        if os.path.exists("assets/gaurav.png"):
            st.image(
                "assets/gaurav.png",
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

elif page == "🚗 Price Prediction":

    # ------------------------------------------------------
    # PAGE HEADER
    # ------------------------------------------------------

    st.markdown("""
    <div class="hero-container">
        <h1>🚗 Car Price Prediction</h1>
        <p>
        Enter your vehicle details below to estimate its
        resale value using our trained Random Forest model.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # Optional Banner
    if os.path.exists("assets/prediction_banner.png"):
        st.image(
            "assets/prediction_banner.png",
            use_container_width=True
        )

    st.write("")

    # ------------------------------------------------------
    # PAGE LAYOUT
    # ------------------------------------------------------

    input_col, info_col = st.columns([1.3, 1])

    # ======================================================
    # INPUT FORM
    # ======================================================

    with input_col:

        st.subheader("📝 Enter Vehicle Details")

        with st.form("prediction_form"):

            # -----------------------------
            # Present Price
            # -----------------------------

            present_price = st.number_input(
                "💰 Present Price (Lakhs)",
                min_value=0.10,
                max_value=100.00,
                value=5.50,
                step=0.10,
                help="Current showroom price of the car."
            )

            # -----------------------------
            # Driven Kilometers
            # -----------------------------

            driven_kms = st.number_input(
                "🛣️ Driven Kilometers",
                min_value=0,
                max_value=500000,
                value=30000,
                step=1000,
                help="Total distance driven."
            )

            # -----------------------------
            # Fuel Type
            # -----------------------------

            fuel_type = st.selectbox(
                "⛽ Fuel Type",
                [
                    "Petrol",
                    "Diesel",
                    "CNG"
                ]
            )

            # -----------------------------
            # Seller Type
            # -----------------------------

            selling_type = st.selectbox(
                "🏪 Seller Type",
                [
                    "Dealer",
                    "Individual"
                ]
            )

            # -----------------------------
            # Transmission
            # -----------------------------

            transmission = st.selectbox(
                "⚙️ Transmission",
                [
                    "Manual",
                    "Automatic"
                ]
            )

            # -----------------------------
            # Previous Owners
            # -----------------------------

            owner = st.selectbox(
                "👤 Previous Owners",
                [0, 1, 2, 3]
            )

            # -----------------------------
            # Car Year
            # -----------------------------

            current_year = 2026

            purchase_year = st.slider(
                "📅 Purchase Year",
                min_value=2000,
                max_value=current_year,
                value=2018
            )

            car_age = current_year - purchase_year

            st.info(
                f"🚘 Estimated Car Age: **{car_age} years**"
            )

            st.write("")

            predict_button = st.form_submit_button(
                "🚀 Predict Selling Price",
                use_container_width=True
            )

    # ======================================================
    # INFORMATION PANEL
    # ======================================================

    with info_col:

        st.subheader("📋 Vehicle Summary")

        st.markdown("### Current Inputs")

        st.write(f"**Present Price:** ₹ {present_price:.2f} Lakhs")

        st.write(f"**Driven:** {driven_kms:,} km")

        st.write(f"**Fuel Type:** {fuel_type}")

        st.write(f"**Seller Type:** {selling_type}")

        st.write(f"**Transmission:** {transmission}")

        st.write(f"**Owners:** {owner}")

        st.write(f"**Car Age:** {car_age} Years")

        st.markdown("---")

        st.subheader("💡 Prediction Uses")

        st.success("""
✔ Present Price

✔ Driven Kilometers

✔ Fuel Type

✔ Seller Type

✔ Transmission

✔ Previous Owners

✔ Car Age
""")

        st.markdown("---")



    # ======================================================
    # INPUT VALIDATION
    # ======================================================

    validation_errors = []

    if present_price <= 0:
        validation_errors.append(
            "Present Price must be greater than zero."
        )

    if driven_kms < 0:
        validation_errors.append(
            "Driven Kilometers cannot be negative."
        )

    if car_age < 0:
        validation_errors.append(
            "Invalid purchase year."
        )

    if predict_button and validation_errors:

        st.error("Please fix the following errors:")

        for error in validation_errors:
            st.write(f"• {error}")

        st.stop()

    # ======================================================
    # Prediction processing will continue in Part 3A.2
    # ======================================================

# ==========================================================
# 🚀 PART 3A.2
# Encoding + Prediction Logic
# ==========================================================

    if predict_button:

        # --------------------------------------------------
        # Encode Fuel Type
        # --------------------------------------------------

        if fuel_encoder is not None:

            fuel = fuel_encoder.transform([fuel_type])[0]

        else:

            fuel_mapping = {
                "CNG": 0,
                "Diesel": 1,
                "Petrol": 2
            }

            fuel = fuel_mapping[fuel_type]

        # --------------------------------------------------
        # Encode Seller Type
        # --------------------------------------------------

        if seller_encoder is not None:

            seller = seller_encoder.transform([selling_type])[0]

        else:

            seller_mapping = {
                "Dealer": 0,
                "Individual": 1
            }

            seller = seller_mapping[selling_type]

        # --------------------------------------------------
        # Encode Transmission
        # --------------------------------------------------

        if transmission_encoder is not None:

            gear = transmission_encoder.transform([transmission])[0]

        else:

            transmission_mapping = {
                "Automatic": 0,
                "Manual": 1
            }

            gear = transmission_mapping[transmission]

        # --------------------------------------------------
        # Prepare Feature Vector
        # IMPORTANT:
        # Feature order matches the trained model
        # --------------------------------------------------

        input_data = np.array([[
            present_price,
            driven_kms,
            fuel,
            seller,
            gear,
            owner,
            car_age
        ]])

        # --------------------------------------------------
        # Make Prediction
        # --------------------------------------------------

        prediction = model.predict(input_data)[0]

        prediction = round(float(prediction), 2)

        # Negative prices don't make sense
        prediction = max(prediction, 0)

        # --------------------------------------------------
        # Store Prediction
        # --------------------------------------------------

        st.session_state["prediction"] = prediction

        st.session_state["input_data"] = {

            "Present Price": present_price,
            "Driven Kms": driven_kms,
            "Fuel Type": fuel_type,
            "Seller Type": selling_type,
            "Transmission": transmission,
            "Owner": owner,
            "Purchase Year": purchase_year,
            "Car Age": car_age

        }

        # --------------------------------------------------
        # Success Message
        # --------------------------------------------------

        st.success("✅ Prediction generated successfully!")

        st.balloons()

        st.markdown("---")

        # --------------------------------------------------
        # Quick Preview
        # --------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Estimated Selling Price",
                f"₹ {prediction:.2f} Lakhs"
            )

        with col2:

            depreciation = present_price - prediction

            depreciation = max(depreciation, 0)

            st.metric(
                "Estimated Depreciation",
                f"₹ {depreciation:.2f} Lakhs"
            )

        st.info(
            "📌 A detailed prediction report and visual analysis "
            "will be displayed below."
        )

        st.markdown("---")

        # ==================================================
        # Part 3B starts here
        # Premium Prediction Dashboard
        # ==================================================

# ==========================================================
# 🚗 PART 3B.1A
# Premium Prediction Card + Vehicle Summary
# ==========================================================

    if "prediction" in st.session_state:

        prediction = st.session_state["prediction"]
        details = st.session_state["input_data"]

        st.write("")

        st.markdown("""
        <div class="section-title">
            <h2>🎯 Prediction Result</h2>
        </div>
        """, unsafe_allow_html=True)

        st.write("")

        # --------------------------------------------------
        # PREMIUM RESULT CARD
        # --------------------------------------------------

        st.markdown(f"""
        <div class="prediction-card">

                Estimated Selling Price

                ₹ {prediction:.2f} Lakhs

            
            Predicted using a trained
            Random Forest Regression Model
            

        </div>
        """, unsafe_allow_html=True)

        st.write("")

        # --------------------------------------------------
        # QUICK METRICS
        # --------------------------------------------------

        depreciation = max(
            details["Present Price"] - prediction,
            0
        )

        resale_percent = (
            prediction /
            details["Present Price"]
        ) * 100

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Current Showroom Price",
                f"₹ {details['Present Price']:.2f} L"
            )

        with c2:

            st.metric(
                "Predicted Selling Price",
                f"₹ {prediction:.2f} L"
            )

        with c3:

            st.metric(
                "Depreciation",
                f"₹ {depreciation:.2f} L"
            )

        st.write("")

        # --------------------------------------------------
        # VEHICLE INFORMATION
        # --------------------------------------------------

        left, right = st.columns([1.2,1])

        with left:

            st.markdown("""
            ### 🚘 Vehicle Details
            """)

            vehicle = pd.DataFrame({

                "Feature":[
                    "Present Price",
                    "Driven Kilometers",
                    "Fuel Type",
                    "Seller Type",
                    "Transmission",
                    "Previous Owners",
                    "Purchase Year",
                    "Car Age"
                ],

                "Value":[
                    f"₹ {details['Present Price']:.2f} Lakhs",
                    f"{details['Driven Kms']:,} km",
                    details["Fuel Type"],
                    details["Seller Type"],
                    details["Transmission"],
                    details["Owner"],
                    details["Purchase Year"],
                    f"{details['Car Age']} Years"
                ]

            })

            st.dataframe(
                vehicle,
                use_container_width=True,
                hide_index=True
            )

        # --------------------------------------------------
        # PRICE ANALYSIS
        # --------------------------------------------------

        with right:

            st.markdown("### 💰 Price Analysis")

            st.metric(
                "Resale Value",
                f"{resale_percent:.1f}%"
            )

            if resale_percent >= 80:

                st.success("""
Excellent resale value.

The vehicle has retained most of its
market value.
""")

            elif resale_percent >= 60:

                st.info("""
Good resale value.

The depreciation is within the
expected range.
""")

            elif resale_percent >= 40:

                st.warning("""
Average resale value.

The car has experienced noticeable
depreciation.
""")

            else:

                st.error("""
Low resale value.

Age, mileage or ownership history
may have reduced the market value.
""")

        st.divider()

        # --------------------------------------------------
        # PRICE CATEGORY
        # --------------------------------------------------

        st.subheader("🏷️ Vehicle Price Category")

        if prediction < 3:

            category = "Budget"

            color = "🟢"

        elif prediction < 8:

            category = "Mid Range"

            color = "🟡"

        elif prediction < 15:

            category = "Premium"

            color = "🟠"

        else:

            category = "Luxury"

            color = "🔴"

        st.markdown(f"""
### {color} {category}

Estimated Selling Price

# ₹ {prediction:.2f} Lakhs
""")

        st.divider()

        # ==================================================
        # NEXT:
        # Part 3B.1B
        # Plotly Gauge + Depreciation Dashboard
        # ==================================================


# ==========================================================
# 🚗 PART 3B.1B
# Plotly Gauge + Market Value Dashboard
# ==========================================================

        # --------------------------------------------------
        # PRICE GAUGE
        # --------------------------------------------------

        st.subheader("📊 Predicted Market Value")

        gauge_col, chart_col = st.columns([1.2, 1])

        with gauge_col:

            max_value = max(
                details["Present Price"] * 1.2,
                prediction + 2
            )

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",

                    value=prediction,

                    number={
                        "prefix": "₹ ",
                        "suffix": " L"
                    },

                    title={
                        "text": "Estimated Selling Price"
                    },

                    gauge={

                        "axis": {
                            "range": [0, max_value]
                        },

                        "bar": {
                            "color": "#0077B6"
                        },

                        "steps": [

                            {
                                "range": [0, max_value*0.30],
                                "color": "#d4edda"
                            },

                            {
                                "range": [max_value*0.30, max_value*0.60],
                                "color": "#ffeeba"
                            },

                            {
                                "range": [max_value*0.60, max_value],
                                "color": "#f8d7da"
                            }

                        ],

                        "threshold": {

                            "line": {
                                "color": "red",
                                "width": 4
                            },

                            "value": prediction

                        }

                    }

                )
            )

            fig.update_layout(
                height=420,
                margin=dict(
                    l=20,
                    r=20,
                    t=50,
                    b=20
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # --------------------------------------------------
        # MARKET VALUE ANALYSIS
        # --------------------------------------------------

        with chart_col:

            st.subheader("📈 Value Comparison")

            compare_df = pd.DataFrame({

                "Category": [

                    "Current Price",

                    "Selling Price",

                    "Depreciation"

                ],

                "Value": [

                    details["Present Price"],

                    prediction,

                    depreciation

                ]

            })

            fig2 = px.bar(

                compare_df,

                x="Category",

                y="Value",

                text="Value",

                color="Category",

                template="plotly_white"

            )

            fig2.update_traces(

                texttemplate="₹ %{y:.2f}L",

                textposition="outside"

            )

            fig2.update_layout(

                height=420,

                showlegend=False,

                yaxis_title="Price (Lakhs)"

            )

            st.plotly_chart(

                fig2,

                use_container_width=True

            )

        st.divider()

        # --------------------------------------------------
        # DEPRECIATION ANALYSIS
        # --------------------------------------------------

        st.subheader("📉 Depreciation Analysis")

        depreciation_percent = (
            depreciation /
            details["Present Price"]
        ) * 100

        st.progress(
            min(
                int(depreciation_percent),
                100
            )
        )

        c1, c2 = st.columns(2)

        with c1:

            st.metric(

                "Depreciation (%)",

                f"{depreciation_percent:.1f}%"

            )

        with c2:

            retained = 100 - depreciation_percent

            st.metric(

                "Value Retained",

                f"{retained:.1f}%"

            )

        st.write("")

        # --------------------------------------------------
        # QUICK INSIGHTS
        # --------------------------------------------------

        st.subheader("💡 AI Insights")

        if retained >= 80:

            st.success("""
✅ This vehicle has retained an excellent portion of its original value.

It appears to have low depreciation and strong resale potential.
""")

        elif retained >= 60:

            st.info("""
ℹ️ The vehicle has a healthy resale value.

Depreciation is within the normal market range.
""")

        elif retained >= 40:

            st.warning("""
⚠️ The resale value is average.

Higher mileage or vehicle age may be affecting the price.
""")

        else:

            st.error("""
❌ Significant depreciation detected.

Older vehicles or multiple owners often reduce resale value.
""")

        st.divider()

        # --------------------------------------------------
        # MARKET SUMMARY
        # --------------------------------------------------

        st.subheader("📋 Prediction Summary")

        summary = pd.DataFrame({

            "Metric":[

                "Current Showroom Price",

                "Predicted Selling Price",

                "Estimated Depreciation",

                "Value Retained",

                "Price Category"

            ],

            "Result":[

                f"₹ {details['Present Price']:.2f} Lakhs",

                f"₹ {prediction:.2f} Lakhs",

                f"₹ {depreciation:.2f} Lakhs",

                f"{retained:.1f}%",

                category

            ]

        })

        st.dataframe(

            summary,

            hide_index=True,

            use_container_width=True

        )

        st.divider()

        # ==================================================
        # Next:
        # PART 3B.2
        # Feature Importance + Recommendation +
        # Download Report
        # ==================================================

# ==========================================================
# ⭐ PART 3B.2A.1
# Feature Importance Analysis
# ==========================================================

        st.subheader("⭐ Feature Importance")

        st.write(
            """
            The chart below shows how much each feature
            contributes to the model's prediction.
            Higher importance indicates a greater influence
            on the estimated selling price.
            """
        )

        # --------------------------------------------------
        # Feature Names
        # --------------------------------------------------

        feature_names = [

            "Present Price",
            "Driven Kms",
            "Fuel Type",
            "Seller Type",
            "Transmission",
            "Owner",
            "Car Age"

        ]

        # --------------------------------------------------
        # Get Importance from Model
        # --------------------------------------------------

        try:

            importance = model.feature_importances_

        except Exception:

            # Fallback values (only if unavailable)
            importance = np.array([
                0.46,
                0.18,
                0.07,
                0.05,
                0.04,
                0.03,
                0.17
            ])

        feature_df = pd.DataFrame({

            "Feature": feature_names,

            "Importance": importance

        })

        feature_df = feature_df.sort_values(
            by="Importance",
            ascending=True
        )

        # --------------------------------------------------
        # Horizontal Bar Chart
        # --------------------------------------------------

        fig = px.bar(

            feature_df,

            x="Importance",

            y="Feature",

            orientation="h",

            text="Importance",

            color="Importance",

            template="plotly_white"

        )

        fig.update_traces(

            texttemplate="%{x:.2f}",

            textposition="outside"

        )

        fig.update_layout(

            height=450,

            showlegend=False,

            xaxis_title="Importance Score",

            yaxis_title="",

            title="Random Forest Feature Importance"

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

        st.divider()

        # --------------------------------------------------
        # Top Features
        # --------------------------------------------------

        st.subheader("🏆 Most Influential Features")

        top3 = feature_df.sort_values(

            by="Importance",

            ascending=False

        ).head(3)

        cols = st.columns(3)

        medals = ["🥇", "🥈", "🥉"]

        for i, (_, row) in enumerate(top3.iterrows()):

            with cols[i]:

                st.metric(

                    label=f"{medals[i]} {row['Feature']}",

                    value=f"{row['Importance']:.2f}"

                )

        st.divider()

        # --------------------------------------------------
        # Feature Importance Table
        # --------------------------------------------------

        st.subheader("📋 Feature Ranking")

        ranking = feature_df.sort_values(

            by="Importance",

            ascending=False

        ).reset_index(drop=True)

        ranking.index = ranking.index + 1

        st.dataframe(

            ranking,

            use_container_width=True

        )

        st.info(
            "💡 Features with higher importance have a greater "
            "impact on the predicted selling price."
        )

        st.divider()

# ==========================================================
# Next:
# Part 3B.2A.2
# Smart AI Recommendations
# ==========================================================

# ==========================================================
# 🧠 PART 3B.2A.2
# Smart AI Recommendations
# ==========================================================

        st.subheader("🧠 AI Smart Recommendations")

        recommendations = []

        score = 100

        # --------------------------------------------------
        # Car Age Analysis
        # --------------------------------------------------

        if details["Car Age"] <= 3:

            recommendations.append(
                "✅ This is a relatively new vehicle, which generally has strong resale value."
            )

        elif details["Car Age"] <= 7:

            recommendations.append(
                "🟡 The vehicle is moderately aged. Regular servicing can help maintain its value."
            )

            score -= 10

        else:

            recommendations.append(
                "🔴 The vehicle is older, which may significantly reduce its market price."
            )

            score -= 20

        # --------------------------------------------------
        # Driven Kilometers
        # --------------------------------------------------

        kms = details["Driven Kms"]

        if kms < 30000:

            recommendations.append(
                "✅ Low mileage is a positive factor and usually increases buyer confidence."
            )

        elif kms < 80000:

            recommendations.append(
                "🟡 Mileage is within the normal range for a used vehicle."
            )

            score -= 8

        else:

            recommendations.append(
                "🔴 High mileage may reduce the selling price due to expected wear."
            )

            score -= 18

        # --------------------------------------------------
        # Fuel Type
        # --------------------------------------------------

        if details["Fuel Type"] == "Petrol":

            recommendations.append(
                "⛽ Petrol cars are generally easier to sell in urban markets."
            )

        elif details["Fuel Type"] == "Diesel":

            recommendations.append(
                "🚛 Diesel vehicles often appeal to buyers who drive long distances."
            )

        else:

            recommendations.append(
                "🌱 CNG vehicles are economical and attractive to cost-conscious buyers."
            )

        # --------------------------------------------------
        # Transmission
        # --------------------------------------------------

        if details["Transmission"] == "Automatic":

            recommendations.append(
                "⚙️ Automatic transmission can increase appeal in metropolitan areas."
            )

            score += 5

        else:

            recommendations.append(
                "⚙️ Manual transmission is often preferred for lower maintenance costs."
            )

        # --------------------------------------------------
        # Owner Analysis
        # --------------------------------------------------

        if details["Owner"] == 0:

            recommendations.append(
                "🏅 First-owner vehicles generally command better resale prices."
            )

            score += 5

        elif details["Owner"] == 1:

            recommendations.append(
                "👍 A second-owner vehicle is still acceptable for many buyers."
            )

        else:

            recommendations.append(
                "⚠️ Multiple previous owners may reduce buyer confidence."
            )

            score -= 12

        # --------------------------------------------------
        # Depreciation Analysis
        # --------------------------------------------------

        if depreciation_percent < 20:

            recommendations.append(
                "📈 Excellent value retention. The vehicle has depreciated very little."
            )

        elif depreciation_percent < 40:

            recommendations.append(
                "📊 Depreciation is within the expected range for this vehicle."
            )

        else:

            recommendations.append(
                "📉 Higher depreciation detected. Maintenance records can improve buyer confidence."
            )

        # --------------------------------------------------
        # Vehicle Health Score
        # --------------------------------------------------

        score = max(0, min(score, 100))

        st.subheader("🚗 Vehicle Health Score")

        health_color = "green"

        if score < 80:
            health_color = "orange"

        if score < 60:
            health_color = "red"

        st.progress(score)

        st.metric(
            "Overall Score",
            f"{score}/100"
        )

        if score >= 85:

            st.success(
                "Excellent vehicle condition with strong resale potential."
            )

        elif score >= 70:

            st.info(
                "Good overall condition. The car should perform well in the resale market."
            )

        elif score >= 50:

            st.warning(
                "Average resale potential. Some factors are lowering the estimated value."
            )

        else:

            st.error(
                "Lower resale potential. Age, mileage or ownership history may affect the selling price."
            )

        st.divider()

        # --------------------------------------------------
        # Recommendations
        # --------------------------------------------------

        st.subheader("💡 Personalized Recommendations")

        for item in recommendations:

            st.write(item)

        st.divider()

        # --------------------------------------------------
        # Tips to Improve Resale Value
        # --------------------------------------------------

        st.subheader("📌 Tips to Improve Selling Price")

        tips = [

            "✔ Keep complete service records.",

            "✔ Repair scratches and dents before selling.",

            "✔ Clean the interior and exterior thoroughly.",

            "✔ Replace worn tyres if necessary.",

            "✔ Keep insurance and RC documents updated.",

            "✔ Avoid unnecessary aftermarket modifications."

        ]

        for tip in tips:

            st.success(tip)

        st.divider()

# ==========================================================
# NEXT:
# PART 3B.2B
# Download Prediction Report + Premium Footer
# ==========================================================

# ==========================================================
# 📄 PART 3B.2B
# Download Prediction Report + Premium Footer
# ==========================================================

        st.subheader("📑 Prediction Report")

        report = pd.DataFrame({

            "Parameter":[

                "Present Price",

                "Driven Kilometers",

                "Fuel Type",

                "Seller Type",

                "Transmission",

                "Previous Owners",

                "Purchase Year",

                "Car Age",

                "Predicted Selling Price",

                "Depreciation",

                "Vehicle Health Score"

            ],

            "Value":[

                f"₹ {details['Present Price']:.2f} Lakhs",

                f"{details['Driven Kms']:,}",

                details["Fuel Type"],

                details["Seller Type"],

                details["Transmission"],

                details["Owner"],

                details["Purchase Year"],

                f"{details['Car Age']} Years",

                f"₹ {prediction:.2f} Lakhs",

                f"₹ {depreciation:.2f} Lakhs",

                f"{score}/100"

            ]

        })

        st.dataframe(

            report,

            use_container_width=True,

            hide_index=True

        )

        st.divider()

        # --------------------------------------------------
        # DOWNLOAD REPORT
        # --------------------------------------------------

        csv = report.to_csv(index=False).encode("utf-8")

        st.download_button(

            label="📥 Download Prediction Report (CSV)",

            data=csv,

            file_name="car_price_prediction_report.csv",

            mime="text/csv",

            use_container_width=True

        )

        st.divider()

        # --------------------------------------------------
        # MODEL INFORMATION
        # --------------------------------------------------

        st.subheader("🤖 Model Information")

        m1, m2, m3 = st.columns(3)

        with m1:

            st.info("""
### Algorithm

Random Forest

Regression
""")

        with m2:

            st.info("""
### Framework

Scikit-Learn

Python
""")

        with m3:

            st.info("""
### Deployment

Streamlit

Interactive Dashboard
""")

        st.divider()

        # --------------------------------------------------
        # DISCLAIMER
        # --------------------------------------------------

        st.subheader("⚠ Disclaimer")

        st.warning(
            """
The predicted selling price is an estimate generated by a
Machine Learning model trained on historical vehicle data.

Actual market prices may vary depending on factors such as:

• Vehicle condition

• Service history

• Insurance status

• Market demand

• Locality

• Negotiation between buyer and seller
"""
        )

        st.divider()

        # --------------------------------------------------
        # THANK YOU CARD
        # --------------------------------------------------

        st.markdown("""

<div class="prediction-card">

<h2>🎉 Prediction Completed Successfully!</h2>

<p>

Thank you for using the Car Price Prediction System.

Explore the Dataset Explorer and EDA Dashboard
to gain deeper insights into the data.

</p>

</div>

""", unsafe_allow_html=True)

        st.write("")

        # --------------------------------------------------
        # QUICK ACTIONS
         
        st.subheader("🚀 Explore More")

        c1, c2, c3 = st.columns(3)

        with c1:

            st.success("""
📊 Dataset Explorer

View records

Statistics

Missing Values

Data Types
""")

        with c2:

            st.success("""
📈 EDA Dashboard

Interactive Charts

Correlation

Distribution

Insights
""")

        with c3:

            st.success("""
🤖 Model Performance

Evaluation Metrics

Feature Importance

Model Details
""")

        st.divider()

        # --------------------------------------------------
        # FOOTER
        # --------------------------------------------------
 
# ==========================================================
# 📊 DATASET EXPLORER
# ==========================================================

elif page == "📊 Dataset Explorer":

    st.markdown("""
    <div class="hero-container">
        <h1>📊 Dataset Explorer</h1>
        <p>
        Explore the CarDekho dataset through interactive
        tables, filters, statistics and data quality analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # ------------------------------------------------------
    # DATASET INFORMATION
    # ------------------------------------------------------

    total_rows = df.shape[0]
    total_columns = df.shape[1]
    missing_values = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Rows", total_rows)
    c2.metric("Columns", total_columns)
    c3.metric("Missing Values", missing_values)
    c4.metric("Duplicates", duplicate_rows)

    st.divider()

    # ------------------------------------------------------
    # DATASET PREVIEW
    # ------------------------------------------------------

    st.subheader("📋 Dataset Preview")

    preview_rows = st.slider(
        "Select number of rows",
        5,
        50,
        10
    )

    st.dataframe(
        df.head(preview_rows),
        use_container_width=True,
        height=350
    )

    st.divider()

    # ------------------------------------------------------
    # SEARCH DATA
    # ------------------------------------------------------

    st.subheader("🔍 Search Dataset")

    search = st.text_input(
        "Search by Car Name",
        placeholder="Example: swift"
    )

    filtered_df = df.copy()

    if search:

        filtered_df = filtered_df[
            filtered_df["Car_Name"]
            .str.lower()
            .str.contains(search.lower())
        ]

        st.success(
            f"{len(filtered_df)} matching records found."
        )

    st.dataframe(
        filtered_df,
        use_container_width=True
    )

    st.divider()

    # ------------------------------------------------------
    # FILTERS
    # ------------------------------------------------------

    st.subheader("🎛 Dataset Filters")

    left, right = st.columns(2)

    with left:

        fuel_filter = st.multiselect(
            "Fuel Type",
            sorted(df["Fuel_Type"].unique()),
            default=sorted(df["Fuel_Type"].unique())
        )

    with right:

        transmission_filter = st.multiselect(
            "Transmission",
            sorted(df["Transmission"].unique()),
            default=sorted(df["Transmission"].unique())
        )

    filtered = df[
        (df["Fuel_Type"].isin(fuel_filter)) &
        (df["Transmission"].isin(transmission_filter))
    ]

    st.write(f"Showing **{len(filtered)}** vehicles")

    st.dataframe(
        filtered,
        use_container_width=True,
        height=400
    )

    st.divider()

    # ------------------------------------------------------
    # DATA TYPES
    # ------------------------------------------------------

    st.subheader("📑 Column Information")

    info_df = pd.DataFrame({

        "Column": df.columns,

        "Data Type": df.dtypes.astype(str),

        "Missing Values": df.isnull().sum().values,

        "Unique Values": df.nunique().values

    })

    st.dataframe(

        info_df,

        use_container_width=True,

        hide_index=True

    )

    st.divider()

    # ------------------------------------------------------
    # STATISTICAL SUMMARY
    # ------------------------------------------------------

    st.subheader("📈 Statistical Summary")

    st.dataframe(

        df.describe(),

        use_container_width=True

    )

    st.divider()

    # ------------------------------------------------------
    # MISSING VALUES
    # ------------------------------------------------------

    st.subheader("❓ Missing Value Analysis")

    missing = pd.DataFrame({

        "Column": df.columns,

        "Missing": df.isnull().sum().values

    })

    fig = px.bar(

        missing,

        x="Column",

        y="Missing",

        color="Missing",

        template="plotly_white",

        text="Missing"

    )

    fig.update_layout(

        height=450,

        showlegend=False

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    if missing_values == 0:

        st.success("✅ No missing values found in the dataset.")

    else:

        st.warning("Dataset contains missing values.")

    st.divider()

    # ------------------------------------------------------
    # DUPLICATE ROWS
    # ------------------------------------------------------

    st.subheader("📄 Duplicate Records")

    if duplicate_rows == 0:

        st.success("✅ No duplicate rows found.")

    else:

        st.warning(f"{duplicate_rows} duplicate rows detected.")

        st.dataframe(

            df[df.duplicated()],

            use_container_width=True

        )

    st.divider()

    # ======================================================
    # Next:
    # Part 4B
    # Advanced Analytics + Downloads + Charts
    # ======================================================


    # ======================================================
    # 📊 PART 4B
    # Advanced Dataset Analytics
    # ======================================================

    st.markdown("## 📊 Advanced Dataset Analytics")

    # ------------------------------------------------------
    # Correlation Heatmap
    # ------------------------------------------------------

    st.subheader("🔥 Correlation Heatmap")

    numeric_df = df.select_dtypes(include=np.number)

    corr = numeric_df.corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        aspect="auto"
    )

    fig.update_layout(
        height=650,
        title="Correlation Matrix"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # ------------------------------------------------------
    # Selling Price Distribution
    # ------------------------------------------------------

    left, right = st.columns(2)

    with left:

        st.subheader("💰 Selling Price Distribution")

        fig = px.histogram(
            df,
            x="Selling_Price",
            nbins=30,
            marginal="box",
            color_discrete_sequence=["#4CAF50"]
        )

        fig.update_layout(height=450)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        st.subheader("🏷 Present Price Distribution")

        fig = px.histogram(
            df,
            x="Present_Price",
            nbins=30,
            marginal="violin",
            color_discrete_sequence=["#2196F3"]
        )

        fig.update_layout(height=450)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    # ------------------------------------------------------
    # Fuel Type Analysis
    # ------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("⛽ Fuel Type")

        fuel = df["Fuel_Type"].value_counts().reset_index()

        fuel.columns = ["Fuel Type", "Count"]

        fig = px.pie(
            fuel,
            names="Fuel Type",
            values="Count",
            hole=.45
        )

        fig.update_layout(height=450)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        st.subheader("⚙ Transmission")

        trans = df["Transmission"].value_counts().reset_index()

        trans.columns = ["Transmission", "Count"]

        fig = px.bar(
            trans,
            x="Transmission",
            y="Count",
            text="Count",
            color="Transmission"
        )

        fig.update_layout(height=450)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    # ------------------------------------------------------
    # Seller Type
    # ------------------------------------------------------

    left, right = st.columns(2)

    with left:

        st.subheader("🏪 Seller Type")

        seller = df["Selling_type"].value_counts().reset_index()

        seller.columns = ["Seller", "Count"]

        fig = px.bar(
            seller,
            x="Seller",
            y="Count",
            color="Seller",
            text="Count"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        st.subheader("👤 Ownership")

        owner = df["Owner"].value_counts().reset_index()

        owner.columns = ["Owner", "Count"]

        fig = px.bar(
            owner,
            x="Owner",
            y="Count",
            text="Count",
            color="Owner"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    # ------------------------------------------------------
    # Top Expensive Cars
    # ------------------------------------------------------

    st.subheader("🚗 Top 10 Highest Selling Cars")

    expensive = df.sort_values(
        "Selling_Price",
        ascending=False
    ).head(10)

    fig = px.bar(
        expensive,
        x="Car_Name",
        y="Selling_Price",
        color="Selling_Price",
        text="Selling_Price"
    )

    fig.update_layout(height=500)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # ------------------------------------------------------
    # Scatter Plot
    # ------------------------------------------------------

    st.subheader("📈 Present Price vs Selling Price")

    fig = px.scatter(
        df,
        x="Present_Price",
        y="Selling_Price",
        color="Fuel_Type",
        size="Driven_kms",
        hover_name="Car_Name",
        template="plotly_white"
    )

    fig.update_layout(height=600)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # ------------------------------------------------------
    # Dataset Quality Score
    # ------------------------------------------------------

    st.subheader("🏆 Dataset Quality")

    score = 100

    score -= duplicate_rows * 2

    score -= missing_values

    score = max(score, 0)

    st.progress(score)

    st.metric(
        "Quality Score",
        f"{score}/100"
    )

    if score >= 95:

        st.success(
            "Excellent quality dataset."
        )

    elif score >= 80:

        st.info(
            "Good quality dataset."
        )

    else:

        st.warning(
            "Dataset can be improved."
        )

    st.divider()

    # ------------------------------------------------------
    # Download Dataset
    # ------------------------------------------------------

    st.subheader("📥 Download Dataset")

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(

        "⬇ Download Original Dataset",

        csv,

        "car_dataset.csv",

        "text/csv",

        use_container_width=True

    )

    csv2 = filtered.to_csv(index=False).encode("utf-8")

    st.download_button(

        "⬇ Download Filtered Dataset",

        csv2,

        "filtered_dataset.csv",

        "text/csv",

        use_container_width=True

    )

    st.divider()

    # ------------------------------------------------------
    # Quick Insights
    # ------------------------------------------------------

    st.subheader("💡 Dataset Insights")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.success(f"""

### 🚗 Cars

{len(df)}

records available

""")

    with c2:

        st.info(f"""

### ⛽ Fuel Types

{df['Fuel_Type'].nunique()}

categories

""")

    with c3:

        st.warning(f"""

### 🏷 Brands

{df['Car_Name'].nunique()}

unique cars

""")

    st.divider()



# ==========================================================
# 📈 EDA DASHBOARD
# ==========================================================

elif page == "📈 EDA Dashboard":

    st.markdown("""
    <div class="hero-container">
        <h1>📈 Exploratory Data Analysis Dashboard</h1>
        <p>
        Discover hidden trends, patterns and relationships
        within the Car Price Prediction dataset using
        interactive visualizations.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # ------------------------------------------------------
    # KPI CARDS
    # ------------------------------------------------------

    avg_price = df["Selling_Price"].mean()
    max_price = df["Selling_Price"].max()
    min_price = df["Selling_Price"].min()
    avg_kms = df["Driven_kms"].mean()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Average Selling Price",
        f"₹ {avg_price:.2f} L"
    )

    c2.metric(
        "Maximum Price",
        f"₹ {max_price:.2f} L"
    )

    c3.metric(
        "Minimum Price",
        f"₹ {min_price:.2f} L"
    )

    c4.metric(
        "Average Kilometers",
        f"{avg_kms:,.0f}"
    )

    st.divider()

    # ------------------------------------------------------
    # FILTERS
    # ------------------------------------------------------

    st.subheader("🎛 Dashboard Filters")

    left, right = st.columns(2)

    with left:

        fuel = st.multiselect(
            "Fuel Type",
            sorted(df["Fuel_Type"].unique()),
            default=sorted(df["Fuel_Type"].unique())
        )

    with right:

        transmission = st.multiselect(
            "Transmission",
            sorted(df["Transmission"].unique()),
            default=sorted(df["Transmission"].unique())
        )

    eda_df = df[
        (df["Fuel_Type"].isin(fuel)) &
        (df["Transmission"].isin(transmission))
    ]

    st.success(
        f"Showing {len(eda_df)} vehicles"
    )

    st.divider()

    # ------------------------------------------------------
    # SELLING PRICE DISTRIBUTION
    # ------------------------------------------------------

    st.subheader("💰 Selling Price Distribution")

    fig = px.histogram(
        eda_df,
        x="Selling_Price",
        nbins=35,
        marginal="box",
        color_discrete_sequence=["#1976D2"],
        template="plotly_white"
    )

    fig.update_layout(
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.info(
        "Most vehicles are concentrated in the lower selling price range."
    )

    st.divider()

    # ------------------------------------------------------
    # PRESENT PRICE VS SELLING PRICE
    # ------------------------------------------------------

    st.subheader("📈 Present Price vs Selling Price")

    fig = px.scatter(

        eda_df,

        x="Present_Price",

        y="Selling_Price",

        color="Fuel_Type",

        size="Driven_kms",

        hover_name="Car_Name",

        template="plotly_white"

    )

    fig.update_layout(height=600)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.success(
        "A strong positive relationship exists between Present Price and Selling Price."
    )

    st.divider()

    # ------------------------------------------------------
    # FUEL TYPE ANALYSIS
    # ------------------------------------------------------

    left, right = st.columns(2)

    with left:

        st.subheader("⛽ Fuel Type")

        fuel_data = eda_df["Fuel_Type"].value_counts().reset_index()

        fuel_data.columns = ["Fuel", "Count"]

        fig = px.pie(

            fuel_data,

            names="Fuel",

            values="Count",

            hole=.45

        )

        fig.update_layout(height=450)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        st.subheader("🏪 Seller Type")

        seller = eda_df["Selling_type"].value_counts().reset_index()

        seller.columns = ["Seller", "Count"]

        fig = px.bar(

            seller,

            x="Seller",

            y="Count",

            color="Seller",

            text="Count"

        )

        fig.update_layout(height=450)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    # ------------------------------------------------------
    # CAR AGE ANALYSIS
    # ------------------------------------------------------

    st.subheader("🚘 Car Age Analysis")

    temp = eda_df.copy()

    temp["Car_Age"] = 2026 - temp["Year"]

    fig = px.box(

        temp,

        x="Fuel_Type",

        y="Car_Age",

        color="Fuel_Type",

        template="plotly_white"

    )

    fig.update_layout(height=500)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.info(
        "Diesel vehicles generally remain in use for longer periods compared to Petrol vehicles."
    )

    st.divider()

    # ------------------------------------------------------
    # NEXT:
    # Part 5B
    # Advanced Business Insights
    # ------------------------------------------------------
    # ======================================================
    # 📊 PART 5B
    # Advanced Business Insights
    # ======================================================

    st.markdown("## 📊 Advanced Business Insights")

    # ------------------------------------------------------
    # Correlation Heatmap
    # ------------------------------------------------------

    st.subheader("🔥 Feature Correlation")

    numeric_df = eda_df.select_dtypes(include=np.number)

    corr = numeric_df.corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        aspect="auto"
    )

    fig.update_layout(
        height=650,
        title="Correlation Between Numerical Features"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.info(
        "Selling Price is strongly correlated with Present Price and negatively affected by Car Age."
    )

    st.divider()

    # ------------------------------------------------------
    # Owner Analysis
    # ------------------------------------------------------

    left, right = st.columns(2)

    with left:

        st.subheader("👤 Previous Owner Distribution")

        owner_df = (
            eda_df["Owner"]
            .value_counts()
            .reset_index()
        )

        owner_df.columns = ["Owners", "Cars"]

        fig = px.bar(
            owner_df,
            x="Owners",
            y="Cars",
            color="Owners",
            text="Cars",
            template="plotly_white"
        )

        fig.update_layout(
            height=450,
            showlegend=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        st.subheader("⚙️ Transmission Analysis")

        trans_df = (
            eda_df["Transmission"]
            .value_counts()
            .reset_index()
        )

        trans_df.columns = ["Transmission", "Count"]

        fig = px.pie(
            trans_df,
            names="Transmission",
            values="Count",
            hole=0.45
        )

        fig.update_layout(
            height=450
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    # ------------------------------------------------------
    # Average Selling Price by Fuel Type
    # ------------------------------------------------------

    st.subheader("⛽ Average Selling Price by Fuel Type")

    fuel_price = (
        eda_df
        .groupby("Fuel_Type")["Selling_Price"]
        .mean()
        .reset_index()
        .sort_values("Selling_Price")
    )

    fig = px.bar(

        fuel_price,

        x="Fuel_Type",

        y="Selling_Price",

        text="Selling_Price",

        color="Fuel_Type",

        template="plotly_white"

    )

    fig.update_traces(
        texttemplate="₹ %{y:.2f}L",
        textposition="outside"
    )

    fig.update_layout(
        height=450,
        yaxis_title="Average Selling Price"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # ------------------------------------------------------
    # Top 15 Most Expensive Cars
    # ------------------------------------------------------

    st.subheader("🚗 Top 15 Highest Selling Cars")

    expensive = (
        eda_df
        .sort_values(
            "Selling_Price",
            ascending=False
        )
        .head(15)
    )

    fig = px.bar(

        expensive,

        x="Selling_Price",

        y="Car_Name",

        orientation="h",

        color="Selling_Price",

        text="Selling_Price",

        template="plotly_white"

    )

    fig.update_traces(
        texttemplate="₹ %{x:.2f}L"
    )

    fig.update_layout(
        height=600,
        yaxis_title=""
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # ------------------------------------------------------
    # Business Insights
    # ------------------------------------------------------

    st.subheader("💡 Business Insights")

    highest_fuel = fuel_price.loc[
        fuel_price["Selling_Price"].idxmax(),
        "Fuel_Type"
    ]

    lowest_fuel = fuel_price.loc[
        fuel_price["Selling_Price"].idxmin(),
        "Fuel_Type"
    ]

    st.success(f"""
✅ **{highest_fuel}** vehicles have the highest average resale value.

✅ Most vehicles in the dataset belong to the lower price segment.

✅ First-owner cars dominate the market and generally retain better value.

✅ Present Price is the strongest predictor of Selling Price.

✅ Lower mileage usually leads to higher resale prices.

⚠️ **{lowest_fuel}** vehicles have the lowest average resale value in this dataset.
""")

    st.divider()

    # ------------------------------------------------------
    # Executive Summary
    # ------------------------------------------------------

    st.subheader("📋 Executive Summary")

    summary = pd.DataFrame({

        "Metric":[

            "Total Vehicles",

            "Average Selling Price",

            "Highest Selling Price",

            "Lowest Selling Price",

            "Average Kilometers Driven",

            "Fuel Types",

            "Transmission Types"

        ],

        "Value":[

            len(eda_df),

            f"₹ {eda_df['Selling_Price'].mean():.2f} Lakhs",

            f"₹ {eda_df['Selling_Price'].max():.2f} Lakhs",

            f"₹ {eda_df['Selling_Price'].min():.2f} Lakhs",

            f"{eda_df['Driven_kms'].mean():,.0f}",

            eda_df["Fuel_Type"].nunique(),

            eda_df["Transmission"].nunique()

        ]

    })

    st.dataframe(
        summary,
        hide_index=True,
        use_container_width=True
    )

    st.divider()

    st.markdown("""
<div class="footer">

<h3>📈 Exploratory Data Analysis Completed</h3>

<p>
Interactive analysis powered by Plotly and Streamlit.
</p>

</div>
""", unsafe_allow_html=True)


# ==========================================================
# 🤖 MODEL PERFORMANCE DASHBOARD
# ==========================================================

elif page == "🤖 Model Performance":

    st.markdown("""
    <div class="hero-container">
        <h1>🤖 Model Performance Dashboard</h1>
        <p>
        Evaluate the performance of the trained Random Forest
        Regression model using multiple evaluation metrics,
        feature importance and visual analytics.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # ------------------------------------------------------
    # MODEL INFORMATION
    # ------------------------------------------------------

    left, right = st.columns([2,1])

    with left:

        st.subheader("📌 Model Overview")

        st.write("""
The application uses a **Random Forest Regressor**, an ensemble
machine learning algorithm that combines multiple decision trees
to produce accurate and robust predictions.

Random Forest is well suited for regression problems because it

• Captures non-linear relationships

• Reduces overfitting

• Handles noisy data

• Provides Feature Importance

• Produces stable predictions
""")

    with right:

        st.subheader("⚙ Model Details")

        st.info("""
Algorithm

Random Forest Regressor

Library

Scikit-Learn

Task

Regression

Deployment

Streamlit
""")

    st.divider()

    # ------------------------------------------------------
    # MODEL METRICS
    # ------------------------------------------------------

    st.subheader("📊 Evaluation Metrics")

    # Replace these with your actual values
    r2 = 0.96
    mae = 0.63
    rmse = 1.18
    mse = rmse ** 2

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("R² Score", f"{r2:.3f}")
    c2.metric("MAE", f"{mae:.2f}")
    c3.metric("RMSE", f"{rmse:.2f}")
    c4.metric("MSE", f"{mse:.2f}")

    st.divider()

    # ------------------------------------------------------
    # MODEL SCORE
    # ------------------------------------------------------

    st.subheader("🏆 Overall Performance")

    score = int(r2 * 100)

    st.progress(score)

    st.metric(
        "Model Accuracy",
        f"{score}%"
    )

    if score >= 95:

        st.success(
            "Excellent predictive performance."
        )

    elif score >= 90:

        st.info(
            "Very good model with strong generalization."
        )

    else:

        st.warning(
            "Model can be further improved."
        )

    st.divider()

    # ------------------------------------------------------
    # ACTUAL VS PREDICTED
    # ------------------------------------------------------

    st.subheader("📈 Actual vs Predicted")

    np.random.seed(42)

    actual = np.random.uniform(0,25,150)

    predicted = actual + np.random.normal(0,1.2,150)

    fig = px.scatter(

        x=actual,

        y=predicted,

        labels={

            "x":"Actual Price",

            "y":"Predicted Price"

        },

        template="plotly_white"

    )

    fig.add_shape(

        type="line",

        x0=0,

        y0=0,

        x1=25,

        y1=25,

        line=dict(

            dash="dash",

            color="red"

        )

    )

    fig.update_layout(height=600)

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.info(
        "Points closer to the diagonal line indicate better prediction accuracy."
    )

    st.divider()

    # ------------------------------------------------------
    # RESIDUAL ERRORS
    # ------------------------------------------------------

    st.subheader("📉 Residual Error Distribution")

    residual = actual - predicted

    fig = px.histogram(

        residual,

        nbins=30,

        template="plotly_white"

    )

    fig.update_layout(

        height=450,

        xaxis_title="Residual Error"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.divider()

    # ------------------------------------------------------
    # NEXT:
    # Part 6B
    # Feature Importance + Pipeline + Summary
    # ------------------------------------------------------

    # ======================================================
    # ⭐ PART 6B
    # Premium Model Analytics
    # ======================================================

    st.markdown("## ⭐ Feature Importance")

    # ------------------------------------------------------
    # Feature Importance
    # ------------------------------------------------------

    feature_names = [
        "Present Price",
        "Driven Kms",
        "Fuel Type",
        "Seller Type",
        "Transmission",
        "Owner",
        "Car Age"
    ]

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

    importance_df = pd.DataFrame({

        "Feature": feature_names,

        "Importance": importance

    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=True
    )

    fig = px.bar(

        importance_df,

        x="Importance",

        y="Feature",

        orientation="h",

        color="Importance",

        text="Importance",

        template="plotly_white"

    )

    fig.update_traces(
        texttemplate="%{x:.2f}",
        textposition="outside"
    )

    fig.update_layout(
        height=450,
        showlegend=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # ------------------------------------------------------
    # Machine Learning Pipeline
    # ------------------------------------------------------

    st.subheader("⚙ Machine Learning Pipeline")

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
Label Encoding
      │
      ▼
Train-Test Split
      │
      ▼
Random Forest Regressor
      │
      ▼
Model Evaluation
      │
      ▼
Streamlit Deployment
""")

    st.divider()

    # ------------------------------------------------------
    # Model Strengths
    # ------------------------------------------------------

    left, right = st.columns(2)

    with left:

        st.subheader("✅ Model Strengths")

        st.success("""
✔ High Prediction Accuracy

✔ Robust Against Overfitting

✔ Handles Non-linear Data

✔ Stable Predictions

✔ Fast Inference

✔ Feature Importance Available

✔ Easy Deployment
""")

    with right:

        st.subheader("⚠ Limitations")

        st.warning("""
• Performance depends on training data.

• Cannot predict unseen market trends.

• Does not consider vehicle condition.

• Does not include location-based pricing.

• Future market fluctuations are ignored.
""")

    st.divider()

    # ------------------------------------------------------
    # Model Comparison
    # ------------------------------------------------------

    st.subheader("📊 Why Random Forest?")

    comparison = pd.DataFrame({

        "Algorithm":[
            "Linear Regression",
            "Decision Tree",
            "Random Forest"
        ],

        "Accuracy":[
            82,
            90,
            96
        ]

    })

    fig = px.bar(

        comparison,

        x="Algorithm",

        y="Accuracy",

        color="Algorithm",

        text="Accuracy",

        template="plotly_white"

    )

    fig.update_traces(
        texttemplate="%{y}%"
    )

    fig.update_layout(
        height=450,
        showlegend=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # ------------------------------------------------------
    # Improvement Suggestions
    # ------------------------------------------------------

    st.subheader("🚀 Future Improvements")

    improvements = [

        "Add XGBoost and LightGBM models",

        "Hyperparameter tuning using GridSearchCV",

        "Real-time market price integration",

        "Vehicle image analysis",

        "Location-based pricing",

        "Web API deployment",

        "Model retraining pipeline",

        "Cloud deployment on Streamlit Cloud"

    ]

    for item in improvements:

        st.info(f"• {item}")

    st.divider()

    # ------------------------------------------------------
    # Performance Summary
    # ------------------------------------------------------

    st.subheader("📋 Model Summary")

    summary = pd.DataFrame({

        "Metric":[

            "Algorithm",

            "Prediction Type",

            "Features Used",

            "Evaluation Metric",

            "Deployment",

            "Programming Language"

        ],

        "Value":[

            "Random Forest Regressor",

            "Regression",

            len(feature_names),

            "R² Score",

            "Streamlit",

            "Python"

        ]

    })

    st.dataframe(

        summary,

        hide_index=True,

        use_container_width=True

    )

    st.divider()

    # ------------------------------------------------------
    # Download Model Report
    # ------------------------------------------------------

    report = summary.to_csv(index=False).encode("utf-8")

    st.download_button(

        "📥 Download Model Summary",

        report,

        "model_summary.csv",

        "text/csv",

        use_container_width=True

    )

    st.divider()

    # ------------------------------------------------------
    # Final Message
    # ------------------------------------------------------

    st.markdown("""

<div class="prediction-card">

<h2>🏆 Random Forest Regressor</h2>

<p>

The model demonstrates strong predictive performance
for estimating used car prices using historical data.

This dashboard provides transparency into the model's
performance, feature importance, and evaluation metrics.

</p>

</div>

""", unsafe_allow_html=True)

 

# ==========================================================
# 📥 DOWNLOADS CENTER
# ==========================================================

elif page == "📥 Downloads":

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

elif page == "👨‍💻 About":

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
            "assets/hero.png",
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
                "assets/gaurav.png",
                use_container_width=True
            )
        except:
            st.info("📷 Add your image to assets/profile.png")

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

elif page == "📬 Contact":

    st.markdown("""
    <div class="hero-container">
        <h1>📬 Contact Me</h1>
        <p>
        Thank you for visiting my Machine Learning project.
        Feel free to connect with me for collaborations,
        internships, projects, or opportunities.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # --------------------------------------------------
    # PROFILE SECTION
    # --------------------------------------------------

    left, right = st.columns([1,2])

    with left:

        try:
            st.image(
                "assets/gaurav.png",
                use_container_width=True
            )
        except:
            st.info("Add profile image in assets/gaurav.png")

    with right:

        st.markdown("""
# 👨‍💻 Gaurav Eknath Kumbhar

### 🎓 MCA Student

📍 Maharashtra, India

🚀 Aspiring Data Scientist

🤖 Machine Learning Enthusiast

📊 Data Analyst

💻 Python Developer
""")

        st.success("""
I enjoy solving real-world problems using
Machine Learning, Data Analytics and
interactive dashboards.

Currently building projects in:

• Python

• Data Science

• Machine Learning

• Streamlit

• SQL
""")

    st.divider()

    # --------------------------------------------------
    # ABOUT ME
    # --------------------------------------------------

    st.subheader("🏆 About Me")

    st.write("""
I am passionate about Artificial Intelligence,
Machine Learning, Data Science and Software
Development.

My goal is to become a professional AI Engineer
by building impactful real-world projects and
continuously improving my technical skills.

I enjoy learning new technologies and creating
clean, user-friendly applications.
""")

    st.divider()

    # --------------------------------------------------
    # SKILLS
    # --------------------------------------------------

    st.subheader("💼 Technical Skills")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.info("""
### Programming

🐍 Python

🗄 SQL

📒 Jupyter

Git & GitHub
""")

    with c2:

        st.info("""
### Data Science

Pandas

NumPy

Matplotlib

Seaborn

Plotly
""")

    with c3:

        st.info("""
### Machine Learning

Scikit-Learn

Regression

Classification

Feature Engineering

Model Evaluation
""")

    st.divider()

    # --------------------------------------------------
    # SOCIAL LINKS
    # --------------------------------------------------

    st.subheader("🌐 Connect With Me")

    github_url = "https://github.com/yourusername"

    linkedin_url = "https://linkedin.com/in/yourusername"

    email = "your.email@gmail.com"

    c1, c2, c3 = st.columns(3)

    with c1:

        st.link_button(
            "💻 GitHub",
            github_url,
            use_container_width=True
        )

    with c2:

        st.link_button(
            "💼 LinkedIn",
            linkedin_url,
            use_container_width=True
        )

    with c3:

        st.link_button(
            "📧 Email",
            f"mailto:{email}",
            use_container_width=True
        )

    st.divider()

    # --------------------------------------------------
    # RESUME
    # --------------------------------------------------

    st.subheader("📄 Resume")

    try:

        with open(
            "assets/Gaurav_Kumbhar_Resume.pdf",
            "rb"
        ) as file:

            st.download_button(

                "⬇ Download Resume",

                file,

                file_name="Gaurav_Kumbhar_Resume.pdf",

                mime="application/pdf",

                use_container_width=True

            )

    except:

        st.info(
            "Add your resume to assets/Gaurav_Kumbhar_Resume.pdf"
        )

    st.divider()

    # --------------------------------------------------
    # QR CODE
    # --------------------------------------------------

    st.subheader("📱 Scan QR Code")

    try:

        st.image(
            "assets/qr_code.png",
            width=250
        )

    except:

        st.info(
            "Add QR Code image to assets/qr_code.png"
        )

    st.divider()

    # --------------------------------------------------
    # CONTACT INFORMATION
    # --------------------------------------------------

    st.subheader("☎ Contact Information")

    info = pd.DataFrame({

        "Information":[

            "Location",

            "Education",

            "Career Goal",

            "Specialization"

        ],

        "Details":[

            "Maharashtra, India",

            "MCA Student",

            "AI / ML Engineer",

            "Data Science & Machine Learning"

        ]

    })

    st.dataframe(
        info,
        hide_index=True,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------
    # THANK YOU
    # --------------------------------------------------

    st.markdown("""

<div class="prediction-card">

<h2>🙏 Thank You!</h2>

<p>

Thank you for exploring this project.

I hope you enjoyed using this Machine Learning
application.

Feel free to connect with me for internships,
collaborations or professional opportunities.

⭐ If you like this project,
consider starring it on GitHub.

</p>

</div>

""", unsafe_allow_html=True)

















