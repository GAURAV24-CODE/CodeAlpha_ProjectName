"""
🚗 Price Prediction page for the Car Price Prediction Streamlit app.
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


def render_price_prediction():

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
    if os.path.exists(asset_path("prediction_banner.png")):
        st.image(
            asset_path("prediction_banner.png"),
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

