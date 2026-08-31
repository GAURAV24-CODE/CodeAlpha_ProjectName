"""
📈 EDA Dashboard page for the Car Price Prediction Streamlit app.
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


def render_eda_dashboard():

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

