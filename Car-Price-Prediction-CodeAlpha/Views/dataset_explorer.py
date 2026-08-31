"""
📊 Dataset Explorer page for the Car Price Prediction Streamlit app.
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


def render_dataset_explorer():

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

