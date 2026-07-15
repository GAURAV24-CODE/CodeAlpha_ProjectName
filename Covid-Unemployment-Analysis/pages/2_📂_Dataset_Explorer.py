# ==========================================================
# 📂 Dataset Explorer
# ==========================================================

import streamlit as st
from utils.load_data import (
    load_main_data,
    get_dataset_summary
)

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------

df = load_main_data()

st.title("📂 Dataset Explorer")

st.markdown(
    "Explore the unemployment dataset with summary statistics and interactive filtering."
)

st.divider()

# ----------------------------------------------------------
# Dataset Summary
# ----------------------------------------------------------

summary = get_dataset_summary(df)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Rows", summary["Rows"])
col2.metric("Columns", summary["Columns"])
col3.metric("Missing Values", summary["Missing Values"])
col4.metric("Duplicate Rows", summary["Duplicate Rows"])

st.divider()

# ----------------------------------------------------------
# Dataset Preview
# ----------------------------------------------------------

st.subheader("📄 Dataset Preview")

rows = st.slider(
    "Select number of rows",
    min_value=5,
    max_value=100,
    value=10
)

st.dataframe(
    df.head(rows),
    use_container_width=True
)

# ----------------------------------------------------------
# Column Information
# ----------------------------------------------------------

st.subheader("📋 Column Information")

column_info = {
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str),
    "Missing Values": df.isnull().sum().values,
    "Unique Values": df.nunique().values
}

st.dataframe(
    column_info,
    use_container_width=True
)

# ----------------------------------------------------------
# Descriptive Statistics
# ----------------------------------------------------------

st.subheader("📊 Statistical Summary")

st.dataframe(
    df.describe(),
    use_container_width=True
)

# ----------------------------------------------------------
# Missing Values
# ----------------------------------------------------------

st.subheader("❗ Missing Values")

missing = df.isnull().sum()

st.dataframe(
    missing.rename("Missing Values"),
    use_container_width=True
)

# ----------------------------------------------------------
# Column Explorer
# ----------------------------------------------------------

st.subheader("🔍 Explore Individual Column")

selected_column = st.selectbox(
    "Choose a Column",
    df.columns
)

st.write(df[selected_column])

# ----------------------------------------------------------
# Search Dataset
# ----------------------------------------------------------

st.subheader("🔎 Search Dataset")

keyword = st.text_input(
    "Enter a keyword"
)

if keyword:

    result = df.astype(str).apply(
        lambda x: x.str.contains(
            keyword,
            case=False,
            na=False
        )
    ).any(axis=1)

    st.dataframe(
        df[result],
        use_container_width=True
    )

# ----------------------------------------------------------
# Download Dataset
# ----------------------------------------------------------

st.subheader("⬇ Download Dataset")

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download CSV",
    data=csv,
    file_name="Unemployment_Data.csv",
    mime="text/csv"
)

# ----------------------------------------------------------
# Footer
# ----------------------------------------------------------

st.divider()

st.success("Dataset Explorer Loaded Successfully ✅")