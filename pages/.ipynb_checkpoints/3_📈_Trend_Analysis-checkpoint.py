# ==========================================================
# 📈 Trend Analysis
# ==========================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.load_data import load_main_data

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------

df = load_main_data()

st.title("📈 Trend Analysis")

st.markdown(
    "Analyze unemployment trends across India over time."
)

st.divider()

# ----------------------------------------------------------
# Sidebar Filters
# ----------------------------------------------------------

st.sidebar.header("📌 Trend Filters")

states = sorted(df["Region"].unique())

selected_states = st.sidebar.multiselect(
    "Select State(s)",
    states,
    default=states
)

filtered_df = df[df["Region"].isin(selected_states)].copy()

# ----------------------------------------------------------
# Monthly Trend
# ----------------------------------------------------------

st.subheader("📅 Monthly Average Unemployment")

monthly = (
    filtered_df
    .groupby("Date")["Estimated Unemployment Rate (%)"]
    .mean()
    .reset_index()
)

fig = px.line(
    monthly,
    x="Date",
    y="Estimated Unemployment Rate (%)",
    markers=True,
    title="Monthly Average Unemployment"
)

fig.update_layout(height=500)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------
# Moving Average
# ----------------------------------------------------------

st.subheader("📊 3-Month Moving Average")

monthly["Moving Average"] = (
    monthly["Estimated Unemployment Rate (%)"]
    .rolling(3)
    .mean()
)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=monthly["Date"],
        y=monthly["Estimated Unemployment Rate (%)"],
        mode="lines+markers",
        name="Actual"
    )
)

fig.add_trace(
    go.Scatter(
        x=monthly["Date"],
        y=monthly["Moving Average"],
        mode="lines",
        name="Moving Average"
    )
)

fig.update_layout(height=500)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------
# Month-wise Seasonality
# ----------------------------------------------------------

st.subheader("🗓 Month-wise Seasonality")

filtered_df["Month"] = filtered_df["Date"].dt.month_name()

month_order = [
    "January","February","March","April",
    "May","June","July","August",
    "September","October","November","December"
]

filtered_df["Month"] = pd.Categorical(
    filtered_df["Month"],
    categories=month_order,
    ordered=True
)

seasonality = (
    filtered_df
    .groupby("Month")["Estimated Unemployment Rate (%)"]
    .mean()
    .reset_index()
)

fig = px.bar(
    seasonality,
    x="Month",
    y="Estimated Unemployment Rate (%)",
    color="Estimated Unemployment Rate (%)",
    title="Average Unemployment by Month"
)

fig.update_layout(height=500)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------
# Box Plot
# ----------------------------------------------------------

st.subheader("📦 Distribution by State")

fig = px.box(
    filtered_df,
    x="Region",
    y="Estimated Unemployment Rate (%)",
    color="Region"
)

fig.update_layout(
    height=700,
    xaxis_tickangle=-45,
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------
# Heatmap
# ----------------------------------------------------------

st.subheader("🔥 Correlation Heatmap")

numeric_df = filtered_df.select_dtypes(include="number")

corr = numeric_df.corr(numeric_only=True)

fig = px.imshow(
    corr,
    text_auto=True,
    color_continuous_scale="RdBu_r",
    title="Correlation Matrix"
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------
# Daily Trend Table
# ----------------------------------------------------------

st.subheader("📋 Trend Data")

st.dataframe(
    monthly,
    use_container_width=True
)

# ----------------------------------------------------------
# Download Trend Data
# ----------------------------------------------------------

csv = monthly.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇ Download Trend Data",
    data=csv,
    file_name="trend_analysis.csv",
    mime="text/csv"
)

st.divider()

st.success("Trend Analysis Completed Successfully ✅")