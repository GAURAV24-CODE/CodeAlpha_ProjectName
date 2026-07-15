# ==========================================================
# 🦠 COVID-19 Analysis
# ==========================================================

import streamlit as st
import plotly.express as px
from utils.load_data import load_covid_data

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------

df = load_covid_data()

st.title("🦠 COVID-19 Impact Analysis")

st.markdown(
    "Analyze the impact of the COVID-19 pandemic on unemployment across India."
)

st.divider()

# ----------------------------------------------------------
# KPI Cards
# ----------------------------------------------------------

avg_unemployment = df["Estimated Unemployment Rate (%)"].mean()

max_state = (
    df.groupby("Region")["Estimated Unemployment Rate (%)"]
    .mean()
    .idxmax()
)

max_rate = (
    df.groupby("Region")["Estimated Unemployment Rate (%)"]
    .mean()
    .max()
)

avg_labour = df["Estimated Labour Participation Rate (%)"].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Average Unemployment", f"{avg_unemployment:.2f}%")
col2.metric("Most Affected State", max_state)
col3.metric("Highest Rate", f"{max_rate:.2f}%")
col4.metric("Average Labour Participation", f"{avg_labour:.2f}%")

st.divider()

# ----------------------------------------------------------
# State Filter
# ----------------------------------------------------------

states = sorted(df["Region"].unique())

selected_states = st.multiselect(
    "Select State(s)",
    states,
    default=states
)

filtered = df[df["Region"].isin(selected_states)]

# ----------------------------------------------------------
# Monthly Trend
# ----------------------------------------------------------

st.subheader("📈 Monthly COVID Unemployment Trend")

trend = (
    filtered.groupby("Date")["Estimated Unemployment Rate (%)"]
    .mean()
    .reset_index()
)

fig = px.line(
    trend,
    x="Date",
    y="Estimated Unemployment Rate (%)",
    markers=True,
    title="Monthly Average Unemployment"
)

fig.update_layout(height=500)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------
# Top 10 Affected States
# ----------------------------------------------------------

st.subheader("🏆 Top 10 Most Affected States")

top10 = (
    filtered.groupby("Region")["Estimated Unemployment Rate (%)"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

fig = px.bar(
    top10,
    x="Region",
    y="Estimated Unemployment Rate (%)",
    color="Estimated Unemployment Rate (%)",
    text_auto=".2f",
    title="Highest Average Unemployment"
)

fig.update_layout(
    xaxis_tickangle=-45,
    height=550
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------
# Employment Comparison
# ----------------------------------------------------------

st.subheader("👨‍💼 Estimated Employment by State")

employment = (
    filtered.groupby("Region")["Estimated Employed"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

fig = px.bar(
    employment,
    x="Region",
    y="Estimated Employed",
    color="Estimated Employed",
    title="Average Estimated Employment"
)

fig.update_layout(
    xaxis_tickangle=-45,
    height=550
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------
# Scatter Plot
# ----------------------------------------------------------

st.subheader("📊 Labour Participation vs Unemployment")

fig = px.scatter(
    filtered,
    x="Estimated Labour Participation Rate (%)",
    y="Estimated Unemployment Rate (%)",
    color="Region",
    hover_name="Region",
    title="Labour Participation vs Unemployment"
)

fig.update_layout(height=600)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------
# Heatmap
# ----------------------------------------------------------

st.subheader("🔥 Correlation Heatmap")

numeric = filtered.select_dtypes(include="number")

corr = numeric.corr(numeric_only=True)

fig = px.imshow(
    corr,
    text_auto=True,
    color_continuous_scale="RdBu_r"
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------
# Data Table
# ----------------------------------------------------------

st.subheader("📋 COVID Dataset")

st.dataframe(
    filtered,
    use_container_width=True,
    hide_index=True
)

# ----------------------------------------------------------
# Download
# ----------------------------------------------------------

csv = filtered.to_csv(index=False).encode("utf-8")

st.download_button(
    "⬇ Download COVID Dataset",
    csv,
    "covid_analysis.csv",
    "text/csv"
)

st.divider()

st.success("COVID-19 Analysis Completed Successfully ✅")