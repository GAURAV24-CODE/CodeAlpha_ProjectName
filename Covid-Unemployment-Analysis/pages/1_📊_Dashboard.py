# ==========================================================
# 📊 Dashboard
# ==========================================================

import streamlit as st
import plotly.express as px
from utils.load_data import load_main_data

# ----------------------------------------------------------
# Load Data
# ----------------------------------------------------------

df = load_main_data()

st.title("📊 Unemployment Dashboard")

st.markdown(
    "Interactive analysis of unemployment trends across India."
)

st.divider()

# ----------------------------------------------------------
# Sidebar Filters
# ----------------------------------------------------------

st.sidebar.header("🔎 Filters")

states = sorted(df["Region"].unique())

selected_states = st.sidebar.multiselect(
    "Select State(s)",
    options=states,
    default=states
)

filtered_df = df[df["Region"].isin(selected_states)]

# ----------------------------------------------------------
# KPI Cards
# ----------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Records",
        len(filtered_df)
    )

with col2:
    st.metric(
        "States",
        filtered_df["Region"].nunique()
    )

with col3:
    st.metric(
        "Average Unemployment (%)",
        f"{filtered_df['Estimated Unemployment Rate (%)'].mean():.2f}"
    )

with col4:
    st.metric(
        "Average Labour Participation (%)",
        f"{filtered_df['Estimated Labour Participation Rate (%)'].mean():.2f}"
    )

st.divider()

# ----------------------------------------------------------
# Monthly Trend
# ----------------------------------------------------------

st.subheader("📈 Monthly Unemployment Trend")

trend = (
    filtered_df
    .groupby("Date")["Estimated Unemployment Rate (%)"]
    .mean()
    .reset_index()
)

fig = px.line(
    trend,
    x="Date",
    y="Estimated Unemployment Rate (%)",
    markers=True,
    title="Average Unemployment Rate Over Time"
)

fig.update_layout(height=500)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------
# State-wise Average
# ----------------------------------------------------------

st.subheader("🗺️ State-wise Average Unemployment")

state_avg = (
    filtered_df
    .groupby("Region")["Estimated Unemployment Rate (%)"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

fig = px.bar(
    state_avg,
    x="Region",
    y="Estimated Unemployment Rate (%)",
    color="Estimated Unemployment Rate (%)",
    title="Average Unemployment by State"
)

fig.update_layout(
    xaxis_title="State",
    yaxis_title="Unemployment (%)",
    xaxis_tickangle=-45,
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------
# Distribution
# ----------------------------------------------------------

st.subheader("📊 Distribution of Unemployment Rate")

fig = px.histogram(
    filtered_df,
    x="Estimated Unemployment Rate (%)",
    nbins=30,
    title="Distribution of Unemployment Rate"
)

fig.update_layout(height=450)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------
# Scatter Plot
# ----------------------------------------------------------

st.subheader("🔵 Labour Participation vs Unemployment")

fig = px.scatter(
    filtered_df,
    x="Estimated Labour Participation Rate (%)",
    y="Estimated Unemployment Rate (%)",
    color="Region",
    hover_name="Region",
    title="Labour Participation vs Unemployment"
)

fig.update_layout(height=600)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------
# Top 10 States
# ----------------------------------------------------------

st.subheader("🏆 Top 10 States with Highest Unemployment")

top10 = state_avg.head(10)

st.dataframe(
    top10,
    use_container_width=True
)

# ----------------------------------------------------------
# Raw Data
# ----------------------------------------------------------

with st.expander("📄 View Filtered Dataset"):

    st.dataframe(
        filtered_df,
        use_container_width=True
    )