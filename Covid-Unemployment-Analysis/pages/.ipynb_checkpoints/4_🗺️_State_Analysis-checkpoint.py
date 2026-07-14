# ==========================================================
# 🗺️ State Analysis
# ==========================================================

import streamlit as st
import plotly.express as px
from utils.load_data import load_main_data

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------

df = load_main_data()

st.title("🗺️ State Analysis")

st.markdown(
    "Compare unemployment statistics across Indian states."
)

st.divider()

# ----------------------------------------------------------
# Calculate State-wise Statistics
# ----------------------------------------------------------

state_df = (
    df.groupby("Region")
    .agg({
        "Estimated Unemployment Rate (%)": "mean",
        "Estimated Employed": "mean",
        "Estimated Labour Participation Rate (%)": "mean"
    })
    .reset_index()
)

state_df = state_df.sort_values(
    "Estimated Unemployment Rate (%)",
    ascending=False
)

# ----------------------------------------------------------
# KPI Cards
# ----------------------------------------------------------

highest_state = state_df.iloc[0]
lowest_state = state_df.iloc[-1]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Highest Unemployment",
        highest_state["Region"],
        f"{highest_state['Estimated Unemployment Rate (%)']:.2f}%"
    )

with col2:
    st.metric(
        "Lowest Unemployment",
        lowest_state["Region"],
        f"{lowest_state['Estimated Unemployment Rate (%)']:.2f}%"
    )

with col3:
    st.metric(
        "Average Unemployment",
        f"{state_df['Estimated Unemployment Rate (%)'].mean():.2f}%"
    )

st.divider()

# ----------------------------------------------------------
# Interactive State Selection
# ----------------------------------------------------------

selected_states = st.multiselect(
    "Select State(s)",
    sorted(state_df["Region"].unique()),
    default=sorted(state_df["Region"].unique())[:5]
)

filtered = state_df[
    state_df["Region"].isin(selected_states)
]

# ----------------------------------------------------------
# Comparison Chart
# ----------------------------------------------------------

st.subheader("📊 State Comparison")

fig = px.bar(
    filtered,
    x="Region",
    y="Estimated Unemployment Rate (%)",
    color="Estimated Unemployment Rate (%)",
    text_auto=".2f",
    title="Average Unemployment Rate"
)

fig.update_layout(
    height=500,
    xaxis_tickangle=-45
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------
# Pie Chart
# ----------------------------------------------------------

st.subheader("🥧 Share of Unemployment")

fig = px.pie(
    filtered,
    names="Region",
    values="Estimated Unemployment Rate (%)",
    hole=0.45
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------
# Top 10 States
# ----------------------------------------------------------

st.subheader("🏆 Top 10 Highest Unemployment States")

top10 = state_df.head(10)

st.dataframe(
    top10,
    use_container_width=True,
    hide_index=True
)

# ----------------------------------------------------------
# Bottom 10 States
# ----------------------------------------------------------

st.subheader("🥇 Lowest 10 Unemployment States")

bottom10 = (
    state_df
    .sort_values(
        "Estimated Unemployment Rate (%)"
    )
    .head(10)
)

st.dataframe(
    bottom10,
    use_container_width=True,
    hide_index=True
)

# ----------------------------------------------------------
# Scatter Plot
# ----------------------------------------------------------

st.subheader("📍 Employment vs Unemployment")

fig = px.scatter(
    state_df,
    x="Estimated Employed",
    y="Estimated Unemployment Rate (%)",
    size="Estimated Labour Participation Rate (%)",
    color="Estimated Labour Participation Rate (%)",
    hover_name="Region",
    size_max=40,
    title="Employment vs Unemployment"
)

fig.update_layout(height=600)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------
# Ranking Table
# ----------------------------------------------------------

st.subheader("📋 State Ranking")

ranking = state_df.copy()

ranking.insert(
    0,
    "Rank",
    range(1, len(ranking) + 1)
)

st.dataframe(
    ranking,
    use_container_width=True,
    hide_index=True
)

# ----------------------------------------------------------
# Download Ranking
# ----------------------------------------------------------

csv = ranking.to_csv(index=False).encode("utf-8")

st.download_button(
    "⬇ Download Ranking",
    csv,
    "state_ranking.csv",
    "text/csv"
)

st.divider()

st.success("State Analysis Completed Successfully ✅")