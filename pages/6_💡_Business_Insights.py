# ==========================================================
# 💡 Business Insights
# ==========================================================

import streamlit as st
import plotly.express as px

from utils.load_data import load_main_data

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------

df = load_main_data()

st.title("💡 Business Insights")

st.markdown(
    """
This page summarizes the unemployment dataset into meaningful
business insights and recommendations.
"""
)

st.divider()

# ----------------------------------------------------------
# State-wise Analysis
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

highest = state_df.loc[
    state_df["Estimated Unemployment Rate (%)"].idxmax()
]

lowest = state_df.loc[
    state_df["Estimated Unemployment Rate (%)"].idxmin()
]

highest_employed = state_df.loc[
    state_df["Estimated Employed"].idxmax()
]

highest_labour = state_df.loc[
    state_df["Estimated Labour Participation Rate (%)"].idxmax()
]

# ----------------------------------------------------------
# KPI Cards
# ----------------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    st.success(
        f"""
### Highest Unemployment

**State:** {highest['Region']}

**Rate:** {highest['Estimated Unemployment Rate (%)']:.2f}%
"""
    )

with col2:

    st.success(
        f"""
### Lowest Unemployment

**State:** {lowest['Region']}

**Rate:** {lowest['Estimated Unemployment Rate (%)']:.2f}%
"""
    )

st.divider()

# ----------------------------------------------------------
# Charts
# ----------------------------------------------------------

st.subheader("📊 Average Unemployment by State")

state_df = state_df.sort_values(
    "Estimated Unemployment Rate (%)",
    ascending=False
)

fig = px.bar(
    state_df,
    x="Region",
    y="Estimated Unemployment Rate (%)",
    color="Estimated Unemployment Rate (%)",
    text_auto=".2f"
)

fig.update_layout(
    height=550,
    xaxis_tickangle=-45
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------
# Key Insights
# ----------------------------------------------------------

st.subheader("📌 Key Insights")

st.info(
    f"""
• Highest unemployment was observed in **{highest['Region']}**.

• Lowest unemployment was observed in **{lowest['Region']}**.

• Highest employment was recorded in **{highest_employed['Region']}**.

• Highest labour participation belongs to **{highest_labour['Region']}**.

• Average unemployment across India is
**{df['Estimated Unemployment Rate (%)'].mean():.2f}%**.
"""
)

# ----------------------------------------------------------
# Recommendations
# ----------------------------------------------------------

st.subheader("🎯 Business Recommendations")

recommendations = [
    "Increase employment opportunities in high-unemployment states.",
    "Promote skill development and vocational training.",
    "Support MSMEs and startup ecosystems.",
    "Invest in rural employment programs.",
    "Strengthen labour market monitoring.",
    "Expand digital employment platforms.",
    "Improve state-level economic planning.",
    "Encourage private-sector job creation."
]

for i, rec in enumerate(recommendations, start=1):
    st.write(f"**{i}.** {rec}")

# ----------------------------------------------------------
# Executive Summary
# ----------------------------------------------------------

st.subheader("📄 Executive Summary")

st.markdown(
    f"""
The unemployment analysis indicates noticeable differences among
Indian states.

- National Average Unemployment:
  **{df['Estimated Unemployment Rate (%)'].mean():.2f}%**

- Highest Unemployment:
  **{highest['Region']}**

- Lowest Unemployment:
  **{lowest['Region']}**

Government agencies can use this analysis to identify priority
regions for employment generation and economic development.
"""
)

# ----------------------------------------------------------
# Download Report
# ----------------------------------------------------------

report = f"""
UNEMPLOYMENT ANALYSIS REPORT

Average Unemployment:
{df['Estimated Unemployment Rate (%)'].mean():.2f}%

Highest Unemployment:
{highest['Region']}

Lowest Unemployment:
{lowest['Region']}

Highest Employment:
{highest_employed['Region']}

Highest Labour Participation:
{highest_labour['Region']}
"""

st.download_button(
    "⬇ Download Business Report",
    report,
    file_name="Business_Insights_Report.txt",
    mime="text/plain"
)

st.divider()

st.success("Business Insights Generated Successfully ✅")