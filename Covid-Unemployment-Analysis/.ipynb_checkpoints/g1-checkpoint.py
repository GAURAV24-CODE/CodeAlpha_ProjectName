# ==========================================================
# 🇮🇳 INDIA UNEMPLOYMENT ANALYSIS DASHBOARD
# Developed by: Gaurav Eknath Kumbhar
# ==========================================================

# ==========================
# IMPORT LIBRARIES
# ==========================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ==========================
# PAGE CONFIGURATION
# ==========================

st.set_page_config(
    page_title="India Unemployment Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# CUSTOM CSS
# ==========================

st.markdown("""
<style>

.main{
    background-color:#F8F9FA;
}

[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#1E3A8A,#0F172A);
}

[data-testid="stSidebar"] *{
    color:white;
}

.title{
    font-size:45px;
    font-weight:bold;
    color:#1E3A8A;
}

.subtitle{
    font-size:20px;
    color:gray;
}

.metric-card{
    background:white;
    padding:18px;
    border-radius:15px;
    box-shadow:0px 5px 15px rgba(0,0,0,0.1);
    text-align:center;
}

.footer{
    text-align:center;
    color:gray;
    padding:20px;
}

</style>
""", unsafe_allow_html=True)

# ==========================
# LOAD DATA
# ==========================

@st.cache_data
def load_data():

    df = pd.read_csv("data/Unemployment in India.csv")

    df.columns = df.columns.str.strip()

    df["Date"] = pd.to_datetime(df["Date"])

    df["Year"] = df["Date"].dt.year

    df["Month"] = df["Date"].dt.month_name()

    return df


df = load_data()

# ==========================
# SIDEBAR
# ==========================

st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/3135/3135715.png",
    width=120
)

st.sidebar.title("Navigation")

page = st.sidebar.radio(

    "Select Page",

    [

        "🏠 Home",

        "📂 Dataset Explorer",

        "📊 Advanced EDA",

        "📈 State Analysis",

        "📅 Time Series",

        "💡 Insights",

        "📥 Downloads",

        "👨‍💻 About"

    ]

)

st.sidebar.markdown("---")

st.sidebar.info(
    "📊 India Unemployment Analysis Dashboard"
)

# =====================================================
# HOME PAGE
# =====================================================

if page == "🏠 Home":

    st.markdown(
        "<h1 class='title'>🇮🇳 India Unemployment Analysis Dashboard</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p class='subtitle'>Interactive Data Analytics Dashboard using Streamlit & Plotly</p>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    col1,col2,col3,col4=st.columns(4)

    col1.metric(
        "📄 Total Records",
        f"{len(df):,}"
    )

    col2.metric(
        "🗺️ States",
        df["Region"].nunique()
    )

    col3.metric(
        "📈 Avg Unemployment",
        f"{df['Estimated Unemployment Rate (%)'].mean():.2f}%"
    )

    col4.metric(
        "📊 Max Rate",
        f"{df['Estimated Unemployment Rate (%)'].max():.2f}%"
    )

    st.markdown("---")

    c1,c2=st.columns([2,1])

    with c1:

        fig=px.line(
            df,
            x="Date",
            y="Estimated Unemployment Rate (%)",
            color="Region",
            title="Unemployment Trend Across India"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with c2:

        st.subheader("📌 Project Overview")

        st.write("""
This dashboard provides an interactive analysis of unemployment trends across India.

### Features

- Interactive Dashboard
- Dataset Explorer
- 20+ Advanced Charts
- State-wise Analysis
- Time Series Analysis
- Key Insights
- Download Reports

Developed using:

- Streamlit
- Plotly
- Pandas
- Python
""")

    st.markdown("---")

    st.success("✅ Dashboard Loaded Successfully")

    st.markdown(
        "<div class='footer'>Developed by <b>Gaurav Eknath Kumbhar</b></div>",
        unsafe_allow_html=True
    )





# =====================================================
# 📂 DATASET EXPLORER
# =====================================================

elif page == "📂 Dataset Explorer":

    st.title("📂 Dataset Explorer")
    st.markdown("Explore, filter, analyze, and download the unemployment dataset.")

    st.markdown("---")

    # ==========================
    # DATASET OVERVIEW
    # ==========================

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("📄 Total Records", len(df))
    col2.metric("📌 Total Columns", len(df.columns))
    col3.metric("🗺️ States", df["Region"].nunique())
    col4.metric("❌ Missing Values", int(df.isnull().sum().sum()))

    st.markdown("---")

    # ==========================
    # FILTERS
    # ==========================

    st.subheader("🔍 Dataset Filters")

    col1, col2, col3 = st.columns(3)

    with col1:
        state = st.selectbox(
            "Select State",
            ["All"] + sorted(df["Region"].dropna().unique().tolist())
        )

    with col2:
        year = st.selectbox(
            "Select Year",
            ["All"] + sorted(df["Year"].unique().tolist())
        )

    with col3:
        month = st.selectbox(
            "Select Month",
            ["All"] + sorted(df["Month"].unique().tolist())
        )

    filtered_df = df.copy()

    if state != "All":
        filtered_df = filtered_df[filtered_df["Region"] == state]

    if year != "All":
        filtered_df = filtered_df[filtered_df["Year"] == year]

    if month != "All":
        filtered_df = filtered_df[filtered_df["Month"] == month]

    st.success(f"Showing {len(filtered_df)} records")

    st.markdown("---")

    # ==========================
    # SEARCH
    # ==========================

    search = st.text_input("🔎 Search State")

    if search:
        filtered_df = filtered_df[
            filtered_df["Region"].str.contains(search, case=False, na=False)
        ]

    # ==========================
    # DATASET PREVIEW
    # ==========================

    st.subheader("📋 Dataset Preview")

    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=500
    )

    # ==========================
    # RANDOM SAMPLE
    # ==========================

    with st.expander("🎲 View Random Sample"):

        sample_size = st.slider(
            "Number of Rows",
            5,
            min(20, len(filtered_df)),
            10
        )

        st.dataframe(
            filtered_df.sample(sample_size),
            use_container_width=True
        )

    # ==========================
    # DATA INFORMATION
    # ==========================

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📏 Shape",
            "📊 Statistics",
            "❌ Missing Values",
            "📋 Data Types"
        ]
    )

    with tab1:

        st.write("Rows :", filtered_df.shape[0])
        st.write("Columns :", filtered_df.shape[1])

    with tab2:

        st.dataframe(
            filtered_df.describe(include="all"),
            use_container_width=True
        )

    with tab3:

        missing = pd.DataFrame({
            "Column": filtered_df.columns,
            "Missing Values": filtered_df.isnull().sum().values
        })

        st.dataframe(
            missing,
            use_container_width=True
        )

    with tab4:

        dtype = pd.DataFrame({
            "Column": filtered_df.columns,
            "Data Type": filtered_df.dtypes.astype(str).values
        })

        st.dataframe(
            dtype,
            use_container_width=True
        )

    # ==========================
    # DUPLICATES
    # ==========================

    st.markdown("---")

    st.subheader("📌 Duplicate Records")

    duplicate_count = filtered_df.duplicated().sum()

    st.metric(
        "Duplicate Rows",
        duplicate_count
    )

    # ==========================
    # COLUMN INFORMATION
    # ==========================

    st.markdown("---")

    st.subheader("📚 Column Information")

    info_df = pd.DataFrame({
        "Column": filtered_df.columns,
        "Non-Null Count": filtered_df.count().values,
        "Data Type": filtered_df.dtypes.astype(str).values
    })

    st.dataframe(
        info_df,
        use_container_width=True
    )

    # ==========================
    # DOWNLOAD DATASET
    # ==========================

    st.markdown("---")

    csv = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Filtered Dataset",
        data=csv,
        file_name="filtered_unemployment_data.csv",
        mime="text/csv"
    )



# =====================================================
# 📊 ADVANCED EDA (Charts 1–10)
# =====================================================

elif page == "📊 Advanced EDA":

    st.title("📊 Advanced Exploratory Data Analysis")
    st.markdown("Interactive visualizations to explore unemployment trends across India.")

    st.markdown("---")

    # -------------------------
    # Sidebar Filters
    # -------------------------

    col1, col2 = st.columns(2)

    with col1:
        selected_state = st.selectbox(
            "Select State",
            ["All"] + sorted(df["Region"].unique().tolist()),
            key="eda_state"
        )

    with col2:
        selected_year = st.selectbox(
            "Select Year",
            ["All"] + sorted(df["Year"].unique().tolist()),
            key="eda_year"
        )

    eda_df = df.copy()

    if selected_state != "All":
        eda_df = eda_df[eda_df["Region"] == selected_state]

    if selected_year != "All":
        eda_df = eda_df[eda_df["Year"] == selected_year]

    st.success(f"Showing {len(eda_df)} records")

    st.markdown("---")

    tab1, tab2 = st.tabs(["📊 Charts 1–5", "📈 Charts 6–10"])

    # ==================================================
    # TAB 1
    # ==================================================

    with tab1:

        # 1 Histogram
        st.subheader("1️⃣ Distribution of Unemployment Rate")

        fig = px.histogram(
            eda_df,
            x="Estimated Unemployment Rate (%)",
            nbins=30,
            color="Region",
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

        # 2 Box Plot

        st.subheader("2️⃣ Box Plot")

        fig = px.box(
            eda_df,
            x="Region",
            y="Estimated Unemployment Rate (%)",
            color="Region",
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

        # 3 Violin Plot

        st.subheader("3️⃣ Violin Plot")

        fig = px.violin(
            eda_df,
            x="Region",
            y="Estimated Unemployment Rate (%)",
            color="Region",
            box=True,
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

        # 4 Pie Chart

        st.subheader("4️⃣ Average Unemployment by Region")

        pie = (
            eda_df
            .groupby("Region")["Estimated Unemployment Rate (%)"]
            .mean()
            .reset_index()
        )

        fig = px.pie(
            pie,
            names="Region",
            values="Estimated Unemployment Rate (%)",
            hole=0,
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

        # 5 Donut Chart

        st.subheader("5️⃣ Donut Chart")

        fig = px.pie(
            pie,
            names="Region",
            values="Estimated Unemployment Rate (%)",
            hole=0.55,
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

    # ==================================================
    # TAB 2
    # ==================================================

    with tab2:

        # 6 Bar Chart

        st.subheader("6️⃣ Average Unemployment by State")

        bar = (
            eda_df
            .groupby("Region")["Estimated Unemployment Rate (%)"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )

        fig = px.bar(
            bar,
            x="Region",
            y="Estimated Unemployment Rate (%)",
            color="Estimated Unemployment Rate (%)",
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

        # 7 Horizontal Bar

        st.subheader("7️⃣ Horizontal Ranking")

        fig = px.bar(
            bar,
            y="Region",
            x="Estimated Unemployment Rate (%)",
            orientation="h",
            color="Estimated Unemployment Rate (%)",
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

        # 8 Scatter Plot

        st.subheader("8️⃣ Labour Participation vs Unemployment")

        fig = px.scatter(
            eda_df,
            x="Estimated Labour Participation Rate (%)",
            y="Estimated Unemployment Rate (%)",
            color="Region",
            size="Estimated Employed",
            hover_name="Region",
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

        # 9 Bubble Chart

        st.subheader("9️⃣ Bubble Chart")

        fig = px.scatter(
            eda_df,
            x="Estimated Employed",
            y="Estimated Unemployment Rate (%)",
            color="Region",
            size="Estimated Labour Participation Rate (%)",
            hover_name="Region",
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

        # 10 Correlation Heatmap

        st.subheader("🔟 Correlation Heatmap")

        corr = eda_df[
            [
                "Estimated Unemployment Rate (%)",
                "Estimated Employed",
                "Estimated Labour Participation Rate (%)"
            ]
        ].corr()

        fig = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            aspect="auto"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.info("✅ Part 3 Complete – 10 Interactive Visualizations Added")

# =====================================================
# 📊 ADVANCED EDA (Charts 11–20)
# =====================================================

st.markdown("---")
st.header("📈 Advanced Visualizations (Charts 11–20)")

tab3, tab4 = st.tabs(["📅 Trends", "🌍 Advanced Charts"])

# =====================================================
# TAB 3 : TREND ANALYSIS
# =====================================================

with tab3:

    # 11 Monthly Trend
    st.subheader("1️⃣1️⃣ Monthly Average Unemployment Rate")

    monthly = (
        eda_df.groupby("Month")["Estimated Unemployment Rate (%)"]
        .mean()
        .reindex([
            "January","February","March","April","May","June",
            "July","August","September","October","November","December"
        ])
        .reset_index()
    )

    fig = px.line(
        monthly,
        x="Month",
        y="Estimated Unemployment Rate (%)",
        markers=True,
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    # 12 Yearly Trend

    st.subheader("1️⃣2️⃣ Yearly Average Unemployment")

    yearly = (
        eda_df.groupby("Year")["Estimated Unemployment Rate (%)"]
        .mean()
        .reset_index()
    )

    fig = px.line(
        yearly,
        x="Year",
        y="Estimated Unemployment Rate (%)",
        markers=True,
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    # 13 Area Chart

    st.subheader("1️⃣3️⃣ Area Chart")

    fig = px.area(
        yearly,
        x="Year",
        y="Estimated Unemployment Rate (%)",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    # 14 Region Trend

    st.subheader("1️⃣4️⃣ Region-wise Trend")

    region = (
        eda_df.groupby(["Year", "Region"])["Estimated Unemployment Rate (%)"]
        .mean()
        .reset_index()
    )

    fig = px.line(
        region,
        x="Year",
        y="Estimated Unemployment Rate (%)",
        color="Region",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    # 15 Employment Trend

    st.subheader("1️⃣5️⃣ Employment Trend")

    employ = (
        eda_df.groupby("Year")["Estimated Employed"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        employ,
        x="Year",
        y="Estimated Employed",
        color="Estimated Employed",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# TAB 4 : ADVANCED CHARTS
# =====================================================

with tab4:

    # 16 Treemap

    st.subheader("1️⃣6️⃣ Treemap")

    tree = (
        eda_df.groupby("Region")[
            "Estimated Unemployment Rate (%)"
        ].mean().reset_index()
    )

    fig = px.treemap(
        tree,
        path=["Region"],
        values="Estimated Unemployment Rate (%)",
        color="Estimated Unemployment Rate (%)",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    # 17 Sunburst

    st.subheader("1️⃣7️⃣ Sunburst Chart")

    fig = px.sunburst(
        tree,
        path=["Region"],
        values="Estimated Unemployment Rate (%)",
        color="Estimated Unemployment Rate (%)",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    # 18 Top 10 States

    st.subheader("1️⃣8️⃣ Top 10 Highest Unemployment")

    top10 = (
        eda_df.groupby("Region")[
            "Estimated Unemployment Rate (%)"
        ]
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
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    # 19 Bottom 10 States

    st.subheader("1️⃣9️⃣ Lowest Unemployment")

    low10 = (
        eda_df.groupby("Region")[
            "Estimated Unemployment Rate (%)"
        ]
        .mean()
        .sort_values()
        .head(10)
        .reset_index()
    )

    fig = px.bar(
        low10,
        x="Region",
        y="Estimated Unemployment Rate (%)",
        color="Estimated Unemployment Rate (%)",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    # 20 Labour Participation

    st.subheader("2️⃣0️⃣ Labour Participation by Region")

    labour = (
        eda_df.groupby("Region")[
            "Estimated Labour Participation Rate (%)"
        ]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        labour,
        x="Region",
        y="Estimated Labour Participation Rate (%)",
        color="Estimated Labour Participation Rate (%)",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.success("✅ Part 4 Completed Successfully")

# =====================================================
# 📈 STATE ANALYSIS & TIME SERIES
# =====================================================

elif page == "📈 State Analysis":

    st.title("📈 State Analysis Dashboard")
    st.markdown("Analyze unemployment trends for individual states.")

    st.markdown("---")

    # -------------------------------
    # State Selection
    # -------------------------------

    state = st.selectbox(
        "🗺️ Select State",
        sorted(df["Region"].unique())
    )

    state_df = df[df["Region"] == state].copy()

    st.markdown("---")

    # -------------------------------
    # KPI Cards
    # -------------------------------

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Average Unemployment",
        f"{state_df['Estimated Unemployment Rate (%)'].mean():.2f}%"
    )

    c2.metric(
        "Maximum Rate",
        f"{state_df['Estimated Unemployment Rate (%)'].max():.2f}%"
    )

    c3.metric(
        "Average Employment",
        f"{state_df['Estimated Employed'].mean():,.0f}"
    )

    c4.metric(
        "Labour Participation",
        f"{state_df['Estimated Labour Participation Rate (%)'].mean():.2f}%"
    )

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(
        [
            "📈 Trend Analysis",
            "📊 Comparison",
            "📋 Insights"
        ]
    )

    # =====================================
    # TREND ANALYSIS
    # =====================================

    with tab1:

        st.subheader("Monthly Unemployment Trend")

        fig = px.line(
            state_df,
            x="Date",
            y="Estimated Unemployment Rate (%)",
            markers=True,
            title=f"{state} Unemployment Trend",
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Employment Trend")

        fig = px.area(
            state_df,
            x="Date",
            y="Estimated Employed",
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Labour Participation")

        fig = px.line(
            state_df,
            x="Date",
            y="Estimated Labour Participation Rate (%)",
            markers=True,
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Moving Average

        st.subheader("30-Day Moving Average")

        state_df = state_df.sort_values("Date")

        state_df["Moving Average"] = (
            state_df["Estimated Unemployment Rate (%)"]
            .rolling(window=3)
            .mean()
        )

        fig = px.line(
            state_df,
            x="Date",
            y=[
                "Estimated Unemployment Rate (%)",
                "Moving Average"
            ],
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

    # =====================================
    # COMPARISON
    # =====================================

    with tab2:

        st.subheader("Before vs During COVID")

        state_df["COVID Period"] = state_df["Date"].apply(
            lambda x: "Before COVID"
            if x.year < 2020
            else "During COVID"
        )

        covid = (
            state_df.groupby("COVID Period")[
                "Estimated Unemployment Rate (%)"
            ]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            covid,
            x="COVID Period",
            y="Estimated Unemployment Rate (%)",
            color="COVID Period",
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Employment Comparison")

        emp = (
            state_df.groupby("COVID Period")[
                "Estimated Employed"
            ]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            emp,
            x="COVID Period",
            y="Estimated Employed",
            color="COVID Period",
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

    # =====================================
    # INSIGHTS
    # =====================================

    with tab3:

        st.subheader("State Summary")

        st.success(
            f"""
📍 State : {state}

📊 Average Unemployment :
{state_df['Estimated Unemployment Rate (%)'].mean():.2f}%

📈 Maximum Rate :
{state_df['Estimated Unemployment Rate (%)'].max():.2f}%

📉 Minimum Rate :
{state_df['Estimated Unemployment Rate (%)'].min():.2f}%

👥 Average Labour Participation :
{state_df['Estimated Labour Participation Rate (%)'].mean():.2f}%

💼 Average Employment :
{state_df['Estimated Employed'].mean():,.0f}
"""
        )

        st.subheader("Year-wise Statistics")

        yearly = (
            state_df.groupby("Year")[
                [
                    "Estimated Unemployment Rate (%)",
                    "Estimated Employed",
                    "Estimated Labour Participation Rate (%)"
                ]
            ]
            .mean()
            .round(2)
        )

        st.dataframe(
            yearly,
            use_container_width=True
        )

    # =====================================
    # TOP & BOTTOM STATES
    # =====================================

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🏆 Top 10 Highest Unemployment")

        top10 = (
            df.groupby("Region")[
                "Estimated Unemployment Rate (%)"
            ]
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
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        st.subheader("🥇 Lowest Unemployment")

        low10 = (
            df.groupby("Region")[
                "Estimated Unemployment Rate (%)"
            ]
            .mean()
            .sort_values()
            .head(10)
            .reset_index()
        )

        fig = px.bar(
            low10,
            x="Region",
            y="Estimated Unemployment Rate (%)",
            color="Estimated Unemployment Rate (%)",
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.success("✅ State Analysis Completed Successfully")

   

                    # =====================================================
# 💡 INSIGHTS
# =====================================================

elif page == "💡 Insights":

    st.title("💡 Dashboard Insights")
    st.markdown("Automatically generated insights from the unemployment dataset.")

    st.markdown("---")

    avg_rate = df["Estimated Unemployment Rate (%)"].mean()
    max_rate = df["Estimated Unemployment Rate (%)"].max()
    min_rate = df["Estimated Unemployment Rate (%)"].min()

    highest_state = (
        df.groupby("Region")["Estimated Unemployment Rate (%)"]
        .mean()
        .idxmax()
    )

    lowest_state = (
        df.groupby("Region")["Estimated Unemployment Rate (%)"]
        .mean()
        .idxmin()
    )

    highest_labour = (
        df.groupby("Region")["Estimated Labour Participation Rate (%)"]
        .mean()
        .idxmax()
    )

    highest_employed = (
        df.groupby("Region")["Estimated Employed"]
        .mean()
        .idxmax()
    )

    col1, col2 = st.columns(2)

    with col1:

        st.success(f"""
### 📈 Key Statistics

• Average Unemployment : **{avg_rate:.2f}%**

• Highest Recorded Rate : **{max_rate:.2f}%**

• Lowest Recorded Rate : **{min_rate:.2f}%**
""")

    with col2:

        st.info(f"""
### 🏆 Best & Worst

🔺 Highest Average Unemployment

**{highest_state}**

🔻 Lowest Average Unemployment

**{lowest_state}**
""")

    st.markdown("---")

    st.subheader("📌 Important Findings")

    st.write("✅ Highest average unemployment state :", highest_state)

    st.write("✅ Lowest average unemployment state :", lowest_state)

    st.write("✅ Highest labour participation :", highest_labour)

    st.write("✅ Highest employment :", highest_employed)

    st.write("✅ Dataset contains", len(df), "records.")

    st.write("✅ Number of States :", df["Region"].nunique())

    st.markdown("---")

    ranking = (
        df.groupby("Region")[
            "Estimated Unemployment Rate (%)"
        ]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    st.subheader("🏅 State Ranking")

    st.dataframe(
        ranking,
        use_container_width=True
    )

# =====================================================
# 📥 DOWNLOADS
# =====================================================

elif page == "📥 Downloads":

    st.title("📥 Download Center")

    st.markdown("---")

    st.subheader("Download Dataset")

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇ Download Dataset",
        csv,
        file_name="India_Unemployment.csv",
        mime="text/csv"
    )

    st.markdown("---")

    st.subheader("Dataset Summary")

    summary = f"""

INDIA UNEMPLOYMENT ANALYSIS

Total Records : {len(df)}

Total States : {df['Region'].nunique()}

Average Unemployment :
{df['Estimated Unemployment Rate (%)'].mean():.2f}%

Maximum Rate :
{df['Estimated Unemployment Rate (%)'].max():.2f}%

Minimum Rate :
{df['Estimated Unemployment Rate (%)'].min():.2f}%

Generated using Streamlit Dashboard.

Developed by
Gaurav Eknath Kumbhar

"""

    st.download_button(

        "📄 Download Summary",

        summary,

        file_name="dashboard_summary.txt"

    )

    st.success("Downloads Ready")

# =====================================================
# 👨‍💻 ABOUT
# =====================================================

elif page == "👨‍💻 About":

    st.title("👨‍💻 About")

    st.markdown("---")

    st.markdown("""
# 🇮🇳 India Unemployment Analysis Dashboard

A professional interactive dashboard developed using **Python, Streamlit, and Plotly** to analyze unemployment trends across India.

### 🎯 Project Objectives

- Explore unemployment data
- Analyze state-wise trends
- Compare employment statistics
- Study labour participation
- Understand COVID-19 impact
- Build an interactive analytics dashboard

---

## 🛠 Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- Matplotlib

---

## 📊 Dashboard Features

✅ Interactive Dashboard

✅ Dataset Explorer

✅ 20+ Interactive Charts

✅ State-wise Analysis

✅ Time Series Analysis

✅ Insights Dashboard

✅ Download Center

---

## 👨‍💻 Developed By

### Gaurav Eknath Kumbhar

🎓 MCA Student

📊 Aspiring Data Scientist

💻 Python | SQL | Power BI | Machine Learning

🌐 GitHub:
https://github.com/YOUR_USERNAME

🔗 LinkedIn:
https://linkedin.com/in/YOUR_PROFILE

📧 Email:
your_email@gmail.com

---

### Thank You ❤️

If you like this project, consider giving it a ⭐ on GitHub.
""")

    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    c1.metric("Charts", "20+")
    c2.metric("States", df["Region"].nunique())
    c3.metric("Records", len(df))

    st.success("Dashboard Developed Successfully 🚀")

# =====================================================
# FOOTER (DISPLAYED ON EVERY PAGE)
# =====================================================
    
  

# ============================================================
# 📊 Unemployment Analysis Dashboard
# CodeAlpha Data Science Internship Project
# Developed by: Gaurav Eknath Kumbhar
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Unemployment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Home",
        "📊 Dashboard",
        "📂 Dataset Explorer",
        "📈 Trend Analysis",
        "🗺️ State Analysis",
        "🦠 COVID Analysis",
        "📌 Insights",
        "ℹ️ About"
    ]
)

# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":

    st.title("📊 Unemployment Analysis Dashboard")

    st.markdown("""
Welcome to the **Unemployment Analysis Dashboard**.

This project was developed as part of the **CodeAlpha Data Science Internship**.

The dashboard analyzes unemployment trends across India using interactive visualizations built with **Python, Plotly, and Streamlit**.

### 🎯 Project Objectives

- Analyze unemployment trends
- Study COVID-19 impact
- Discover seasonal patterns
- Compare state-wise unemployment
- Generate business insights
""")

    st.markdown("---")

    # ========================================================
    # Information Cards
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "📁 Datasets",
            "2"
        )

    with col2:
        st.metric(
            "📊 Charts",
            "20+"
        )

    with col3:
        st.metric(
            "🗺️ States",
            "28+"
        )

    with col4:
        st.metric(
            "🦠 COVID Analysis",
            "Included"
        )

    st.markdown("---")

    # ========================================================
    # Technologies
    # ========================================================

    st.subheader("🛠️ Technologies Used")

    tech1, tech2, tech3, tech4 = st.columns(4)

    tech1.success("🐍 Python")
    tech2.info("📄 Pandas")
    tech3.warning("📊 Plotly")
    tech4.success("🚀 Streamlit")

    st.markdown("---")

    # ========================================================
    # Dashboard Features
    # ========================================================

    st.subheader("✨ Dashboard Features")

    feature1, feature2 = st.columns(2)

    with feature1:

        st.markdown("""
✅ Interactive Dashboard

✅ Dataset Explorer

✅ Trend Analysis

✅ State-wise Analysis

✅ COVID-19 Analysis

✅ Correlation Analysis
""")

    with feature2:

        st.markdown("""
✅ Plotly Interactive Charts

✅ KPI Cards

✅ Business Insights

✅ Download Ready

✅ Recruiter Friendly

✅ Streamlit Based
""")

    st.markdown("---")

    st.success("👈 Use the sidebar to explore the dashboard.")

    st.caption("Developed by **Gaurav Eknath Kumbhar**")

# ============================================================
# PLACEHOLDER PAGES
# ============================================================

elif page == "📊 Dashboard":
    st.title("📊 Dashboard")
    st.info("Coming in the next step...")

elif page == "📂 Dataset Explorer":
    st.title("📂 Dataset Explorer")
    st.info("Coming in the next step...")

elif page == "📈 Trend Analysis":
    st.title("📈 Trend Analysis")
    st.info("Coming in the next step...")

elif page == "🗺️ State Analysis":
    st.title("🗺️ State Analysis")
    st.info("Coming in the next step...")

elif page == "🦠 COVID Analysis":
    st.title("🦠 COVID Analysis")
    st.info("Coming in the next step...")

elif page == "📌 Insights":
    st.title("📌 Insights")
    st.info("Coming in the next step...")

elif page == "ℹ️ About":
    st.title("ℹ️ About")
    st.info("Coming in the next step...")
    