# ==========================================================
# 📊 Unemployment Analysis Dashboard
# CodeAlpha Data Science Internship
# Developed by: Gaurav Eknath Kumbhar
# ==========================================================

import streamlit as st

# ----------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------
st.set_page_config(
    page_title="Unemployment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# Custom Sidebar Styling
# ==========================================================

st.markdown("""
<style>

/* ---------------- Hide default page title (home/app) ---------------- */
[data-testid="stSidebarNav"] > div:first-child{
    display:none;
}

/* ---------------- Reduce top spacing ---------------- */
.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

/* ---------------- Sidebar width ---------------- */
[data-testid="stSidebar"]{
    min-width:280px;
    max-width:280px;
}

/* ---------------- Sidebar navigation ---------------- */
[data-testid="stSidebarNav"]{
    padding-top:10px;
}

/* ---------------- Navigation links ---------------- */
[data-testid="stSidebarNav"] a{
    font-size:18px;
    font-weight:500;
    padding:10px 14px;
    border-radius:10px;
    margin-bottom:6px;
}

[data-testid="stSidebarNav"] a:hover{
    background:#E8F0FE;
}

/* ---------------- Custom sidebar heading ---------------- */
.sidebar-title{
    font-size:24px;
    font-weight:700;
    text-align:center;
    color:#1f77b4;
    margin-bottom:15px;
}

.sidebar-footer{
    text-align:center;
    font-size:14px;
    color:gray;
    margin-top:30px;
}

</style>
""", unsafe_allow_html=True)

# Sidebar Header
st.sidebar.markdown(
    """
<div class="sidebar-title">
📊 Unemployment Dashboard
</div>
""",
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")

st.sidebar.markdown(
"""
### 👨‍💻 Developer

**Gaurav Eknath Kumbhar**

📚 MCA Student

💼 CodeAlpha Intern

📍 Maharashtra, India
"""
)

st.sidebar.markdown("---")

st.sidebar.markdown(
"""
<div class="sidebar-footer">
Made with ❤️ using Streamlit
</div>
""",
unsafe_allow_html=True
)

# ----------------------------------------------------------
# Custom CSS
# ----------------------------------------------------------

def load_css():
    try:
        with open("assets/style.css") as f:
            st.markdown(
                f"<style>{f.read()}</style>",
                unsafe_allow_html=True
            )
    except FileNotFoundError:
        pass


load_css()

# ----------------------------------------------------------
# Home Page
# ----------------------------------------------------------

st.title("📊 Unemployment Analysis Dashboard")

st.markdown("""
Welcome to the **Unemployment Analysis Dashboard**.

This project analyzes unemployment trends across India using
interactive visualizations built with **Python**, **Pandas**,
**Plotly**, and **Streamlit**.

---

### 🎯 Project Objectives

- Analyze unemployment trends
- Compare state-wise unemployment
- Study COVID-19 impact
- Discover seasonal patterns
- Generate business insights
- Build an interactive dashboard
""")

# ----------------------------------------------------------
# KPI Cards
# ----------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📁 Datasets", "2")

with col2:
    st.metric("📊 Charts", "20+")

with col3:
    st.metric("🗺️ States", "28+")

with col4:
    st.metric("🦠 COVID Analysis", "Included")

st.divider()

# ----------------------------------------------------------
# Technologies
# ----------------------------------------------------------

st.subheader("🛠 Technologies Used")

tech1, tech2, tech3, tech4 = st.columns(4)

tech1.success("🐍 Python")
tech2.info("📄 Pandas")
tech3.warning("📊 Plotly")
tech4.success("🚀 Streamlit")

st.divider()

# ----------------------------------------------------------
# Dashboard Features
# ----------------------------------------------------------

left, right = st.columns(2)

with left:
    st.markdown("""
### 📌 Features

- Interactive Dashboard
- KPI Cards
- State-wise Analysis
- Monthly Trends
- Dataset Explorer
- COVID Analysis
""")

with right:
    st.markdown("""
### ⭐ Highlights

- Plotly Charts
- Business Insights
- Download Data
- Responsive Layout
- Portfolio Ready
- Recruiter Friendly
""")

st.divider()

st.info(
    "👈 Use the sidebar to navigate through the different pages."
)

st.caption("Developed by **Gaurav Eknath Kumbhar**")







from utils.load_data import load_main_data

df = load_main_data()

st.write("Dataset Preview")

st.dataframe(df.head())