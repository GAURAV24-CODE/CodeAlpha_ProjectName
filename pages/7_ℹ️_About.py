# ==========================================================
# ℹ️ About Page
# ==========================================================

import streamlit as st

st.set_page_config(
    page_title="About",
    page_icon="ℹ️",
    layout="wide"
)

# ----------------------------------------------------------
# HERO SECTION
# ----------------------------------------------------------
st.markdown(
    """
    <div style='text-align:center;padding:20px;'>
        <h1>👨‍💻 Gaurav Eknath Kumbhar</h1>
        <h3 style='color:#4F8BF9;'>Data Science & Machine Learning Enthusiast</h3>
        <h4>📚 MCA Student | 💼 CodeAlpha Data Science Intern | 📍 Maharashtra, India</h4>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()
# ----------------------------------------------------------
# PROJECT OVERVIEW
# ----------------------------------------------------------

with st.container(border=True):
    st.header("📊 Project Overview")

    st.write("""
The **Unemployment Analysis Dashboard** is an interactive data analytics project
developed using **Python** and **Streamlit**.

This dashboard explores unemployment trends across India using interactive
visualizations, KPIs, state-wise analysis, and COVID-19 impact analysis.

It was developed as part of the **CodeAlpha Data Science Internship** and
demonstrates practical data science, business analytics, and dashboard
development skills.
""")

st.write("")

# ----------------------------------------------------------
# DASHBOARD STATS
# ----------------------------------------------------------

st.header("📈 Project Highlights")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Pages", "7")
c2.metric("Visualizations", "20+")
c3.metric("Datasets", "2")
c4.metric("Technologies", "8")

st.divider()

# ----------------------------------------------------------
# OBJECTIVES & FEATURES
# ----------------------------------------------------------

left, right = st.columns(2)

with left:
    with st.container(border=True):
        st.subheader("🎯 Objectives")

        st.markdown("""
- Analyze unemployment trends
- State-wise comparison
- COVID-19 impact analysis
- Identify hidden patterns
- Business insights
- Interactive dashboard
""")

with right:
    with st.container(border=True):
        st.subheader("✨ Key Features")

        st.markdown("""
- Interactive Dashboard
- KPI Cards
- Plotly Charts
- Dataset Explorer
- COVID Analysis
- State Analysis
- Download Reports
- Responsive UI
""")

st.divider()

# ----------------------------------------------------------
# TECHNOLOGY STACK
# ----------------------------------------------------------

st.header("🛠️ Technology Stack")

t1, t2, t3 = st.columns(3)

with t1:
    with st.container(border=True):
        st.markdown("""
### Programming

🐍 Python

📄 Pandas

🔢 NumPy
""")

with t2:
    with st.container(border=True):
        st.markdown("""
### Visualization

📊 Plotly

🚀 Streamlit

🎨 CSS
""")

with t3:
    with st.container(border=True):
        st.markdown("""
### Tools

💻 VS Code

🔀 Git

🌐 GitHub
""")

st.divider()

# ----------------------------------------------------------
# SKILLS
# ----------------------------------------------------------

st.header("💡 Skills Demonstrated")

s1, s2 = st.columns(2)

with s1:
    with st.container(border=True):
        st.markdown("""
✅ Data Cleaning

✅ Data Analysis

✅ Exploratory Data Analysis

✅ Python Programming
""")

with s2:
    with st.container(border=True):
        st.markdown("""
✅ Dashboard Development

✅ Business Analytics

✅ Interactive Reporting

✅ Streamlit Development
""")

st.divider()

# ----------------------------------------------------------
# CONTACT
# ----------------------------------------------------------

st.header("📬 Connect With Me")

c1, c2, c3 = st.columns(3)

with c1:
    with st.container(border=True):
        st.markdown("""
### 📧 Email

**kumbhargaurav24@gmail.com**
""")

with c2:
    with st.container(border=True):
        st.markdown("""
### 🐙 GitHub

https://github.com/GAURAV24-CODE
""")

with c3:
    with st.container(border=True):
        st.markdown("""
### 💼 LinkedIn

https://www.linkedin.com/in/gaurav-kumbhar-0b4a39293
""")

st.divider()

# ----------------------------------------------------------
# FOOTER
# ----------------------------------------------------------

st.markdown(
    """
    <div style='text-align:center;padding:15px;'>
        <h3>⭐ Thank You for Visiting ⭐</h3>
        <p>Developed with ❤️ using Python & Streamlit</p>
        <b>CodeAlpha Data Science Internship Project</b>
    </div>
    """,
    unsafe_allow_html=True,
)