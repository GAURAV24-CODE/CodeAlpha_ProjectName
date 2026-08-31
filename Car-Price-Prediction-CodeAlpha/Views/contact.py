"""
📬 Contact page for the Car Price Prediction Streamlit app.
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


def render_contact():

    st.markdown("""
    <div class="hero-container">
        <h1>📬 Contact Me</h1>
        <p>
        Thank you for visiting my Machine Learning project.
        Feel free to connect with me for collaborations,
        internships, projects, or opportunities.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # --------------------------------------------------
    # PROFILE SECTION
    # --------------------------------------------------

    left, right = st.columns([1,2])

    with left:

        try:
            st.image(
                asset_path("gaurav.png"),
                use_container_width=True
            )
        except:
            st.info("Add profile image in assets/gaurav.png")

    with right:

        st.markdown("""
# 👨‍💻 Gaurav Eknath Kumbhar

### 🎓 MCA Student

📍 Maharashtra, India

🚀 Aspiring Data Scientist

🤖 Machine Learning Enthusiast

📊 Data Analyst

💻 Python Developer
""")

        st.success("""
I enjoy solving real-world problems using
Machine Learning, Data Analytics and
interactive dashboards.

Currently building projects in:

• Python

• Data Science

• Machine Learning

• Streamlit

• SQL
""")

    st.divider()

    # --------------------------------------------------
    # ABOUT ME
    # --------------------------------------------------

    st.subheader("🏆 About Me")

    st.write("""
I am passionate about Artificial Intelligence,
Machine Learning, Data Science and Software
Development.

My goal is to become a professional AI Engineer
by building impactful real-world projects and
continuously improving my technical skills.

I enjoy learning new technologies and creating
clean, user-friendly applications.
""")

    st.divider()

    # --------------------------------------------------
    # SKILLS
    # --------------------------------------------------

    st.subheader("💼 Technical Skills")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.info("""
### Programming

🐍 Python

🗄 SQL

📒 Jupyter

Git & GitHub
""")

    with c2:

        st.info("""
### Data Science

Pandas

NumPy

Matplotlib

Seaborn

Plotly
""")

    with c3:

        st.info("""
### Machine Learning

Scikit-Learn

Regression

Classification

Feature Engineering

Model Evaluation
""")

    st.divider()

    # --------------------------------------------------
    # SOCIAL LINKS
    # --------------------------------------------------

    st.subheader("🌐 Connect With Me")

    github_url = "https://github.com/yourusername"

    linkedin_url = "https://linkedin.com/in/yourusername"

    email = "your.email@gmail.com"

    c1, c2, c3 = st.columns(3)

    with c1:

        st.link_button(
            "💻 GitHub",
            github_url,
            use_container_width=True
        )

    with c2:

        st.link_button(
            "💼 LinkedIn",
            linkedin_url,
            use_container_width=True
        )

    with c3:

        st.link_button(
            "📧 Email",
            f"mailto:{email}",
            use_container_width=True
        )

    st.divider()

    # --------------------------------------------------
    # RESUME
    # --------------------------------------------------

    st.subheader("📄 Resume")

    try:

        with open(
            asset_path("Gaurav_Kumbhar_Resume.pdf"),
            "rb"
        ) as file:

            st.download_button(

                "⬇ Download Resume",

                file,

                file_name="Gaurav_Kumbhar_Resume.pdf",

                mime="application/pdf",

                use_container_width=True

            )

    except:

        st.info(
            "Add your resume to assets/Gaurav_Kumbhar_Resume.pdf"
        )

    st.divider()

    # --------------------------------------------------
    # QR CODE
    # --------------------------------------------------

    st.subheader("📱 Scan QR Code")

    try:

        st.image(
            asset_path("qr_code.png"),
            width=250
        )

    except:

        st.info(
            "Add QR Code image to assets/qr_code.png"
        )

    st.divider()

    # --------------------------------------------------
    # CONTACT INFORMATION
    # --------------------------------------------------

    st.subheader("☎ Contact Information")

    info = pd.DataFrame({

        "Information":[

            "Location",

            "Education",

            "Career Goal",

            "Specialization"

        ],

        "Details":[

            "Maharashtra, India",

            "MCA Student",

            "AI / ML Engineer",

            "Data Science & Machine Learning"

        ]

    })

    st.dataframe(
        info,
        hide_index=True,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------
    # THANK YOU
    # --------------------------------------------------

    st.markdown("""

<div class="prediction-card">

<h2>🙏 Thank You!</h2>

<p>

Thank you for exploring this project.

I hope you enjoyed using this Machine Learning
application.

Feel free to connect with me for internships,
collaborations or professional opportunities.

⭐ If you like this project,
consider starring it on GitHub.

</p>

</div>

""", unsafe_allow_html=True)

















