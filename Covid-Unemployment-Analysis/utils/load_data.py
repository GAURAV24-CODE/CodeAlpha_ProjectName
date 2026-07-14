# ==========================================================
# Data Loader
# ==========================================================

import pandas as pd
import streamlit as st


# ==========================================================
# Load Main Dataset
# ==========================================================

@st.cache_data
def load_main_data():
    """
    Load and preprocess the main unemployment dataset.
    """

    df = pd.read_csv("data/Unemployment in India.csv")

    # Remove extra spaces from column names
    df.columns = df.columns.str.strip()

    # Convert Date column
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(
            df["Date"],
            dayfirst=True,
            errors="coerce"
        )

    # Remove missing values
    df = df.dropna()

    return df


# ==========================================================
# Load COVID Dataset
# ==========================================================

@st.cache_data
def load_covid_data():
    """
    Load and preprocess the COVID unemployment dataset.
    """

    df = pd.read_csv("data/Unemployment_Rate_upto_11_2020.csv")

    # Remove extra spaces
    df.columns = df.columns.str.strip()

    # Convert Date column
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(
            df["Date"],
            dayfirst=True,
            errors="coerce"
        )

    # Remove missing values
    df = df.dropna()

    return df


# ==========================================================
# Dataset Information
# ==========================================================

def get_dataset_summary(df):
    """
    Return summary statistics.
    """

    summary = {
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Missing Values": df.isnull().sum().sum(),
        "Duplicate Rows": df.duplicated().sum()
    }

    return summary


# ==========================================================
# State List
# ==========================================================

def get_states(df):
    """
    Return sorted list of states.
    """

    if "Region" in df.columns:
        return sorted(df["Region"].dropna().unique())

    return []


# ==========================================================
# Numeric Columns
# ==========================================================

def get_numeric_columns(df):
    """
    Return numeric columns only.
    """

    return df.select_dtypes(include=["number"]).columns.tolist()