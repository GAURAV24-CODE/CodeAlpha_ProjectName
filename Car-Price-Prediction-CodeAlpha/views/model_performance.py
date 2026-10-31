"""Model performance page backed by the project's saved evaluation artifacts."""
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils import model, model_resource_path, MODEL_NAME


def _load_artifact(name, default=None):
    try:
        return joblib.load(model_resource_path(name))
    except Exception:
        return default


def render_model_performance():
    metrics = _load_artifact("model_metrics.pkl", {}) or {}
    training_info = _load_artifact("training_info.pkl", {}) or {}
    y_test = np.asarray(_load_artifact("y_test.pkl", []))
    y_pred = np.asarray(_load_artifact("y_pred.pkl", []))
    feature_names = training_info.get("Features", [])

    st.title("🤖 Model Performance Dashboard")
    st.caption("Metrics and charts below are generated from the project's saved model artifacts.")

    st.subheader("📌 Model Overview")
    st.info(f"**Algorithm:** {metrics.get('algorithm', MODEL_NAME)}  \n**Task:** Regression  \n**Library:** Scikit-learn  \n**Deployment:** Streamlit")

    st.subheader("📊 Evaluation Metrics")
    r2 = float(metrics.get("r2", np.nan))
    mae = float(metrics.get("mae", np.nan))
    rmse = float(metrics.get("rmse", np.nan))
    mse = float(metrics.get("mse", np.nan))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² Score", f"{r2:.3f}")
    c2.metric("MAE", f"{mae:.3f}")
    c3.metric("RMSE", f"{rmse:.3f}")
    c4.metric("MSE", f"{mse:.3f}")

    if np.isfinite(r2):
        st.progress(max(0, min(100, int(r2 * 100))))
        st.caption("R² indicates how much variation in the test target is explained by the model.")

    if len(y_test) and len(y_test) == len(y_pred):
        st.subheader("📈 Actual vs Predicted")
        results = pd.DataFrame({"Actual Price": y_test, "Predicted Price": y_pred})
        fig = px.scatter(results, x="Actual Price", y="Predicted Price", template="plotly_white")
        low = float(min(results.min()))
        high = float(max(results.max()))
        fig.add_shape(type="line", x0=low, y0=low, x1=high, y1=high, line=dict(dash="dash"))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📉 Residual Error Distribution")
        residuals = results["Actual Price"] - results["Predicted Price"]
        fig = px.histogram(x=residuals, nbins=20, template="plotly_white", labels={"x": "Residual Error"})
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("⭐ Feature Importance")
    importance = getattr(model, "feature_importances_", None)
    if importance is not None:
        names = feature_names or [f"Feature {i+1}" for i in range(len(importance))]
        importance_df = pd.DataFrame({"Feature": names, "Importance": importance}).sort_values("Importance")
        fig = px.bar(importance_df, x="Importance", y="Feature", orientation="h", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance is not available for the saved model.")

    st.subheader("📋 Training Summary")
    summary = pd.DataFrame({"Item": list(training_info.keys()), "Value": list(training_info.values())})
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.subheader("⚠️ Limitations")
    st.markdown("- Performance depends on the available training data.
- Vehicle condition and location are not represented by the saved feature set.
- Market prices can change over time, so periodic retraining would improve long-term reliability.")
