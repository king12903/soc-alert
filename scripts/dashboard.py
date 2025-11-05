import streamlit as st
import pandas as pd
import requests
import time
import io
import json

from typing import Optional
import numpy as np

# try both pickle and joblib
try:
    import joblib
except:
    joblib = None
import pickle

st.set_page_config(page_title="AI Alerts Dashboard", layout="wide")
st.title("🚨 AI Alerts Dashboard")
st.write("Upload CSV or fetch via API → load your trained model → get predictions & alerts.")

# ---------------------- Session Logs ----------------------
if "logs" not in st.session_state:
    st.session_state.logs = []

def add_log(msg: str):
    ts = time.strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{ts}] {msg}")

# ---------------------- Loaders ----------------------
def load_pickle(file) -> object:
    if joblib:
        try:
            return joblib.load(file)
        except:
            file.seek(0)
    file.seek(0)
    return pickle.load(file)

def ensure_dataframe(obj) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, list):
        return pd.DataFrame(obj)
    return pd.DataFrame(obj)

# ---------------------- Data Sources ----------------------
st.header("📥 Get Data")

left, right = st.columns(2)

with left:
    st.subheader("Upload CSV")
    csv_file = st.file_uploader("Choose a CSV file", type=["csv"], key="csv_upl")
    df: Optional[pd.DataFrame] = None
    if csv_file is not None:
        try:
            df = pd.read_csv(csv_file)
            st.success("CSV uploaded.")
            add_log(f"CSV loaded: {df.shape[0]} rows, {df.shape[1]} cols")
            st.dataframe(df.head(50), use_container_width=True)
        except Exception as e:
            st.error(f"CSV load error: {e}")
            add_log(f"CSV load error: {e}")

with right:
    st.subheader("Fetch from API")
    api_url = st.text_input("Enter API URL (returns JSON array/records)")
    if st.button("Fetch Dataset"):
        try:
            add_log("Fetching data from API...")
            t0 = time.time()
            resp = requests.get(api_url, timeout=30)
            resp.raise_for_status()
            data_json = resp.json()
            df_api = ensure_dataframe(data_json)
            if df is None:
                df = df_api
            else:
                st.info("Showing API data (separate from uploaded CSV).")
                st.dataframe(df_api.head(50), use_container_width=True)
            add_log(f"API OK {resp.status_code} in {time.time()-t0:.2f}s, rows={df_api.shape[0]}")
            st.success("API data fetched.")
        except Exception as e:
            st.error(f"API fetch error: {e}")
            add_log(f"API error: {e}")

st.divider()

# ---------------------- Model & Optional Scaler ----------------------
st.header("🧠 Load Model")
mcol1, mcol2 = st.columns(2)
with mcol1:
    model_file = st.file_uploader("Upload model file (.pkl/.joblib)", type=["pkl","joblib"], key="model_upl")
with mcol2:
    scaler_file = st.file_uploader("Upload optional preprocessor/scaler (.pkl/.joblib)", type=["pkl","joblib"], key="scaler_upl")

model = None
scaler = None
if model_file is not None:
    try:
        model = load_pickle(model_file)
        st.success("Model loaded.")
        add_log("Model loaded.")
    except Exception as e:
        st.error(f"Model load error: {e}")
        add_log(f"Model load error: {e}")

if scaler_file is not None:
    try:
        scaler = load_pickle(scaler_file)
        st.info("Scaler/Preprocessor loaded.")
        add_log("Scaler loaded.")
    except Exception as e:
        st.error(f"Scaler load error: {e}")
        add_log(f"Scaler load error: {e}")

st.divider()

# ---------------------- Feature Selection & Prediction ----------------------
st.header("🚦 Run Inference & Generate Alerts")

if (model is None) or (df is None):
    st.warning("Upload or fetch data AND load a model to proceed.")
else:
    # choose features
    numeric_cols = list(df.select_dtypes(include=['int64','float64','int32','float32']).columns)
    st.caption("Select feature columns for model input (default: numeric columns).")
    features = st.multiselect("Feature columns", options=list(df.columns), default=numeric_cols)

    # optional threshold (for classifiers with predict_proba)
    use_threshold = st.checkbox("Use probability threshold for HIGH alert (if supported)", value=True)
    threshold = st.slider("Alert threshold (probability of positive class)", 0.05, 0.95, 0.6, 0.01)

    # optional label/target column (if present in dataset but not for inference)
    drop_target = st.selectbox(
        "Drop target/label column from features (optional)",
        options=["<none>"] + list(df.columns),
        index=0
    )

    if st.button("🔮 Run Model"):
        try:
            X = df.copy()
            if drop_target != "<none>" and drop_target in X.columns:
                X = X.drop(columns=[drop_target])

            if features:
                X = X[features]

            # keep only numeric for most sklearn models
            X = X.select_dtypes(include=['int64','float64','int32','float32'])
            if X.shape[1] == 0:
                raise ValueError("No numeric feature columns available after selection.")

            if scaler is not None:
                X = scaler.transform(X)

            preds = None
            proba = None

            # try predict_proba
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)
                preds = np.argmax(proba, axis=1)
            else:
                preds = model.predict(X)

            out = df.copy()
            out["prediction"] = preds

            # add risk score from proba if available
            if proba is not None and proba.shape[1] >= 2:
                out["risk_score"] = proba[:, 1]
                if use_threshold:
                    out["alert"] = np.where(out["risk_score"] >= threshold, "HIGH", "LOW")
                else:
                    out["alert"] = "N/A"
            else:
                out["risk_score"] = np.nan
                out["alert"] = "N/A"

            st.success("Inference completed.")
            add_log("Inference completed.")

            st.subheader("📋 Predictions & Alerts")
            st.dataframe(out.head(200), use_container_width=True)

            # download results
            csv_buf = io.StringIO()
            out.to_csv(csv_buf, index=False)
            st.download_button("⬇️ Download predictions CSV", data=csv_buf.getvalue(),
                               file_name="predictions_with_alerts.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Inference error: {e}")
            add_log(f"Inference error: {e}")

st.divider()

# ---------------------- Logs Panel ----------------------
st.header("📝 Logs")
if len(st.session_state.logs) == 0:
    st.info("No logs yet…")
else:
    for log in st.session_state.logs[::-1]:
        st.write("•", log)

