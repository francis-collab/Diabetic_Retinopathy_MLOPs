import streamlit as st
import requests
import os
from PIL import Image

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Diabetic Retinopathy MLOps", page_icon="👁️", layout="wide")
st.title("👁️ Diabetic Retinopathy Severity Classifier – MLOps Pipeline")
st.markdown("""
**Use case continuation** from the previous Diabetes Risk Prediction project.  
This pipeline classifies **fundus images** into 5 severity levels (No_DR → Proliferate_DR) to support early screening tools for diabetic patients.
""")

# Sidebar health / uptime
st.sidebar.header("Model Status / Uptime")
try:
    h = requests.get(f"{API_URL}/health", timeout=5).json()
    st.sidebar.success(f"Status: {h.get('status')}")
    st.sidebar.metric("Uptime (seconds)", h.get("uptime_seconds", 0))
    st.sidebar.write(f"Model loaded: {h.get('model_loaded')}")
    st.sidebar.write(f"IMG_SIZE: {h.get('img_size', 'N/A')}")
except Exception as e:
    st.sidebar.error(f"API offline: {e}")

tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict", "📊 Visualizations", "📤 Upload & Retrain", "ℹ️ About"])

with tab1:
    st.header("Single Image Prediction")
    st.write("Upload a fundus (retinal) image. The model returns one of: No_DR, Mild, Moderate, Severe, Proliferate_DR.")
    uploaded = st.file_uploader("Fundus image", type=["png", "jpg", "jpeg"])
    if uploaded:
        st.image(Image.open(uploaded), caption="Uploaded fundus", width=280)
        if st.button("Predict Severity", type="primary"):
            files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
            try:
                r = requests.post(f"{API_URL}/predict", files=files, timeout=30)
                if r.status_code == 200:
                    data = r.json()
                    st.success(f"**Predicted severity: {data['prediction']}**  (confidence {data['confidence']:.1%})")
                    st.json(data.get("probabilities", {}))
                else:
                    st.error(r.text)
            except Exception as e:
                st.error(str(e))

with tab2:
    st.header("Dataset Insights & Visualizations")
    try:
        stats = requests.get(f"{API_URL}/dataset_stats", timeout=10).json()
        st.subheader("Class distribution")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.write("**Train**")
            st.bar_chart(stats.get("train", {}))
        with c2:
            st.write("**Test**")
            st.bar_chart(stats.get("test", {}))
        with c3:
            st.write("**Retrain buffer**")
            st.bar_chart(stats.get("retrain_buffer", {}))

        st.markdown("""
        ### Interpretation of key features / story the data tells
        1. **Severity imbalance** – In real APTOS-style datasets the majority of images are “No_DR” or “Mild”. This mirrors real screening populations and forces the model to handle class imbalance (we use `class_weight='balanced'` and transfer learning).
        2. **Lesion patterns** – Higher severity classes contain more hemorrhages, exudates and neovascularization. Pixel-level (CNN / MobileNetV2) features capture these visual markers that a clinician would look for.
        3. **Clinical link to previous project** – The tabular diabetes risk model predicted *probability of developing diabetes*. This image model detects an actual *complication* (retinopathy). Together they form a more complete screening story: risk score → if high, recommend fundus photo → severity grade → referral decision.
        """)
        metrics = requests.get(f"{API_URL}/metrics", timeout=5).json()
        if metrics and "message" not in metrics:
            st.subheader("Experiment results (from notebook)")
            st.json(metrics)
    except Exception as e:
        st.warning(f"Could not load stats (is the API running?): {e}")

with tab3:
    st.header("Upload New Fundus Images & Trigger Retraining")
    st.info(
        "1. Upload one or more fundus images  \n"
        "2. Provide matching labels (comma-separated, exact class names)  \n"
        "3. Click **Upload to Retrain Buffer**  \n"
        "4. Click **Trigger Retraining** (fine-tunes the Advanced CNN)"
    )
    files = st.file_uploader("New fundus images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    labels = st.text_input(
        "Labels (exact class names, same order)",
        placeholder="No_DR,Mild,Moderate,Severe,Proliferate_DR",
    )
    if st.button("Upload to Retrain Buffer"):
        if files and labels:
            multipart = [("files", (f.name, f.getvalue(), f.type)) for f in files]
            try:
                r = requests.post(
                    f"{API_URL}/upload_retrain",
                    files=multipart,
                    data={"labels": labels},
                    timeout=120,
                )
                st.write(r.json())
            except Exception as e:
                st.error(str(e))
        else:
            st.warning("Provide both files and labels")

    if st.button("🚀 Trigger Retraining", type="primary"):
        try:
            with st.spinner("Fine-tuning Advanced CNN (this may take a few minutes)..."):
                r = requests.post(f"{API_URL}/retrain", params={"epochs": 5}, timeout=600)
            st.write(r.json())
            if r.status_code == 200 and r.json().get("status") == "success":
                st.balloons()
        except Exception as e:
            st.error(str(e))

with tab4:
    st.markdown("""
    ## Project Overview
    Full MLOps pipeline for **Diabetic Retinopathy severity classification** from fundus images.

    - Builds on the previous **Diabetes Risk Prediction** use case (tabular → now image-based complication detection)
    - Uses a real Kaggle-style medical image dataset (APTOS 2019 style)
    - 6 modelling experiments with regularization, optimizers, early-stopping, class weighting, transfer learning
    - Best model: **Advanced CNN (MobileNetV2 transfer learning)** – highest ROC-AUC
    - FastAPI + Streamlit UI with model uptime, visualizations, single prediction, bulk upload + retrain button
    - Docker-ready, Locust load-test ready

    **Severity classes**: No_DR · Mild · Moderate · Severe · Proliferate_DR
    """)
