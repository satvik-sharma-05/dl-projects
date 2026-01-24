import streamlit as st
import tempfile
from PIL import Image

from kidney_disease_classification.pipeline.prediction import PredictionPipeline

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Kidney CT Scan Classifier",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Kidney CT Scan Classification")
st.caption("AI-assisted screening tool (Educational use only)")

st.markdown("---")

# ---------------- USER GUIDANCE ----------------
st.info(
    """
**How predictions work:**

- ✅ **Confidence ≥ 70%** → Model prediction is shown (Normal / Tumor)
- ⚠️ **Confidence < 70%** → Marked as **Uncertain**
- This helps **reduce medical risk** and false confidence

⚠️ **This tool is NOT a medical diagnosis. Always consult a doctor.**
"""
)

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload a Kidney CT Scan image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=700)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.getbuffer())
        image_path = tmp.name


    st.markdown("---")
    st.subheader("🧪 Prediction Result")

    predictor = PredictionPipeline(image_path)
    result = predictor.predict()

    # ---------------- RESULTS ----------------
    confidence = result["confidence"]

    if result["prediction"] == "Tumor":
        st.error(f"🧠 Prediction: **Tumor**")
    elif result["prediction"] == "Normal":
        st.success(f"🧠 Prediction: **Normal**")
    else:
        st.warning("🧠 Prediction: **Uncertain**")

    st.metric("Confidence", f"{confidence}%")
    st.progress(int(confidence))

    # Show class probabilities
    with st.expander("🔍 Detailed Probabilities"):
        st.write(f"Normal: **{result['normal_prob']}%**")
        st.write(f"Tumor: **{result['tumor_prob']}%**")

    # ---------------- DISCLAIMER ----------------
    st.markdown("---")
    st.warning(
        """
⚠️ **Medical Disclaimer**

This AI model is trained on limited public data  
It **cannot replace professional medical diagnosis**  
Use this tool only for learning and experimentation
"""
    )
