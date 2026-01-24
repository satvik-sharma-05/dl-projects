import streamlit as st
import tempfile
from PIL import Image

from kidney_disease_classification.pipeline.prediction import PredictionPipeline

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="KidneyScan AI",
    page_icon="🩺",
    layout="centered"
)

# ---------------- HEADER ----------------
st.markdown(
    """
    <h1 style="text-align:center;">🩺 KidneyScan AI</h1>
    <p style="text-align:center;">
    AI-based Kidney CT Scan Classification (Educational Use)
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# ---------------- INFO BOX ----------------
st.info("""
### 🔍 How Prediction Works
- The model predicts **Normal** or **Tumor**
- Confidence ≥ **70%** → Reliable prediction
- Confidence < **70%** → **Uncertain (manual review recommended)**

⚠️ This tool is **NOT for medical diagnosis**
""")

# ---------------- UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload a Kidney CT Scan Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=350)

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.getbuffer())
        image_path = tmp.name

    # ---------------- PREDICTION ----------------
    with st.spinner("Analyzing image..."):
        predictor = PredictionPipeline(image_path)
        result = predictor.predict()

    prediction = result["prediction"]
    confidence = result["confidence"]

    st.divider()

    # ---------------- RESULT CARD ----------------
    if confidence >= 70:
        if prediction == "Tumor":
            st.error(f"🧠 **Prediction:** Tumor\n\n📊 **Confidence:** {confidence}%")
        else:
            st.success(f"🧠 **Prediction:** Normal\n\n📊 **Confidence:** {confidence}%")
    else:
        st.warning(
            f"""
            ⚠️ **Uncertain Prediction**
            
            - Model confidence is **{confidence}% (<70%)**
            - Image may be unclear or ambiguous
            - Manual review recommended
            """
        )

    # ---------------- PROBABILITIES ----------------
    st.subheader("📈 Class Probabilities")
    st.progress(result["normal_prob"] / 100)
    st.write(f"Normal: **{result['normal_prob']}%**")

    st.progress(result["tumor_prob"] / 100)
    st.write(f"Tumor: **{result['tumor_prob']}%**")

# ---------------- FOOTER ----------------
st.divider()
st.caption(
    "⚠️ Educational project only • Not a medical device • Built with TensorFlow & Streamlit"
)
