import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from PIL import Image

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Fruit & Vegetable Classifier",
    page_icon="🥦",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding-top: 0rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 25px;
        color: white;
        margin: 20px 0;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    .upload-area {
        border: 3px dashed #4CAF50;
        border-radius: 10px;
        padding: 40px;
        text-align: center;
        margin: 30px 0;
        background: #f8f9fa;
    }
    .confidence-bar {
        height: 20px;
        background: linear-gradient(90deg, #4CAF50 0%, #8BC34A 100%);
        border-radius: 10px;
        margin: 10px 0;
    }
    .stButton button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 15px;
        border-radius: 10px;
        font-size: 16px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.markdown("# 🍎 Fruit & Vegetable Classifier")
st.markdown("**CNN-based Image Classification Web App**")
st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# -------------------- ABOUT SECTION --------------------
with st.expander("ℹ️ About this Project", expanded=True):
    st.markdown("""
**What is CNN?**  
A **Convolutional Neural Network (CNN)** is a deep learning model designed to
automatically learn visual features such as edges, textures, and shapes from images.

**What this app does**
- Accepts an image of a fruit or vegetable
- Processes it using a trained CNN
- Predicts the correct category with confidence

**Model Details**
- Custom CNN architecture
- Trained on **36 fruit & vegetable classes**
- Input size: `180 × 180 RGB`

**Tech Stack**
- TensorFlow & Keras (Model)
- Streamlit (UI & Deployment)
""")

# -------------------- LOAD MODEL (DEPLOY SAFE) --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "image_classifier.keras")

@st.cache_resource
def load_cnn_model():
    return load_model(MODEL_PATH)

model = load_cnn_model()

# -------------------- CLASS LABELS --------------------
data_cat = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum',
    'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant',
    'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango',
    'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate',
    'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato',
    'tomato', 'turnip', 'watermelon'
]

IMG_HEIGHT = 180
IMG_WIDTH = 180

# -------------------- IMAGE UPLOAD --------------------
st.markdown("### 📷 Upload an Image")
st.markdown('<div class="upload-area">', unsafe_allow_html=True)

uploaded_image = st.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

st.markdown('</div>', unsafe_allow_html=True)

# -------------------- PREDICTION --------------------
if uploaded_image is not None:
    col1, col2 = st.columns(2)

    with col1:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        if st.button("🔍 Classify Image"):
            image_resized = image.resize((IMG_WIDTH, IMG_HEIGHT))
            img_arr = np.array(image_resized)
            img_bat = np.expand_dims(img_arr, 0)

            with st.spinner("Analyzing image using CNN..."):
                prediction = model.predict(img_bat)
                score = tf.nn.softmax(prediction)

            predicted_class = data_cat[np.argmax(score)]
            confidence = np.max(score) * 100

            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            st.markdown(f"### 🎯 Prediction: **{predicted_class.capitalize()}**")
            st.markdown(f"**Confidence: {confidence:.2f}%**")

            st.markdown(f"""
            <div style="width:100%; background:rgba(255,255,255,0.2); border-radius:10px;">
                <div class="confidence-bar" style="width:{confidence}%;"></div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

            with st.expander("📋 Supported Categories"):
                cols = st.columns(3)
                for i, cat in enumerate(data_cat):
                    cols[i % 3].write(f"• {cat}")

# -------------------- FOOTER --------------------
st.divider()
st.caption("🚀 Deep Learning Project | CNN Image Classification | Streamlit Deployment")
