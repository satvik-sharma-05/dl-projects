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
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.markdown("# 🍎 Fruit & Vegetable Classifier")
st.markdown("**CNN-based Image Classification Model**")
st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# -------------------- SIMPLE ABOUT --------------------
with st.expander("ℹ️ About this Project", expanded=True):
    st.markdown("""
    - **Model**: Custom CNN trained on 36 fruits & vegetables
    - **Tech Stack**: TensorFlow, Keras, Streamlit
    - **Purpose**: Demonstrate end-to-end deep learning application
    """)

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_cnn_model():
    return load_model("model/image_classifier.keras")

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

# -------------------- MAIN INTERFACE --------------------
st.markdown("### 📷 Upload Image")
st.markdown('<div class="upload-area">', unsafe_allow_html=True)
uploaded_image = st.file_uploader(
    "Drag and drop or click to browse",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)
st.markdown('</div>', unsafe_allow_html=True)

# -------------------- PREDICTION SECTION --------------------
if uploaded_image is not None:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Your Image", use_container_width=True)
    
    with col2:
        if st.button("🔍 Classify Image", type="primary"):
            # Preprocess image
            image_resized = image.resize((IMG_WIDTH, IMG_HEIGHT))
            img_arr = np.array(image_resized)
            img_bat = np.expand_dims(img_arr, 0)
            
            # Make prediction
            with st.spinner("Analyzing..."):
                predict = model.predict(img_bat)
                score = tf.nn.softmax(predict)
            
            predicted_class = data_cat[np.argmax(score)]
            confidence = np.max(score) * 100
            
            # Display results
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            st.markdown(f"### 🎯 **{predicted_class.capitalize()}**")
            st.markdown(f"**Confidence: {confidence:.1f}%**")
            
            # Confidence bar
            st.markdown(f"""
            <div style="width: 100%; background: rgba(255,255,255,0.2); border-radius: 10px;">
                <div class="confidence-bar" style="width: {confidence}%;"></div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Categories list
            with st.expander("📋 All Categories"):
                cols = st.columns(3)
                for i, cat in enumerate(data_cat):
                    cols[i % 3].write(f"• {cat}")

# -------------------- MINIMAL FOOTER --------------------
st.divider()
st.caption("Deep Learning Project | CNN Image Classification")