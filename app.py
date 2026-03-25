import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Kidney Stone Detector", page_icon="🔬")
st.title("🔬 Kidney Stone Detector")
st.write("Upload a kidney ultrasound image to detect kidney stones")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("fetus_ai_model.h5")

model = load_model()

# Upload image
uploaded_file = st.file_uploader("Choose an ultrasound image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", width=300)
    
    # Process image
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Show analyzing message
    with st.spinner("Analyzing image..."):
        # Predict
        prediction = model.predict(img_array)[0][0]
    
    # Show result
    if prediction > 0.5:
        confidence = round(prediction * 100, 2)
        st.error(f"⚠️ ABNORMAL - Kidney Stone Detected!\n\nConfidence: {confidence}%")
        st.warning("Please consult a doctor for medical advice.")
    else:
        confidence = round((1 - prediction) * 100, 2)
        st.success(f"✅ NORMAL - No Kidney Stone Detected!\n\nConfidence: {confidence}%")
        st.info("Regular check-ups are recommended.")