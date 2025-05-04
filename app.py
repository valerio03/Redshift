HEAD
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Set page config
st.set_page_config(page_title="RedShift Malaria Detection", layout="centered")

# Title
st.title("🧬 RedShift - Malaria Detection from RBC Images")
st.markdown("Upload an RBC image and let the model predict if it's **Parasitized** or **Uninfected**.")

# Load the model
@st.cache_resource

def load_model():
    model_path = "redshift.keras"
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

# Image preprocessing function
def preprocess_image(image):
    img = np.array(image)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Upload image
uploaded_file = st.file_uploader("Choose an RBC image (PNG or JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Predict button
    if st.button("🔍 Predict"):
        with st.spinner("Analyzing image..."):
            processed = preprocess_image(image)
            prediction = model.predict(processed)[0][0]
            label = "Parasitized" if prediction >= 0.5 else "Uninfected"
            confidence = prediction if prediction >= 0.5 else 1 - prediction

        st.success(f"**Prediction:** {label}")
        st.info(f"**Confidence:** {confidence * 100:.2f}%")

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Set page config
st.set_page_config(page_title="RedShift Malaria Detection", layout="centered")

# Title
st.title("🧬 RedShift - Malaria Detection from RBC Images")
st.markdown("Upload an RBC image and let the model predict if it's **Parasitized** or **Uninfected**.")

# Load the model
@st.cache_resource

def load_model():
    model_path = "redshift.keras"
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

# Image preprocessing function
def preprocess_image(image):
    img = np.array(image)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Upload image
uploaded_file = st.file_uploader("Choose an RBC image (PNG or JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Predict button
    if st.button("🔍 Predict"):
        with st.spinner("Analyzing image..."):
            processed = preprocess_image(image)
            prediction = model.predict(processed)[0][0]
            label = "Parasitized" if prediction >= 0.5 else "Uninfected"
            confidence = prediction if prediction >= 0.5 else 1 - prediction

        st.success(f"**Prediction:** {label}")
        st.info(f"**Confidence:** {confidence * 100:.2f}%")
 19de6cf595114dd7d68f547603a922041a713776
