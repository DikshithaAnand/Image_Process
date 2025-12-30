import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model (path from frontend folder)
model = tf.keras.models.load_model("../model/image_tagger.h5")

# Class names (must match training folders)
class_names = ['cat', 'dog']

# App title and description
st.title("🐱🐶 Image Tagging App")
st.write("Upload an image and the model will predict whether it is a cat or a dog.")

# Image uploader
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Open and show image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = prediction[0][predicted_index] * 100

    # Display result with UNKNOWN handling
    if confidence < 70:
        st.warning("⚠️ Unknown object (not cat or dog)")
        st.write(f"Confidence was only **{confidence:.2f}%**")
    else:
        st.success(f"Prediction: **{predicted_class.upper()}**")
        st.write(f"Confidence: **{confidence:.2f}%**")
