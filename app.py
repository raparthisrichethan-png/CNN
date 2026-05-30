import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np

# Load trained model
model = keras.models.load_model("cnn_pet_classifier.h5")

# App title
st.title("🐶🐱 Cat vs Dog Classifier")
st.write("Upload an image and the CNN model will predict whether it is a Cat or Dog.")

# Image size used during training
IMG_SIZE = (128, 128)

# Function for preprocessing image

def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    image = np.array(image)

    # Convert grayscale image to RGB if needed
    if image.shape[-1] == 4:
        image = image[:, :, :3]

    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    return image

# File uploader
uploaded_file = st.file_uploader(
    "Upload a Cat or Dog Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file)

    # Display image
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    processed_image = preprocess_image(image)

    # Prediction
    prediction = model.predict(processed_image)

    # Display prediction
    if prediction[0][0] > 0.5:
        st.success("Prediction: 🐶 Dog")
        st.write(f"Confidence: {prediction[0][0] * 100:.2f}%")
    else:
        st.success("Prediction: 🐱 Cat")
        st.write(f"Confidence: {(1 - prediction[0][0]) * 100:.2f}%")
