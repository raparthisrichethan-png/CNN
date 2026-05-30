import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import fashion_mnist

st.set_page_config(
    page_title="CNN Fashion Classifier",
    layout="wide"
)

st.title("Fashion Classification using CNN")

labels = [
    "T-shirt",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot"
]

model = load_model("model/cnn_model.h5", compile=False)

(_, _), (X_test, y_test) = fashion_mnist.load_data()

X_eval = X_test.reshape(-1, 28, 28, 1) / 255.0

# Manual Accuracy
predictions = model.predict(X_eval, verbose=0)
pred_classes = np.argmax(predictions, axis=1)

accuracy = np.mean(pred_classes == y_test)

st.success(f"Model Accuracy: {accuracy:.4f}")

# Image Selector
index = st.slider(
    "Select Test Image",
    0,
    len(X_test) - 1,
    0
)

image = X_test[index]

st.image(
    image,
    caption=f"Actual: {labels[y_test[index]]}",
    width=300
)

img = image.reshape(1, 28, 28, 1) / 255.0

prediction = model.predict(img, verbose=0)

predicted_class = np.argmax(prediction)

st.success(f"Predicted: {labels[predicted_class]}")