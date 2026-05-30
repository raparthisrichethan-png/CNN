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

model = load_model("model/cnn_model.h5")

(_, _), (X_test, y_test) = fashion_mnist.load_data()

# Model Accuracy
X_eval = X_test.reshape(-1, 28, 28, 1) / 255.0

loss, accuracy = model.evaluate(
    X_eval,
    y_test,
    verbose=0
)

st.success(f"Model Accuracy: {accuracy:.4f}")

# Test image selector
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