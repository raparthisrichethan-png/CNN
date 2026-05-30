import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fashion Classification using CNN")

st.title("Fashion Classification using CNN")

# Load model WITHOUT compile
model = load_model("model/cnn_model.keras", compile=False)

# Recompile correctly
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Load dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

st.success(f"Model Accuracy: {accuracy:.4f}")

class_names = [
    "T-shirt/Top",
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

# Random sample prediction
idx = np.random.randint(0, len(X_test))
sample = X_test[idx]

prediction = model.predict(sample.reshape(1, 28, 28, 1), verbose=0)
predicted_class = np.argmax(prediction)

st.subheader("Prediction Result")
st.write(f"Predicted Class: **{class_names[predicted_class]}**")

fig, ax = plt.subplots()
ax.imshow(sample.reshape(28, 28), cmap="gray")
st.pyplot(fig)