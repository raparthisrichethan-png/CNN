import numpy as np
from PIL import Image


def preprocess_image(image):
    image = image.convert("L")
    image = image.resize((28, 28))

    img = np.array(image) / 255.0

    img = img.reshape(1, 28, 28, 1)

    return img