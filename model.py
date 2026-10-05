"""Model loading, image preprocessing, and prediction helpers."""
from functools import lru_cache

import numpy as np
from keras.models import load_model
from keras.utils import img_to_array
from PIL import Image

MODEL_PATH = "digit_model.h5"
IMAGE_SIZE = (224, 224)


@lru_cache(maxsize=1)
def get_model():
    """Load the trained Keras model once and reuse it for later requests."""
    return load_model(MODEL_PATH)


def preprocess_img(img_path):
    """Open an image and convert it into a normalised (1, 224, 224, 3) array."""
    op_img = Image.open(img_path)
    img_resize = op_img.resize(IMAGE_SIZE)
    img2arr = img_to_array(img_resize) / 255.0
    img_reshape = img2arr.reshape(1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    return img_reshape


def predict_result(predict):
    """Return the index of the most likely digit class for a preprocessed image."""
    pred = get_model().predict(predict)
    return np.argmax(pred[0], axis=-1)
