import os

import numpy as np
import tensorflow as tf
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.keras")
CLASS_FILE = os.path.join(BASE_DIR, "data", "class_names.txt")
IMG_SIZE = (160, 160)

# Load once when the app starts, not every upload
model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_FILE, encoding="utf-8") as f:
    CLASS_NAMES = [line.strip() for line in f.read().splitlines() if line.strip()]

def prepare_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict(image_path):
    arr = prepare_image(image_path)
    probs = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], round(float(probs[idx]), 2)

if __name__ == "__main__":
    print(predict(os.path.join(BASE_DIR, "static", "uploads", "test.jpg")))