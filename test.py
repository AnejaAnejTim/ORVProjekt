import os
import numpy as np
import cv2 as cv
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")
from urllib3.exceptions import NotOpenSSLWarning

warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

IMAGE_SIZE = (160, 160)
MODEL_PATH = "tim_andrejc_facenet.h5"
TEST_FOLDER = "Test"

model = load_model(MODEL_PATH)

def preprocess_image(path):
    try:
        img = cv.imread(path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.resize(img, IMAGE_SIZE)
        img = img.astype("float32") / 255.0
        return img
    except:
        print(f"Failed to load: {path}")
        return None

# Predict a single image
def predict_image(path):
    img = preprocess_image(path)
    if img is None:
        return None
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)[0][0]
    label = 1 if pred > 0.5 else 0
    confidence = pred if label == 1 else 1 - pred
    return label, confidence


# Test all images in the folder
def test_images_in_folder(folder):
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        if not os.path.isfile(path):
            continue

        result = predict_image(path)
        if result is None:
            continue

        label, confidence = result
        label_str = "Tim_Andrejc" if label == 1 else "Not Tim"
        print(f"{file}: {label_str} (confidence: {confidence:.2f})")

        # Optional: display image
        img = cv.imread(path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.title(f"{label_str} ({confidence:.2f})")
        plt.axis('off')
        plt.show()

# Run
test_images_in_folder(TEST_FOLDER)