import os
import cv2
import numpy as np
import random

INPUT_DIR = 'dataset/person'
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
IMG_SIZE = 224
AUGMENTATIONS = 4  # koliko novih primerkov na sliko

def detect_face(img):
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return faces

def crop_and_resize(img, face):
    x, y, w, h = face
    face_img = img[y:y+h, x:x+w]
    return cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))

def augment_image(img):
    img = img.astype(np.float32) / 255.0
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
    if random.random() < 0.3:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    if random.random() < 0.4:
        img = np.clip(img + np.random.normal(0, 0.05, img.shape), 0, 1)
    if random.random() < 0.4:
        factor = random.uniform(0.7, 1.3)
        img = np.clip(img * factor, 0, 1)
    if random.random() < 0.3:
        hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[..., 0] = (hsv[..., 0] + random.randint(-10, 10)) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] + random.randint(-30, 30), 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] + random.randint(-30, 30), 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0
    if random.random() < 0.3:
        h, w = img.shape[:2]
        mh, mw = random.randint(h//4, h//2), random.randint(w//4, w//2)
        x, y = random.randint(0, w - mw), random.randint(0, h - mh)
        img[y:y+mh, x:x+mw] = np.random.uniform(0, 1, (mh, mw, 3))
    return (img * 255).astype(np.uint8)

def process_directory():
    for fname in os.listdir(INPUT_DIR):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        path = os.path.join(INPUT_DIR, fname)
        img = cv2.imread(path)
        if img is None:
            continue

        faces = detect_face(img)
        if len(faces) == 0:
            continue

        # Obreži obraz in shrani pod istim imenom (prepiše original)
        face = crop_and_resize(img, faces[0])
        cv2.imwrite(path, face)

        # Augmentiraj in shrani dodatne primerke
        base_name, ext = os.path.splitext(fname)
        for i in range(AUGMENTATIONS):
            aug_img = augment_image(face.copy())
            aug_name = os.path.join(INPUT_DIR, f"{base_name}_{i+1}{ext}")
            cv2.imwrite(aug_name, aug_img)

if __name__ == "__main__":
    process_directory()
    print("✅ Obstoječe slike obrezane in prepisane, + dodane augmentacije.")
