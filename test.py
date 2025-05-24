import os
import numpy as np
import cv2 as cv
import tensorflow as tf
from tensorflow.keras.models import load_model
from tkinter import filedialog
from tkinter import Tk
import matplotlib.pyplot as plt

# Parametri
MODEL_PATH = "tim_andrejc_facenet.h5"
IMG_SIZE = (160, 160)
PRAG = 0.8

# Naloži model
model = load_model(MODEL_PATH)

# Izberi sliko prek GUI
def izberi_sliko():
    root = Tk()
    root.withdraw()
    pot = filedialog.askopenfilename(title="Izberi sliko za preverjanje",
                                     filetypes=[("Slike", "*.png *.jpg *.jpeg")])
    root.destroy()
    return pot

# Predprocesiranje slike
def nalozi_in_pripravi(pot):
    img = cv.imread(pot)
    if img is None:
        raise ValueError("Neveljavna slika.")
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

# Glavna logika
if __name__ == "__main__":
    pot_slike = izberi_sliko()

    if not pot_slike:
        print("Slika ni bila izbrana.")
        exit()

    try:
        slika = nalozi_in_pripravi(pot_slike)
        verjetnost = model.predict(slika)[0][0]
        print(f"Verjetnost, da je to Tim Andrejc: {verjetnost * 100:.2f}%")

        plt.imshow(slika[0])
        plt.axis("off")
        plt.title(f"{'✅ ISTA OSEBA' if verjetnost >= PRAG else '❌ NI ISTA OSEBA'}\n({verjetnost*100:.1f}%)")
        plt.show()

    except Exception as e:
        print(f"Napaka: {e}")
