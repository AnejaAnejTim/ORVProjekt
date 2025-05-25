import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageOps
import cv2 as cv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

# Parameters
BATCH_SIZE =32
EPOCHS = 50
LEARNING_RATE = 0.0001
IMG_SIZE = 120

# Load image paths
tim_images = glob.glob(os.path.join("Tim_Andrejc", "**", "*.*"), recursive=True)
drugi_images = glob.glob(os.path.join("drugo", "**", "*.*"), recursive=True)

# Undersample "drugi" class to match closer balance
random.seed(42)
drugi_images = random.sample(drugi_images, min(len(tim_images) * 2, len(drugi_images)))

# Labels: 1 for Tim, 0 for others
all_images = tim_images + drugi_images
all_labels = [1] * len(tim_images) + [0] * len(drugi_images)

# Train/Val split
X_train, X_val, y_train, y_val = train_test_split(
    all_images, all_labels, test_size=0.2, stratify=all_labels, random_state=42
)

pos = sum(y_train)
neg = len(y_train) - pos
weight_for_0 = (1 / neg) * (len(y_train)) / 2.0
weight_for_1 = (1 / pos) * (len(y_train)) / 2.0
class_weight = {0: weight_for_0, 1: weight_for_1}

# Load and preprocess image
def nalozi_sliko(pot):
    try:
        img = cv.imread(pot)
        img = cv.resize(img, (IMG_SIZE, IMG_SIZE))
        return img
    except:
        return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

# Augment image
def augmentiraj(img):
    pil = Image.fromarray(img.astype('uint8'))
    if random.random() > 0.8:
        pil = pil.rotate(random.uniform(-90, 90), resample=Image.BICUBIC)
    if random.random() > 0.8:
        pil = ImageEnhance.Brightness(pil).enhance(random.uniform(0.5, 1.4))
    if random.random() > 0.6:
        pil = ImageOps.mirror(pil)
    if random.random() > 0.5:
        pil = ImageOps.flip(pil)
    return np.array(pil)

# Data generator yielding images, labels, and sample weights
def generator(image_paths, labels, batch_size, augment=False, shuffle=True, class_weight=None):
    while True:
        idxs = np.arange(len(image_paths))
        if shuffle:
            np.random.shuffle(idxs)
        for i in range(0, len(image_paths), batch_size):
            batch_idxs = idxs[i:i + batch_size]
            batch_images, batch_labels, batch_weights = [], [], []
            for j in batch_idxs:
                img = nalozi_sliko(image_paths[j])
                if augment:
                    img = augmentiraj(img)
                img = img.astype('float32') / 255.0
                batch_images.append(img)
                batch_labels.append(labels[j])
                if class_weight is not None:
                    batch_weights.append(class_weight[labels[j]])
                else:
                    batch_weights.append(1.0)
            yield np.array(batch_images), np.array(batch_labels).reshape(-1, 1), np.array(batch_weights)

# Build the model
def zgradi_model():
    model = models.Sequential()
    model.add(layers.Conv2D(16, (5, 5), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(36, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))  # binary output
    model.compile(optimizer=optimizers.Adam(LEARNING_RATE),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

# Prepare generators with class weights for training
train_gen = generator(X_train, y_train, BATCH_SIZE, augment=True, class_weight=class_weight)
val_gen = generator(X_val, y_val, BATCH_SIZE, augment=False, shuffle=False)

# Train model (note: do NOT pass class_weight argument here)
model = zgradi_model()
history = model.fit(
    train_gen,
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=val_gen,
    validation_steps=len(X_val) // BATCH_SIZE,
    callbacks=[
        callbacks.EarlyStopping(patience=6, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-6)
    ]
)

# Save model
model.save("TimAndrejc.keras")

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

print("\nClassification Report on Validation Data:")
y_true, y_pred = [], []
for path, label in zip(X_val, y_val):
    img = nalozi_sliko(path).astype('float32') / 255.0
    pred = model.predict(np.expand_dims(img, axis=0))[0][0]
    y_true.append(label)
    y_pred.append(1 if pred > 0.3 else 0)  # custom threshold

print(classification_report(y_true, y_pred, target_names=["drugo", "Tim_Andrejc"]))