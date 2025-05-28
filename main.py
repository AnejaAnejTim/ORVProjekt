import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import cv2
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input

# ===== CONFIG =====
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.0001
DATASET_DIR = 'dataset'

# ===== ASK TO PREPARE IMAGES =====
if input("🔄 Želiš zagnati 'prepareImages.py' za obrezovanje & augmentacijo? (da/ne): ").lower() == "da":
    os.system("python prepareImages.py")


# ===== LOAD DATA =====
def load_images_from_folder(folder_path, label):
    images, labels = [], []
    if not os.path.exists(folder_path): return np.array([]), np.array([])

    for img_file in os.listdir(folder_path):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')): continue
        img_path = os.path.join(folder_path, img_file)
        img = cv2.imread(img_path)
        if img is None: continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = preprocess_input(img.astype(np.float32), version=2)
        images.append(img)
        labels.append(label)
    return np.array(images), np.array(labels)


def load_data():
    pos, l1 = load_images_from_folder(os.path.join(DATASET_DIR, 'person'), 1.0)
    neg, l0 = load_images_from_folder(os.path.join(DATASET_DIR, 'others'), 0.0)
    return np.concatenate([pos, neg]), np.concatenate([l1, l0])


def data_generator(images, labels, batch_size, augment=False):
    while True:
        idxs = np.random.permutation(len(images))
        for i in range(0, len(images), batch_size):
            batch_idxs = idxs[i:i + batch_size]
            batch_imgs, batch_labels = [], []
            for j in batch_idxs:
                img = images[j]
                label = labels[j]
                batch_imgs.append(img)
                batch_labels.append(label)
            yield np.array(batch_imgs), np.array(batch_labels)


# ===== MODEL DEFINITION (VGGFace ResNet50) =====
def build_vggface_model():
    base_model = VGGFace(model='resnet50', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling='avg')
    for layer in base_model.layers:
        layer.trainable = False  # Zamrzni feature extractor

    x = base_model.output
    x = layers.Dense(256, activation='swish')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='swish')(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=base_model.input, outputs=output)
    return model


def focal_loss(y_true, y_pred, alpha=0.8, gamma=2.0):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    bce = - (y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
    loss = alpha * tf.pow(1 - y_pred, gamma) * bce
    return tf.reduce_mean(loss)


# ===== LOAD DATA =====
images, labels = load_data()
train_imgs, val_imgs, train_lbls, val_lbls = train_test_split(images, labels, test_size=0.2, stratify=labels,
                                                              random_state=42)

print(f"✅ Train: {len(train_imgs)}, Val: {len(val_imgs)}")

# ===== GENERATORS =====
train_gen = data_generator(train_imgs, train_lbls, BATCH_SIZE)
val_gen = data_generator(val_imgs, val_lbls, BATCH_SIZE)

steps_per_epoch = len(train_imgs) // BATCH_SIZE
val_steps = len(val_imgs) // BATCH_SIZE

# ===== COMPILE AND TRAIN =====
model = build_vggface_model()
model.compile(
    optimizer=optimizers.Adam(LEARNING_RATE),
    loss=focal_loss,
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

cb = [
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4),
    callbacks.EarlyStopping(monitor='val_auc', patience=8, mode='max', restore_best_weights=True),
    callbacks.ModelCheckpoint('best_model_vggface.h5', monitor='val_auc', save_best_only=True, mode='max')
]

print("🚀 Treniram model...")
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    callbacks=cb
)

model.save("vggface_final_model.h5")
model.summary()

# ===== VIZUALIZACIJA =====
plt.figure(figsize=(12, 6))
for i in range(min(8, len(train_imgs))):
    plt.subplot(2, 4, i + 1)
    img = train_imgs[i][:, :, ::-1]  # RGB to BGR
    label = "PERSON" if train_lbls[i] == 1 else "OTHER"
    plt.imshow(img.astype(np.uint8))
    plt.title(label)
    plt.axis('off')
plt.tight_layout()
plt.show()
