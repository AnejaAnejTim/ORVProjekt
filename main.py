import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

# Configuration
gpu_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available:", len(gpu_devices))

BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 1e-4
IMG_SIZE = (160, 160)
CLASS_NAMES = ["drugo", "Tim_Andrejc"]

# 1) Load file paths and labels
def load_image_paths_and_labels(base_dir):
    paths, labels = [], []
    for idx, cls in enumerate(CLASS_NAMES):
        cls_dir = os.path.join(base_dir, cls)
        imgs = glob.glob(os.path.join(cls_dir, "*.jpg")) + glob.glob(os.path.join(cls_dir, "*.png"))
        paths += imgs
        labels += [idx] * len(imgs)
    return paths, labels

all_paths, all_labels = load_image_paths_and_labels('.')

# Balance classes via weights
class_weights = class_weight.compute_class_weight(
    'balanced', classes=np.unique(all_labels), y=all_labels
)
class_weights_dict = dict(enumerate(class_weights))

# Split dataset
train_paths, val_paths, train_labels, val_labels = train_test_split(
    all_paths, all_labels, test_size=0.2, stratify=all_labels, random_state=42
)

# 2) Preprocessing + Augmentation using tf.data API
# Define a keras augmentation layer
augmentation_layer = tf.keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.15),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2)
])

def preprocess_and_augment(path, label, training=True):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = image / 255.0
    if training:
        image = augmentation_layer(image)
    return image, label

train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_ds = (
    train_ds.shuffle(len(train_paths))
            .map(lambda p, l: preprocess_and_augment(p, l, True), num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE)
)

val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_ds = (
    val_ds.map(lambda p, l: preprocess_and_augment(p, l, False), num_parallel_calls=tf.data.AUTOTUNE)
          .batch(BATCH_SIZE)
          .prefetch(tf.data.AUTOTUNE)
)

# 3) Build model with pretrained MobileNetV2 backbone
base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
base_model.trainable = False

inputs = layers.Input(shape=IMG_SIZE + (3,))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs, outputs)
model.compile(
    optimizer=optimizers.Adam(LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)
model.summary()

# 4) Callbacks
cb = [
    callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-6)
]

# 5) Train model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights_dict,
    callbacks=cb
)

# 6) Unfreeze some layers and fine-tune
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=optimizers.Adam(LEARNING_RATE / 10),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

fine_history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    class_weight=class_weights_dict,
    callbacks=cb
)

# 7) Save and plot results
model.save('tim_andrejc_facenet.h5')

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'] + fine_history.history['loss'], label='train')
plt.plot(history.history['val_loss'] + fine_history.history['val_loss'], label='val')
plt.title('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'] + fine_history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'] + fine_history.history['val_accuracy'], label='val')
plt.title('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('training_results.png')
plt.show()
