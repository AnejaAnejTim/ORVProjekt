import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import cv2
import random
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models

# Nastavitve
IMG_SIZE = 224  # Povečano za boljše rezultate
BATCH_SIZE = 16  # Zmanjšano za stabilnost
EPOCHS = 60
LEARNING_RATE = 0.00007
DATASET_DIR = 'dataset'


# Funkcija za nalaganje slik in oznak - BREZ obrezovanja
def load_data(images_dir, labels_dir):
    images = []
    targets = []
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    tim_count = 0
    not_tim_count = 0

    for img_file in image_files:
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)
        if not os.path.exists(label_path):
            print(f"⚠️ Preskočena slika brez oznake: {img_file}")
            continue

        img_path = os.path.join(images_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Napaka pri branju slike: {img_file}")
            continue

        with open(label_path, 'r') as f:
            line = f.readline().strip()
            if not line:
                print(f"⚠️ Prazna oznaka v datoteki: {label_file}")
                continue
            parts = line.split()
            if len(parts) != 5:
                print(f"⚠️ Napačen format oznake v datoteki: {label_file}")
                continue
            class_id = int(parts[0])  # 0 = Ne Tim, 1 = Tim
            bbox = list(map(float, parts[1:]))

        # POMEMBNO: Ne obrezuj slike, samo jo rescale!
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # Konvertiraj iz BGR v RGB in normaliziraj
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0

        images.append(img_norm)

        # Target ostane bbox + class
        class_prob = float(class_id)
        target = bbox + [class_prob]
        targets.append(target)

        if class_id == 1:
            tim_count += 1
        else:
            not_tim_count += 1

    print(f"📊 Naloženo: {len(images)} slik (Tim: {tim_count}, Ne Tim: {not_tim_count})")
    return np.array(images, dtype=np.float32), np.array(targets, dtype=np.float32)


# Poenostavljene augmentacije - samo tiste ki ne vplivajo na bbox
def augment_image_simple(img):
    # Brightness & contrast
    if random.random() < 0.7:
        factor = random.uniform(0.4, 1.6)
        img = np.clip(img * factor, 0, 1)

    # Gaussian blur
    if random.random() < 0.4:
        ksize = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)

    # Add Gaussian noise
    if random.random() < 0.5:
        noise = np.random.normal(0, 0.08, img.shape)
        img = np.clip(img + noise, 0, 1)

    # Color jitter (Hue/Saturation shift)
    if random.random() < 0.4:
        hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        hsv = hsv.astype(np.int16)
        hsv[..., 0] = (hsv[..., 0] + random.randint(-15, 15)) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] + random.randint(-50, 50), 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] + random.randint(-50, 50), 0, 255)
        hsv = hsv.astype(np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB) / 255.0

    # Invert colors
    if random.random() < 0.3:
        img = 1.0 - img

    # Pixelation
    if random.random() < 0.4:
        h, w = img.shape[:2]
        scale = random.uniform(0.1, 0.3)
        small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        img = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    # Random blackout (rectangle)
    if random.random() < 0.4:
        h, w = img.shape[:2]
        rect_w = random.randint(w // 8, w // 3)
        rect_h = random.randint(h // 8, h // 3)
        x1 = random.randint(0, w - rect_w)
        y1 = random.randint(0, h - rect_h)
        img[y1:y1+rect_h, x1:x1+rect_w] = 0.0

    # CLAHE (adaptive histogram equalization on L-channel)
    if random.random() < 0.3:
        lab = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)
        lab_eq = cv2.merge((l_eq, a, b))
        img = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0

    # Channel shuffle
    if random.random() < 0.3:
        channels = list(cv2.split(img))
        random.shuffle(channels)
        img = cv2.merge(channels)

    return img

def visualize_batch(images, targets, batch_size=8):
    plt.figure(figsize=(16, 12))
    for i in range(min(batch_size, len(images))):
        img = images[i]
        bbox = targets[i][:4]
        class_prob = targets[i][4]

        h, w = img.shape[:2]
        x_c, y_c, bw, bh = bbox
        x1 = int((x_c - bw / 2) * w)
        y1 = int((y_c - bh / 2) * h)
        x2 = int((x_c + bw / 2) * w)
        y2 = int((y_c + bh / 2) * h)

        # Preveri meje
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        plt.subplot(2, 4, i + 1)
        plt.imshow(img)
        color = 'g' if class_prob > 0.5 else 'r'
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor='none')
        plt.gca().add_patch(rect)
        plt.title(
            f"Class: {'Tim' if class_prob > 0.5 else 'Not Tim'}\nBbox: [{x_c:.2f}, {y_c:.2f}, {bw:.2f}, {bh:.2f}]")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# Generator za batch-e
def data_generator(images, targets, batch_size, augment=False):
    while True:
        idxs = np.arange(len(images))
        np.random.shuffle(idxs)
        for i in range(0, len(images), batch_size):
            batch_idxs = idxs[i:i + batch_size]
            batch_imgs = []
            batch_targets = []

            for idx in batch_idxs:
                img = images[idx].copy()
                target = targets[idx].copy()

                if augment:
                    img = augment_image_simple(img)

                batch_imgs.append(img)
                batch_targets.append(target)

            yield np.array(batch_imgs, dtype=np.float32), np.array(batch_targets, dtype=np.float32)


# Izboljšan model z boljšo arhitekturo
def build_improved_model():
    # Uporabi EfficientNet ali MobileNet namesto ResNet50
    base_model = tf.keras.applications.MobileNetV2(
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        weights='imagenet'
    )

    # Najprej zamrzni, potem odtajaj zadnje sloje
    base_model.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)

    # Dodaj več slojev za boljše učenje
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    # Ločeni sloji za bbox in klasifikacijo
    bbox_branch = layers.Dense(64, activation='relu')(x)
    bbox_output = layers.Dense(4, activation='sigmoid', name='bbox')(bbox_branch)

    class_branch = layers.Dense(32, activation='relu')(x)
    class_output = layers.Dense(1, activation='sigmoid', name='classification')(class_branch)

    outputs = layers.Concatenate()([bbox_output, class_output])
    model = models.Model(inputs=base_model.input, outputs=outputs)
    return model, base_model


# Izboljšana loss funkcija
def improved_yolo_loss(y_true, y_pred):
    bbox_true = y_true[:, :4]
    bbox_pred = y_pred[:, :4]
    class_true = y_true[:, 4]
    class_pred = y_pred[:, 4]

    # IoU loss za bounding box
    def compute_iou_loss(bbox_true, bbox_pred):
        # Pretvori iz center format v corner format
        true_x1 = bbox_true[:, 0] - bbox_true[:, 2] / 2
        true_y1 = bbox_true[:, 1] - bbox_true[:, 3] / 2
        true_x2 = bbox_true[:, 0] + bbox_true[:, 2] / 2
        true_y2 = bbox_true[:, 1] + bbox_true[:, 3] / 2

        pred_x1 = bbox_pred[:, 0] - bbox_pred[:, 2] / 2
        pred_y1 = bbox_pred[:, 1] - bbox_pred[:, 3] / 2
        pred_x2 = bbox_pred[:, 0] + bbox_pred[:, 2] / 2
        pred_y2 = bbox_pred[:, 1] + bbox_pred[:, 3] / 2

        # Izračunaj IoU
        inter_x1 = tf.maximum(true_x1, pred_x1)
        inter_y1 = tf.maximum(true_y1, pred_y1)
        inter_x2 = tf.minimum(true_x2, pred_x2)
        inter_y2 = tf.minimum(true_y2, pred_y2)

        inter_area = tf.maximum(0.0, inter_x2 - inter_x1) * tf.maximum(0.0, inter_y2 - inter_y1)

        true_area = (true_x2 - true_x1) * (true_y2 - true_y1)
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        union_area = true_area + pred_area - inter_area

        iou = inter_area / (union_area + 1e-8)
        return 1.0 - tf.reduce_mean(iou)

    # Coordinate loss (center points)
    coord_loss = tf.reduce_mean(tf.square(bbox_true[:, :2] - bbox_pred[:, :2]))

    # Size loss
    size_loss = tf.reduce_mean(tf.square(
        tf.sqrt(tf.maximum(bbox_true[:, 2:4], 1e-8)) -
        tf.sqrt(tf.maximum(bbox_pred[:, 2:4], 1e-8))
    ))

    # IoU loss
    iou_loss = compute_iou_loss(bbox_true, bbox_pred)

    # Fokusna izguba za klasifikacijo (za neuravnotežene razrede)
    alpha = 0.25
    gamma = 2.0

    bce = tf.keras.losses.binary_crossentropy(class_true, class_pred)
    pt = tf.where(tf.equal(class_true, 1), class_pred, 1 - class_pred)
    focal_loss = alpha * tf.pow(1 - pt, gamma) * bce
    class_loss = tf.reduce_mean(focal_loss)

    # Uteži
    coord_weight = 5.0
    size_weight = 3.0
    iou_weight = 2.0
    class_weight = 10.0

    total_loss = (coord_weight * coord_loss +
                  size_weight * size_loss +
                  iou_weight * iou_loss +
                  class_weight * class_loss)

    return total_loss


# Metrike
def classification_accuracy(y_true, y_pred):
    class_true = y_true[:, 4]
    class_pred = y_pred[:, 4]
    predictions = tf.cast(class_pred > 0.5, tf.float32)
    return tf.reduce_mean(tf.cast(tf.equal(class_true, predictions), tf.float32))


def bbox_mae(y_true, y_pred):
    bbox_true = y_true[:, :4]
    bbox_pred = y_pred[:, :4]
    return tf.reduce_mean(tf.abs(bbox_true - bbox_pred))


# --- Glavni del ---
if __name__ == "__main__":
    print("📊 Nalaganje podatkov...")

    train_images, train_targets = load_data(
        os.path.join(DATASET_DIR, 'images/train'),
        os.path.join(DATASET_DIR, 'labels/train')
    )
    val_images, val_targets = load_data(
        os.path.join(DATASET_DIR, 'images/val'),
        os.path.join(DATASET_DIR, 'labels/val')
    )

    print(f"✅ Train: {len(train_images)} slik")
    print(f"✅ Val: {len(val_images)} slik")

    # Preveri če imamo dovolj podatkov
    if len(train_images) < 10:
        print("❌ Premalo training podatkov! Potrebujete vsaj 10 slik.")
        exit()

    print("🔧 Ustvarjanje izboljšanega modela...")
    model, base_model = build_improved_model()
    print(f"📈 Model ima {model.count_params():,} parametrov")

    # Kompajliraj model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=improved_yolo_loss,
        metrics=[classification_accuracy, bbox_mae]
    )

    # Callback-i
    callbacks_list = [
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=7,
            min_lr=1e-7,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            'best_tim_detector.h5',
            monitor='val_classification_accuracy',
            save_best_only=True,
            verbose=1,
            mode='max'
        ),
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            verbose=1,
            restore_best_weights=True
        )
    ]

    # Generatorji
    train_gen = data_generator(train_images, train_targets, BATCH_SIZE, augment=True)
    val_gen = data_generator(val_images, val_targets, BATCH_SIZE, augment=False)

    steps_per_epoch = max(1, len(train_images) // BATCH_SIZE)
    validation_steps = max(1, len(val_images) // BATCH_SIZE)

    print(f"🚀 Začenjam učenje za {EPOCHS} epoch...")
    print(f"   - Korakov na epoch: {steps_per_epoch}")
    print(f"   - Validacijskih korakov: {validation_steps}")

    # Prvi faza: Učenje s zamrznjenimi sloji
    print("\n🎯 FAZA 1: Učenje z zamrznjenimi base sloji")
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS // 2,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks_list,
        verbose=1
    )

    # Druga faza: Fine-tuning z odtajanimi zadnjimi sloji
    print("\n🎯 FAZA 2: Fine-tuning z odtajanimi sloji")
    base_model.trainable = True

    # Zamrzni prve sloje, odtajaj zadnje
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    # Zmanjšaj learning rate za fine-tuning
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE / 10),
        loss=improved_yolo_loss,
        metrics=[classification_accuracy, bbox_mae]
    )

    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS // 2,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks_list,
        verbose=1
    )

    # Shrani končni model
    model.save('tim_detector_final.h5')
    print("✅ Model shranjen kot 'tim_detector_final.h5'")
    print("✅ Najboljši model shranjen kot 'best_tim_detector.h5'")

    # Prikaži rezultate
    best_val_acc = max(max(history1.history.get('val_classification_accuracy', [0])),
                       max(history2.history.get('val_classification_accuracy', [0])))
    print(f"📊 Najbolša validacijska točnost: {best_val_acc:.4f}")

    # Prikaži primer batch-a
    print("\n📊 Prikazujem primer batch-a...")
    gen = data_generator(train_images, train_targets, batch_size=8, augment=False)
    batch_imgs, batch_targets = next(gen)
    visualize_batch(batch_imgs, batch_targets)