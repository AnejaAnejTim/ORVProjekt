import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Nastavitve - morajo biti enake kot pri treniranju
IMG_SIZE = 224  # Spremenjeno za nov model
TEST_DIR = 'Test'  # Mapa s testnimi slikami
MODEL_PATH = 'best_tim_detector.h5'  # Posodobljena pot do modela
CONFIDENCE_THRESHOLD = 0.5  # Prag za klasifikacijo (Tim/Ne Tim)


def load_and_preprocess_image(image_path):
    """Naloži in pripravi sliko za napovedovanje"""
    img = cv2.imread(image_path)
    if img is None:
        return None, None

    original_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Spremeni velikost na IMG_SIZE x IMG_SIZE
    img_resized = cv2.resize(original_img, (IMG_SIZE, IMG_SIZE))

    # Normaliziraj
    img_norm = img_resized.astype(np.float32) / 255.0

    return img_norm, original_img


def draw_bbox_on_image(img, bbox, class_prob, img_size_original):
    """Nariše bounding box na originalno sliko"""
    h_orig, w_orig = img_size_original[:2]

    # Pretvori normalizirane koordinate bbox v piksle glede na originalno velikost
    x_c, y_c, bw, bh = bbox
    x1 = int((x_c - bw / 2) * w_orig)
    y1 = int((y_c - bh / 2) * h_orig)
    x2 = int((x_c + bw / 2) * w_orig)
    y2 = int((y_c + bh / 2) * h_orig)

    # Preveri meje
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w_orig, x2)
    y2 = min(h_orig, y2)

    # Barva glede na verjetnost
    color = 'green' if class_prob > CONFIDENCE_THRESHOLD else 'red'
    confidence_text = f"Tim: {class_prob:.3f}" if class_prob > CONFIDENCE_THRESHOLD else f"Ne Tim: {1 - class_prob:.3f}"

    # Nariši bbox
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(img)

    # Bounding box
    rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=3,
                     edgecolor=color, facecolor='none')
    ax.add_patch(rect)

    # Tekst z verjetnostjo
    ax.text(x1, y1 - 10, confidence_text, color=color, fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

    ax.set_title(f"Rezultat: {'TIM ZAZNAN' if class_prob > CONFIDENCE_THRESHOLD else 'TIM NI ZAZNAN'}",
                 fontsize=14, fontweight='bold')
    ax.axis('off')

    return fig


def predict_single_image(model, image_path):
    """Testira eno sliko"""
    print(f"🔍 Testiram: {os.path.basename(image_path)}")

    # Naloži in pripravi sliko
    img_processed, img_original = load_and_preprocess_image(image_path)
    if img_processed is None:
        print(f"❌ Napaka pri nalaganju slike: {image_path}")
        return None

    # Napovej (dodaj batch dimenzijo)
    img_batch = np.expand_dims(img_processed, axis=0)
    prediction = model.predict(img_batch, verbose=0)[0]

    # Razdeli napoved na bbox in klasifikacijo
    bbox = prediction[:4]  # x_center, y_center, width, height
    class_prob = prediction[4]  # verjetnost da je Tim

    print(f"   📦 Bbox: [{bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f}]")
    print(f"   🎯 Verjetnost Tim: {class_prob:.3f}")
    print(f"   ✅ Rezultat: {'TIM' if class_prob > CONFIDENCE_THRESHOLD else 'NE TIM'}")

    return {
        'image_path': image_path,
        'bbox': bbox,
        'class_prob': class_prob,
        'is_tim': class_prob > CONFIDENCE_THRESHOLD,
        'original_image': img_original
    }


def main():
    print("🚀 Začenjam testiranje modela za detekcijo Tima")
    print(f"📁 Testna mapa: {TEST_DIR}")
    print(f"🤖 Model: {MODEL_PATH}")
    print(f"🎯 Prag zaupanja: {CONFIDENCE_THRESHOLD}")
    print("-" * 50)

    # Preveri ali obstajata mapa in model
    if not os.path.exists(TEST_DIR):
        print(f"❌ Mapa {TEST_DIR} ne obstaja!")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model {MODEL_PATH} ne obstaja!")
        return

    # Naloži model
    print("📥 Nalagam model...")
    try:
        # Definiraj custom loss funkcijo za nalaganje
        def improved_yolo_loss(y_true, y_pred):
            bbox_true = y_true[:, :4]
            bbox_pred = y_pred[:, :4]
            class_true = y_true[:, 4]
            class_pred = y_pred[:, 4]

            def compute_iou_loss(bbox_true, bbox_pred):
                true_x1 = bbox_true[:, 0] - bbox_true[:, 2] / 2
                true_y1 = bbox_true[:, 1] - bbox_true[:, 3] / 2
                true_x2 = bbox_true[:, 0] + bbox_true[:, 2] / 2
                true_y2 = bbox_true[:, 1] + bbox_true[:, 3] / 2

                pred_x1 = bbox_pred[:, 0] - bbox_pred[:, 2] / 2
                pred_y1 = bbox_pred[:, 1] - bbox_pred[:, 3] / 2
                pred_x2 = bbox_pred[:, 0] + bbox_pred[:, 2] / 2
                pred_y2 = bbox_pred[:, 1] + bbox_pred[:, 3] / 2

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

            coord_loss = tf.reduce_mean(tf.square(bbox_true[:, :2] - bbox_pred[:, :2]))
            size_loss = tf.reduce_mean(tf.square(
                tf.sqrt(tf.maximum(bbox_true[:, 2:4], 1e-8)) -
                tf.sqrt(tf.maximum(bbox_pred[:, 2:4], 1e-8))
            ))
            iou_loss = compute_iou_loss(bbox_true, bbox_pred)

            alpha = 0.25
            gamma = 2.0
            bce = tf.keras.losses.binary_crossentropy(class_true, class_pred)
            pt = tf.where(tf.equal(class_true, 1), class_pred, 1 - class_pred)
            focal_loss = alpha * tf.pow(1 - pt, gamma) * bce
            class_loss = tf.reduce_mean(focal_loss)

            coord_weight = 5.0
            size_weight = 3.0
            iou_weight = 2.0
            class_weight = 10.0

            total_loss = (coord_weight * coord_loss +
                          size_weight * size_loss +
                          iou_weight * iou_loss +
                          class_weight * class_loss)
            return total_loss

        def classification_accuracy(y_true, y_pred):
            class_true = y_true[:, 4]
            class_pred = y_pred[:, 4]
            predictions = tf.cast(class_pred > 0.5, tf.float32)
            return tf.reduce_mean(tf.cast(tf.equal(class_true, predictions), tf.float32))

        def bbox_mae(y_true, y_pred):
            bbox_true = y_true[:, :4]
            bbox_pred = y_pred[:, :4]
            return tf.reduce_mean(tf.abs(bbox_true - bbox_pred))

        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={
                'improved_yolo_loss': improved_yolo_loss,
                'classification_accuracy': classification_accuracy,
                'bbox_mae': bbox_mae
            }
        )
        print("✅ Model uspešno naložen!")
    except Exception as e:
        print(f"❌ Napaka pri nalaganju modela: {e}")
        return

    # Najdi vse slike v testni mapi
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = []

    for file in os.listdir(TEST_DIR):
        if file.lower().endswith(image_extensions):
            image_files.append(os.path.join(TEST_DIR, file))

    if not image_files:
        print(f"❌ V mapi {TEST_DIR} ni najdenih slik!")
        return

    print(f"📸 Najdeno {len(image_files)} slik za testiranje")
    print("-" * 50)

    # Testiraj vse slike
    results = []
    tim_count = 0

    for image_path in image_files:
        result = predict_single_image(model, image_path)
        if result:
            results.append(result)
            if result['is_tim']:
                tim_count += 1
        print()

    # Povzetek rezultatov
    # Povzetek rezultatov
    print("=" * 50)
    print("📊 POVZETEK REZULTATOV")
    print("=" * 50)
    print(f"📸 Skupaj testiranih slik: {len(results)}")
    print(f"✅ Slik z zaznanim Timom: {tim_count}")
    print(f"❌ Slik brez Tima: {len(results) - tim_count}")
    print()

    # Izpiši vse rezultate
    for result in results:
        filename = os.path.basename(result['image_path'])
        prob = result['class_prob']
        label = "TIM" if result['is_tim'] else "NE TIM"
        print(f"🖼️ {filename} - {label} (verjetnost: {prob:.3f})")

    print()
    print("📊 Vizualiziram vse slike...")

    for i, result in enumerate(results):
        fig = draw_bbox_on_image(
            result['original_image'],
            result['bbox'],
            result['class_prob'],
            result['original_image'].shape
        )
        plt.figure(fig.number)
        plt.suptitle(f"{i + 1}: {os.path.basename(result['image_path'])}", fontsize=16)
        plt.tight_layout()
        plt.show()

    print("✅ Testiranje in vizualizacija zaključena!")


if __name__ == "__main__":
    main()