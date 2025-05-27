import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from keras_vggface.utils import preprocess_input

# Nastavitve
IMG_SIZE = 224
TEST_DIR = 'Test'
MODEL_PATH = 'vggface_final_model.h5'  # Model from training script
CONFIDENCE_THRESHOLD = 0.5
VISUALIZE_RESULTS = True

# Naloži Haar cascade za obraz (v paketu OpenCV)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def focal_loss(y_true, y_pred, alpha=0.8, gamma=2.0):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    bce = - (y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
    loss = alpha * tf.pow(1 - y_pred, gamma) * bce
    return tf.reduce_mean(loss)

def detect_and_crop_face(img):
    """Detektira obraz na sliki in vrne obrezano področje obraza.
    Če ni najden obrazov, vrne None."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None
    # Izberi največji obraz (če je več obrazov)
    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
    # Prilagodi bbox, da malo povečamo okvir okoli obraza (neobvezno)
    pad = int(0.15 * w)
    x_new = max(x - pad, 0)
    y_new = max(y - pad, 0)
    w_new = min(w + 2 * pad, img.shape[1] - x_new)
    h_new = min(h + 2 * pad, img.shape[0] - y_new)
    cropped_face = img[y_new:y_new + h_new, x_new:x_new + w_new]
    return cropped_face

def load_and_preprocess_image(image_path):
    """Naloži, croppa na obraz in pripravi sliko za predikcijo z VGGFace preprocess_input v2"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Napaka: slika ni bila uspešno prebrana: {image_path}")
        return None
    face = detect_and_crop_face(img)
    if face is None:
        print(f"⚠️ Opozorilo: Obraz ni bil najden na sliki: {image_path} — uporaba cele slike")
        face = img  # fallback: uporabi celo sliko, če obraz ni najden
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    face = face.astype(np.float32)
    face = preprocess_input(face, version=2)  # VGGFace ResNet50 preprocess
    return face

def predict_single_image(model, image_path):
    img = load_and_preprocess_image(image_path)
    if img is None:
        return None
    img_batch = np.expand_dims(img, axis=0)
    try:
        person_prob = model.predict(img_batch, verbose=0)[0][0]
        is_target = person_prob > CONFIDENCE_THRESHOLD
        return {
            'image_path': image_path,
            'person_prob': float(person_prob),
            'is_target': is_target,
            'image_array': img  # For visualization
        }
    except Exception as e:
        print(f"❌ Napaka pri predikciji za {image_path}: {e}")
        return None

def visualize_results(results):
    for r in results:
        plt.figure(figsize=(4, 4))
        # Convert back from preprocess_input to displayable image (approximate)
        img_display = (r['image_array'] + 1.0) * 127.5  # preprocess_input v2 centers ~ [-1,1]
        img_display = np.clip(img_display, 0, 255).astype(np.uint8)
        plt.imshow(img_display)
        title = f"{'✅ TARGET' if r['is_target'] else '❌ OTHER'}\nVerjetnost: {r['person_prob']:.3f}"
        color = 'green' if r['is_target'] else 'red'
        plt.title(title, color=color)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

def main():
    print("🚀 Začenjam testiranje facial recognition modela")
    print(f"   Model: {MODEL_PATH}")
    print(f"   Prag zaupanja: {CONFIDENCE_THRESHOLD}")

    if not os.path.exists(TEST_DIR):
        print(f"❌ Mapa {TEST_DIR} ne obstaja!")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model {MODEL_PATH} ne obstaja!")
        return

    tf.keras.backend.clear_session()

    print("📥 Nalagam model...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'focal_loss': focal_loss})
        print("✅ Model uspešno naložen!")
        print(f"   Arhitektura: {len(model.layers)} slojev")
        print(f"   Parametri: {model.count_params():,}")
    except Exception as e:
        print(f"❌ Napaka pri nalaganju modela: {e}")
        return

    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    image_files = [os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR)
                   if f.lower().endswith(image_extensions)]

    if not image_files:
        print(f"❌ Ni najdenih slik v mapi {TEST_DIR}")
        return

    print(f"📸 Najdenih {len(image_files)} slik za testiranje")
    print("=" * 50)

    results = []
    target_count = 0

    for i, path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] Testiram: {os.path.basename(path)}", end=" ... ")
        result = predict_single_image(model, path)
        if result:
            results.append(result)
            if result['is_target']:
                target_count += 1
                print(f"✅ TARGET (verjetnost: {result['person_prob']:.3f})")
            else:
                print(f"❌ OTHER (verjetnost: {result['person_prob']:.3f})")
        else:
            print("❌ NAPAKA")

    print("\n📊 PODROBNI REZULTATI")
    print("=" * 60)
    print("{:<30} | {:<12} | {:<6} | {}".format("Slika", "Klasifikacija", "Verj.", "Zaupanje"))
    print("-" * 60)

    results_sorted = sorted(results, key=lambda x: x['person_prob'], reverse=True)
    for r in results_sorted:
        fname = os.path.basename(r['image_path'])
        label = "✅ TARGET" if r['is_target'] else "❌ OTHER"
        confidence = "🔥 VISOKA" if r['person_prob'] > 0.8 else (
            "🔶 SREDNJA" if r['person_prob'] > 0.5 else "🔻 NIZKA")
        print(f"{fname:<30} | {label:<12} | {r['person_prob']:.3f} | {confidence}")

    print("\n📋 POVZETEK REZULTATOV")
    print("=" * 40)
    print(f"Skupaj testiranih slik: {len(results)}")
    print(f"Zaznanih ciljnih oseb (TARGET): {target_count}")
    print(f"Ostali (OTHER): {len(results) - target_count}")

    if VISUALIZE_RESULTS and results:
        print("\n🖼️ Vizualizacija rezultatov...")
        visualize_results(results)

    print("\n✅ Testiranje zaključeno!")

if __name__ == "__main__":
    main()
