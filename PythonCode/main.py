import os
from pathlib import Path
from types import ModuleType
import cv2
import sys
import numpy as np
import random
import subprocess
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import shutil
import requests
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from decompress_lib import decompress_file
from sklearn.model_selection import train_test_split
try:
    from keras_vggface.vggface import VGGFace
    from keras_vggface.utils import preprocess_input
except Exception:
    import keras
    # ensure modules exist so "from keras.engine.topology import get_source_inputs" succeeds
    sys.modules.setdefault('keras.engine', ModuleType('keras.engine'))
    sys.modules.setdefault('keras.engine.topology', ModuleType('keras.engine.topology'))
    # map the newer TF-Keras helper into the expected location if available
    try:
        from tensorflow.keras.utils import get_source_inputs
        setattr(sys.modules['keras.engine.topology'], 'get_source_inputs', get_source_inputs)
    except Exception:
        pass
    # now import VGGFace and preprocess_input
    from keras_vggface.vggface import VGGFace
    from keras_vggface.utils import preprocess_input

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
INCOMING_FOLDER = os.path.join(os.getcwd(), 'incoming')
OUTPUT_FOLDER = os.path.join(os.getcwd(), 'output')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(INCOMING_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
IMG_SIZE = 224
AUGMENTATIONS = 4
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.0001


def detect_face(img):
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return faces


def crop_and_resize(img, face):
    x, y, w, h = face
    face_img = img[y:y + h, x:x + w]
    return cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))


import numpy as np
import random

def augment_image(img):
    # pretvori v float [0, 1] za enostavnejšo obdelavo
    img = img.astype(np.float32) / 255.0 #normalizacija

    # horizontalno zrcaljenje (z verjetnostjo 50 %)
    if random.random() < 0.5:
        img = img[:, ::-1, :] #horizontalno

    # gaussov blur s 3x3 jedrom (verjetnost 20 %)
    if random.random() < 0.2:
        #3x3 Gauss blur
        kernel = np.array([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]], dtype=np.float32)
        kernel /= kernel.sum()

        h, w, c = img.shape
        padded = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode='reflect')
        blurred = np.zeros_like(img)

        # aplicira gaussov blur na vsako barvno komponento
        for i in range(h):
            for j in range(w):
                for k in range(c):
                    region = padded[i:i+3, j:j+3, k]
                    blurred[i, j, k] = np.sum(region * kernel)
        img = blurred

    # doda naključni Gaussian šum (verjetnost 20 %)
    if random.random() < 0.2:
        noise = np.random.normal(0, 0.02, img.shape)
        img = np.clip(img + noise, 0, 1)

    # naključna sprememba svetlosti (verjetnost 30 %)
    if random.random() < 0.3:
        factor = random.uniform(0.8, 1.2)
        img = np.clip(img * factor, 0, 1)

    # naključna sprememba kontrasta (verjetnost 20 %)
    if random.random() < 0.2:
        v = random.randint(-15, 15) / 255.0
        img = np.clip(img + v, 0, 1)

    # eandom cutout (prekrije del slike z naključnim pravokotnikom – verjetnost 10 %)
    if random.random() < 0.1:
        #random cutout
        h, w = img.shape[:2]
        mh = random.randint(h // 8, h // 4) #visina
        mw = random.randint(w // 8, w // 4) #sirina
        x = random.randint(0, w - mw) # naključni x koordinat
        y = random.randint(0, h - mh) # naključni y koordinat
        # luknjo zapolni z naključnimi vrednostmi
        img[y:y + mh, x:x + mw] = np.random.uniform(0, 1, (mh, mw, 3))

    return (img * 255).astype(np.uint8)

def process_directory(input_dir):
    print(f"Processing images in {input_dir}")
    
    # Check for .bin files
    bin_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.bin')]
    
    if len(bin_files) > 0:
        print(f"Found {len(bin_files)} .bin files. Using MPI for parallel decompression...")
        try:
            # Determine current script directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            mpi_script = os.path.join(current_dir, 'mpi_decompress.py')
            
            # Prepare MPI command
            cmd = ['mpiexec', '-n', '2', 'python', mpi_script, input_dir, input_dir]
            print(f"DEBUG: Preparing MPI command: {' '.join(cmd)}")
            print(f"Executing MPI: {' '.join(cmd)}")
            
            # Use subprocess.run to wait for completion
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=current_dir)
            
            if result.returncode == 0:
                print("MPI Decompression logs:")
                print(result.stdout)
            else:
                print("MPI Decompression failed:")
                print(result.stderr)
                # Fallback to sequential decompression if MPI fails
                print("Falling back to sequential decompression...")
                for fname in bin_files:
                    path = os.path.join(input_dir, fname)
                    output_name = os.path.splitext(fname)[0] + ".bmp"
                    output_path = os.path.join(input_dir, output_name)
                    decompress_file(path, output_path)
        except Exception as e:
            print(f"Error during MPI decompression: {e}")
            # Fallback
            for fname in bin_files:
                path = os.path.join(input_dir, fname)
                output_name = os.path.splitext(fname)[0] + ".bmp"
                output_path = os.path.join(input_dir, output_name)
                try: decompress_file(path, output_path)
                except: pass

    face_count = 0
    total_images = 0

    for fname in os.listdir(input_dir):
        path = os.path.join(input_dir, fname)
        
        # We already handled decompression above via MPI or fallback
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
        
        total_images += 1
        print(f"Detecting faces in {fname}...")
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: unable to read image {fname}")
            continue

        faces = detect_face(img)
        if len(faces) == 0:
            print(f"No faces detected in {fname}")
            continue

        face_count += 1
        face = crop_and_resize(img, faces[0])
        cv2.imwrite(path, face)

        base_name, ext = os.path.splitext(fname)
        for i in range(AUGMENTATIONS):
            aug_img = augment_image(face.copy())
            aug_name = os.path.join(input_dir, f"{base_name}_{i + 1}{ext}")
            cv2.imwrite(aug_name, aug_img)

    return face_count, total_images


def load_images_from_folder(folder_path, label):
    images, labels = [], []
    if not os.path.exists(folder_path):
        return np.array([]), np.array([])

    for img_file in os.listdir(folder_path):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
        img_path = os.path.join(folder_path, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = preprocess_input(img.astype(np.float32), version=2)
        images.append(img)
        labels.append(label)

    return np.array(images), np.array(labels)


def load_data(user_folder):
    user_path = Path(user_folder)
    if not user_path.exists():
        raise FileNotFoundError(f"The directory {user_folder} does not exist.")

    user_imgs, user_lbls = load_images_from_folder(user_folder, 1.0)

    others_folder = os.path.join(UPLOAD_FOLDER, 'others')
    others_imgs, others_lbls = load_images_from_folder(others_folder, 0.0)

    if len(user_imgs) > 0 and len(others_imgs) > 0:
        images = np.concatenate([user_imgs, others_imgs], axis=0)
        labels = np.concatenate([user_lbls, others_lbls], axis=0)
    elif len(user_imgs) > 0:
        images, labels = user_imgs, user_lbls
    elif len(others_imgs) > 0:
        images, labels = others_imgs, others_lbls
    else:
        images, labels = np.array([]), np.array([])

    return images, labels


def data_generator(images, labels, batch_size):
    while True:
        # naključno premešaj indekse vseh slik
        idxs = np.random.permutation(len(images))
        #deli premešane podatke na zaporedne batch-e velikosti batch_size
        for i in range(0, len(images), batch_size):
            batch_idxs = idxs[i:i + batch_size]
            #inicializira prazna seznama za slike in oznake trenutnega batch-a
            batch_imgs = []
            batch_labels = []

            #doda slike in oznake v trenutni batch
            for j in batch_idxs:
                batch_imgs.append(images[j])
                batch_labels.append(labels[j])

            # vrni (yield) trenutni batch kot numpy array (slike, oznake)
            yield np.array(batch_imgs), np.array(batch_labels)


def build_vggface_model():
    base_model = VGGFace(model='resnet50', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling='avg')

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = layers.Dense(256, activation='swish')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='swish')(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation='sigmoid')(x)

    return models.Model(inputs=base_model.input, outputs=output)

#zmanjša vpliv lahko klasificiranih primerov, poudari težke
def focal_loss(y_true, y_pred, alpha=0.8, gamma=2.0):
    # poskrbi, da napovedi ne vsebujejo ekstremnih vrednosti, ki bi povzročile log(0)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    # binary cross-entropy formula
    bce = - (y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
    # Focal loss zmanjša vpliv lahko klasificiranih primerov (tisti z visoko verjetnostjo),
    # in poudari težko klasificirane (tam kjer je model negotov)
    loss = alpha * tf.pow(1 - y_pred, gamma) * bce
    # vrne povprečno izgubo čez vse primere
    return tf.reduce_mean(loss)


def train_model(user_folder):
    print("Starting model training...")
    # naloži podatke (slike in pripadajoče oznake) iz uporabniške mape
    images, labels = load_data(user_folder)

    if len(images) == 0:
        print("No images found for training.")
        return False

    #preveri, če imata oba razreda (pozitivni in negativni) vsaj en primer
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        print("Need both positive and negative samples for training.")
        return False

    #razdeli slike na trenirni in validacijski set (z ohranjanjem razmerja razredov)
    train_imgs, val_imgs, train_lbls, val_lbls = train_test_split(
        images, labels, test_size=0.2, stratify=labels, random_state=42
    )

    print(f"Train samples: {len(train_imgs)}, Validation samples: {len(val_imgs)}")
    print(f"Train positive: {np.sum(train_lbls)}, Train negative: {len(train_lbls) - np.sum(train_lbls)}")
    print(f"Val positive: {np.sum(val_lbls)}, Val negative: {len(val_lbls) - np.sum(val_lbls)}")

    #ustvari generatorje podatkov za treniranje in validacijo
    train_gen = data_generator(train_imgs, train_lbls, BATCH_SIZE)
    val_gen = data_generator(val_imgs, val_lbls, BATCH_SIZE)

    # izračun število korakov na epoh
    steps_per_epoch = max(1, len(train_imgs) // BATCH_SIZE)
    val_steps = max(1, len(val_imgs) // BATCH_SIZE)

    #zgradi model in ga skompilira z izbrano funkcijo izgube in metrikami
    model = build_vggface_model()
    model.compile(
        optimizer=optimizers.Adam(LEARNING_RATE),
        loss=focal_loss,
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    model_path = 'vggface_final_model.h5'

    #callback-i za zgodnje ustavljanje, prilagoditev LR in shranjevanje najboljšega modela
    cb = [
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4),
        callbacks.EarlyStopping(monitor='val_auc', patience=8, mode='max', restore_best_weights=True),
        callbacks.ModelCheckpoint(model_path, monitor='val_auc', save_best_only=True, mode='max')
    ]

    #dejanski zagon treniranja
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        callbacks=cb
    )

    # shrani končni model
    model.save(model_path)
    print(f"Training complete. Model saved at {model_path}")
    return True


@app.route('/register-face', methods=['POST'])
def register_face():
    print("Received request at /register-face")
    if 'email' not in request.form or 'username' not in request.form:
        print("Error: Email or username not provided")
        return jsonify({'success': False, 'message': 'Email or username not provided'}), 400

    email = request.form['email']
    username = request.form['username']
    print(f"Registration for user: {username} ({email})")
    sanitized_email = ''.join(c if c.isalnum() or c in ['@', '.', '_', '-'] else '_' for c in email)
    user_folder = os.path.join(UPLOAD_FOLDER, sanitized_email)

    try:
        print(f"Checking if user {email} exists...")
        response = requests.post('http://192.168.0.13:3001/users/userExists',
                                 json={'email': email, 'username': username}, timeout=5)
        response.raise_for_status()
        if response.json().get('exists', False):
            print(f"User {email} already exists")
            return jsonify({'success': False, 'message': 'User already exists'}), 401
    except requests.RequestException as e:
        print(f"Error checking user existence: {e}")
        return jsonify({'success': False, 'message': f'User existence check failed: {str(e)}'}), 500

    # if os.path.exists(user_folder):
    #     print(f"Cleaning up existing folder for {email}")
    #     shutil.rmtree(user_folder)
    os.makedirs(user_folder, exist_ok=True)

    if 'images' not in request.files:
        print("Error: No images uploaded")
        return jsonify({'success': False, 'message': 'No images uploaded'}), 400

    files = request.files.getlist('images')
    print(f"Received {len(files)} files")
    if not files:
        print("Error: No images found in request")
        return jsonify({'success': False, 'message': 'No images found in request'}), 400

    try:
        for file in files:
            filename = secure_filename(file.filename)
            file_path = os.path.join(user_folder, filename)
            print(f"Saving file: {filename} to {file_path}")
            file.save(file_path)
            # Ensure file is written to disk
            with open(file_path, 'rb+') as f:
                f.flush()
                os.fsync(f.fileno())

        print("Starting image processing...")
        face_count, total_images = process_directory(user_folder)
        print(f"Processing result: {face_count}/{total_images} faces detected")
        
        if total_images == 0:
            print("Error: No valid images processed")
            # shutil.rmtree(user_folder)
            return jsonify({'success': False, 'message': 'No valid images processed. Registration failed.'}), 400
            
        if face_count < total_images / 2:
            print(f"Error: Too few faces detected ({face_count}/{total_images})")
            # shutil.rmtree(user_folder)
            return jsonify({'success': False, 'message': f'Too few faces detected ({face_count}/{total_images}). Registration failed.'}), 400

        print("Starting model training...")
        if not train_model(user_folder):
            print("Error: Model training failed")
            return jsonify({'success': False, 'message': 'Training failed due to insufficient data or error during training'}), 500

        print("Registration successful")
        return jsonify({'success': True, 'message': 'Faces processed and model trained successfully'}), 200
    except Exception as e:
        import traceback
        print(f"Error during face registration: {e}")
        traceback.print_exc()
        # shutil.rmtree(user_folder, ignore_errors=True)
        return jsonify({'success': False, 'message': f'Server error during face registration: {str(e)}'}), 500


@app.route('/authenticateFace', methods=['POST'])
def authenticate_face():
    print("Received request at /authenticateFace")
    if 'email' not in request.form or 'image' not in request.files:
        print("Error: Email or image not provided")
        return jsonify({'success': False, 'message': 'Email or image not provided'}), 400

    email = request.form['email']
    file = request.files['image']
    filename = secure_filename(file.filename)
    print(f"Authentication request for user: {email}, file: {filename}")
    sanitized_email = ''.join(c if c.isalnum() or c in ['@', '.', '_', '-'] else '_' for c in email)
    user_folder = os.path.join(UPLOAD_FOLDER, sanitized_email)
    os.makedirs(user_folder, exist_ok=True)
    model_path = 'vggface_final_model.h5'

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return jsonify({'success': False, 'message': 'User model not found'}), 404

    temp_img_path = os.path.join(user_folder, filename)
    print(f"Saving temporary image to {temp_img_path}")
    file.save(temp_img_path)

    try:
        if filename.lower().endswith('.bin'):
            print(f"Decompressing {filename} for authentication (Color)...")
            output_name = os.path.splitext(filename)[0] + ".bmp"
            output_path = os.path.join(user_folder, output_name)
            decompress_file(temp_img_path, output_path)
            # os.remove(temp_img_path)
            temp_img_path = output_path
            print(f"Decompression successful (Color), new path: {temp_img_path}")

        print("Reading image with OpenCV...")
        img = cv2.imread(temp_img_path)
        if img is None:
            print(f"Error: Could not read image at {temp_img_path}")
            return jsonify({'success': False, 'message': 'Invalid image uploaded or decompression failed'}), 400

        print("Detecting faces...")
        faces = detect_face(img)
        if len(faces) == 0:
            print("Error: No face detected in authentication image")
            return jsonify({'success': False, 'message': 'No face detected'}), 400

        print(f"Detected {len(faces)} face(s). Preprocessing...")
        # Consistent preprocessing with training
        face = crop_and_resize(img, faces[0])
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # Convert to RGB
        face = preprocess_input(face.astype(np.float32), version=2)  # VGGFace preprocessing
        face = np.expand_dims(face, axis=0)

        print("Loading model and predicting...")
        model = tf.keras.models.load_model(model_path, custom_objects={'focal_loss': focal_loss})
        prediction = model.predict(face)[0][0]
        print(f"Prediction score: {prediction}")

        if prediction > 0.8:
            print("Authentication successful")
            return jsonify({'success': True, 'message': 'Face authenticated', 'confidence': float(prediction)}), 200
        else:
            print(f"Authentication failed (confidence: {prediction})")
            return jsonify({'success': False, 'message': 'Authentication failed', 'confidence': float(prediction)}), 401

    except Exception as e:
        import traceback
        print(f"Error loading or using model: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Internal server error during authentication: {str(e)}'}), 500
    finally:
        # if os.path.exists(temp_img_path):
        #     try:
        #         os.remove(temp_img_path)
        #         print(f"Removed temporary image: {temp_img_path}")
        #     except Exception as e:
        #         print(f"Warning: Could not remove temporary image: {e}")
        pass


@app.route('/decompress', methods=['POST'])
def trigger_decompression():
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'}), 400
    
    if file and file.filename.endswith('.bin'):
        filename = secure_filename(file.filename)
        input_path = os.path.join(INCOMING_FOLDER, filename)
        file.save(input_path)

        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            mpi_script = os.path.join(current_dir, 'mpi_decompress.py')
            
            cmd = ['mpiexec', '-n', '2', 'python', mpi_script, INCOMING_FOLDER, OUTPUT_FOLDER]
            print(f"Executing MPI job: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=current_dir)
            
            if result.returncode == 0:
                print("MPI job finished successfully")
                print(result.stdout)
                return jsonify({'success': True, 'message': 'Decompression completed', 'stdout': result.stdout}), 200
            else:
                print(f"MPI job failed with return code {result.returncode}")
                print(result.stderr)
                return jsonify({'success': False, 'message': 'Decompression failed', 'stderr': result.stderr}), 500
                
        except Exception as e:
            print(f"Error spawning MPI job: {e}")
            return jsonify({'success': False, 'message': f'Error spawning MPI job: {str(e)}'}), 500
    else:
        return jsonify({'success': False, 'message': 'Invalid file type. Only .flocic supported.'}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)