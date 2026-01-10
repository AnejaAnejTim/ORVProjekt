import os
import random
import shutil

# Mape
images_train_dir = "dataset/images/train"
images_val_dir = "dataset/images/val"
labels_train_dir = "dataset/labels/train"
labels_val_dir = "dataset/labels/val"

val_split = 0.2  # 20 % za validacijo

# 🔁 Počisti val mape
for folder in [images_val_dir, labels_val_dir]:
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

# 📸 Preberi slike iz train
slike = [f for f in os.listdir(images_train_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
random.shuffle(slike)

stevilo_val = int(len(slike) * val_split)
val_slike = slike[:stevilo_val]

# 🚚 Premakni slike in oznake
for slika in val_slike:
    # Slika
    src_img = os.path.join(images_train_dir, slika)
    dst_img = os.path.join(images_val_dir, slika)
    shutil.move(src_img, dst_img)

    # Oznaka
    ime_txt = os.path.splitext(slika)[0] + ".txt"
    src_lbl = os.path.join(labels_train_dir, ime_txt)
    dst_lbl = os.path.join(labels_val_dir, ime_txt)
    if os.path.exists(src_lbl):
        shutil.move(src_lbl, dst_lbl)

print(f"✅ Premaknil {stevilo_val} slik in pripadajočih .txt iz 'train' v 'val'.")
