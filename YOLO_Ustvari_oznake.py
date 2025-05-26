import os
import cv2
from mtcnn import MTCNN

# Poti
images_folder = "dataset/images/train"
labels_folder = "dataset/labels/train"
if os.path.exists(labels_folder):
    import shutil
    shutil.rmtree(labels_folder)
os.makedirs(labels_folder, exist_ok=True)

detector = MTCNN()

def to_yolo_bbox(x, y, w, h, img_width, img_height):
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    width = w / img_width
    height = h / img_height
    return x_center, y_center, width, height

slike = [f for f in os.listdir(images_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for ime in slike:
    pot_slike = os.path.join(images_folder, ime)
    pot_oznaka = os.path.join(labels_folder, os.path.splitext(ime)[0] + ".txt")

    img = cv2.cvtColor(cv2.imread(pot_slike), cv2.COLOR_BGR2RGB)
    h_img, w_img = img.shape[:2]

    detections = detector.detect_faces(img)

    if len(detections) == 0:
        print(f"⚠️ Noben obraz ni zaznan v: {ime}, oznake ni ustvarjena.")
        continue

    with open(pot_oznaka, 'w') as f:
        for det in detections:
            x, y, w, h = det['box']
            # včasih so x,y negativni, popravimo:
            x = max(0, x)
            y = max(0, y)
            x_c, y_c, w_r, h_r = to_yolo_bbox(x, y, w, h, w_img, h_img)
            f.write(f"0 {x_c:.6f} {y_c:.6f} {w_r:.6f} {h_r:.6f}\n")

print("✅ Vse oznake uspešno ustvarjene v dataset/labels/train/")
