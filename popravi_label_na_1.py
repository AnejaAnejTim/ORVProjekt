import os

def popravi_oznako(mapa_s_slikami, mapa_z_oznakami):
    for filename in os.listdir(mapa_s_slikami):
        if 'tim' in filename.lower() and filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            base_name = os.path.splitext(filename)[0]
            txt_path = os.path.join(mapa_z_oznakami, base_name + '.txt')

            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    line = f.readline().strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 5:
                        parts[0] = '1'  # nastavi class_id na 1
                        nova_vrstica = ' '.join(parts)
                        with open(txt_path, 'w') as f:
                            f.write(nova_vrstica + '\n')
                        print(f"✔️ Posodobljeno: {txt_path}")
                    else:
                        print(f"⚠️ Napačna oznaka: {txt_path}")
                else:
                    print(f"⚠️ Prazna datoteka: {txt_path}")
            else:
                print(f"⚠️ Ne najdem oznake za: {filename}")

# Glavne poti
BASE_DIR = 'dataset'

for podmapa in ['train', 'val']:
    images_path = os.path.join(BASE_DIR, 'images', podmapa)
    labels_path = os.path.join(BASE_DIR, 'labels', podmapa)
    print(f"\n📂 Obdelujem {podmapa}...")
    popravi_oznako(images_path, labels_path)
