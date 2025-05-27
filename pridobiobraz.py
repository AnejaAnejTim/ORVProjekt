import cv2
import os

def zajemi_slike_iz_videa(video_pot, ciljna_mapa, velikost_slike=(250, 250), vsak_n_ti=2, ime_prefixa="tim"):
    if not os.path.exists(ciljna_mapa):
        os.makedirs(ciljna_mapa)

    cap = cv2.VideoCapture(video_pot)
    count = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # konec videa

        if count % vsak_n_ti == 0:
            frame_resized = cv2.resize(frame, velikost_slike)
            ime_slike = os.path.join(ciljna_mapa, f"{ime_prefixa}_{saved:04d}.jpg")
            cv2.imwrite(ime_slike, frame_resized)
            saved += 1

        count += 1

    cap.release()
    print(f"✅ Shrani {saved} slik iz videa v mapo '{ciljna_mapa}' kot {ime_prefixa}_####.jpg")

# Uporaba:
video_pot = "./videiOseb/Aneja.mp4"
ciljna_mapa = "dataset/person"
zajemi_slike_iz_videa(video_pot, ciljna_mapa, velikost_slike=(250, 250), vsak_n_ti=2, ime_prefixa="tim")
