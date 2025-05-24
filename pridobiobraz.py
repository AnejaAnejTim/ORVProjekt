import cv2
import os

def zajemi_slike_iz_videa(video_pot, ciljna_mapa, velikost_slike=(160, 160), vsak_n_ti=4):
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
            ime_slike = os.path.join(ciljna_mapa, f"frame_{saved:04d}.jpg")
            cv2.imwrite(ime_slike, frame_resized)
            saved += 1

        count += 1

    cap.release()
    print(f"Shranjeno {saved} slik iz videa v mapo '{ciljna_mapa}'.")


video_pot = "./videiOseb/Tim_Andrejc.mp4"
ciljna_mapa = "Tim_Andrejc"
zajemi_slike_iz_videa(video_pot, ciljna_mapa, velikost_slike=(160, 160), vsak_n_ti=4)