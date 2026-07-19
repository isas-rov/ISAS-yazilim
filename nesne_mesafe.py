import cv2
import time
from ultralytics import YOLO

# ---------------- AYARLAR ----------------
KAMERA = 0
GUVEN  = 0.5

KALIBRASYON_MESAFE = 100.0   # cm  -> nesneyi TAM 1 metreye koyup 'c' bas

# Senin eGittiGin modelin siniflari ve GERCEK genislikleri (cm)
# NOT: samandira gercek capini yarisma teknik ciziminden alacaksin. Simdilik tahmin.
GERCEK_GENISLIK = {
    "green_buoy":  20.0,
    "orange_buoy": 20.0,
}
# -----------------------------------------

odak = None

model = YOLO("runs/detect/samandira_v1/weights/best.pt")   # KENDI modelin

cap = cv2.VideoCapture(KAMERA)
if not cap.isOpened():
    raise SystemExit("Kamera acilamadi. KAMERA degerini 1 veya 2 yap.")

print("Bilinen bir samandirayi TAM 1 METREYE koy ve 'c' tusuna bas.")
print("ESC = cikis")

onceki = time.time()

while True:
    ret, kare = cap.read()
    if not ret:
        break

    sonuclar = model(kare, conf=GUVEN, verbose=False)[0]

    en_genis = None

    for kutu in sonuclar.boxes:
        x1, y1, x2, y2 = map(int, kutu.xyxy[0])
        sinif = model.names[int(kutu.cls[0])]
        guven = float(kutu.conf[0])

        pik_genislik = x2 - x1
        etiket = f"{sinif} {guven:.2f}"
        renk = (0, 200, 255)

        if sinif in GERCEK_GENISLIK and pik_genislik > 0:
            if en_genis is None or pik_genislik > en_genis[1]:
                en_genis = (sinif, pik_genislik)

            if odak is not None:
                mesafe = (GERCEK_GENISLIK[sinif] * odak) / pik_genislik
                etiket = f"{sinif}  {mesafe:.0f} cm"
                renk = (0, 255, 0)
                if mesafe < 40:
                    renk = (0, 0, 255)
                    etiket += "  ! YAKIN"

        cv2.rectangle(kare, (x1, y1), (x2, y2), renk, 2)
        cv2.putText(kare, etiket, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, renk, 2)

    if odak is None:
        cv2.putText(kare, "KALIBRE DEGIL - samandirayi 1m'ye koy, 'c' bas",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        cv2.putText(kare, f"Odak: {odak:.0f} px", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    simdi = time.time()
    fps = 1.0 / max(simdi - onceki, 1e-6)
    onceki = simdi
    cv2.putText(kare, f"FPS: {fps:.1f}", (10, kare.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    cv2.imshow("Samandira Tespiti + Mesafe", kare)

    tus = cv2.waitKey(1)
    if tus == 27:
        break
    if tus == ord("c"):
        if en_genis:
            sinif, pik = en_genis
            odak = (pik * KALIBRASYON_MESAFE) / GERCEK_GENISLIK[sinif]
            print(f"KALIBRE: '{sinif}' {pik}px @ {KALIBRASYON_MESAFE}cm  ->  odak = {odak:.1f}px")
        else:
            print("Samandira gormuyorum, kalibre edemedim.")

cap.release()
cv2.destroyAllWindows()