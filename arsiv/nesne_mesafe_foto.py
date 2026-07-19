import cv2
from ultralytics import YOLO

# ---------------- AYARLAR ----------------
GORSEL = "test2.jpg"        # test edeceGin gorselin adi
GUVEN  = 0.4

ODAK              = 700.0   # kamera sabiti (gercek kalibrasyonla degisir)
GERCEK_GENISLIK = {
    "green_buoy":  20.0,   # cm
    "orange_buoy": 20.0,   # cm
}
# -----------------------------------------

model = YOLO("runs/detect/samandira_v1/weights/best.pt")

kare = cv2.imread(GORSEL)
if kare is None:
    raise SystemExit(f"Gorsel bulunamadi: {GORSEL}")

sonuclar = model(kare, conf=GUVEN, verbose=False)[0]

if len(sonuclar.boxes) == 0:
    print("Hicbir samandira bulunamadi.")

for kutu in sonuclar.boxes:
    x1, y1, x2, y2 = map(int, kutu.xyxy[0])
    sinif = model.names[int(kutu.cls[0])]
    guven = float(kutu.conf[0])
    pik   = x2 - x1

    etiket = f"{sinif} {guven:.2f}"
    renk = (0, 255, 0)

    if sinif in GERCEK_GENISLIK and pik > 0:
        mesafe = (GERCEK_GENISLIK[sinif] * ODAK) / pik
        etiket = f"{sinif}  ~{mesafe:.0f} cm"
        print(f"{sinif}: guven {guven:.2f}, genislik {pik}px, tahmini mesafe ~{mesafe:.0f} cm")

    cv2.rectangle(kare, (x1, y1), (x2, y2), renk, 2)
    cv2.putText(kare, etiket, (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, renk, 2)

cv2.imshow("Sonuc (bir tusa bas -> kapat)", kare)
cv2.waitKey(0)
cv2.destroyAllWindows()