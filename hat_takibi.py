import cv2
import numpy as np

DOSYA = "test_hat_1_temiz.png"

img = cv2.imread(DOSYA)
if img is None:                                   # dosya adi yanlissa sessizce patlamasin
    raise SystemExit(f"Dosya bulunamadi: {DOSYA}")

h, w = img.shape[:2]

roi_ust = int(h * 0.6)          # sadece alt %40'a bak
roi = img[roi_ust:, :]

hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
alt1, ust1 = np.array([0, 70, 40]),   np.array([12, 255, 255])
alt2, ust2 = np.array([168, 70, 40]), np.array([180, 255, 255])
maske = cv2.inRange(hsv, alt1, ust1) | cv2.inRange(hsv, alt2, ust2)

kernel = np.ones((5, 5), np.uint8)
maske = cv2.morphologyEx(maske, cv2.MORPH_OPEN,  kernel)   # kucuk lekeleri sil
maske = cv2.morphologyEx(maske, cv2.MORPH_CLOSE, kernel)   # seritteki delikleri kapat

konturlar, _ = cv2.findContours(maske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if konturlar:
    en_buyuk = max(konturlar, key=cv2.contourArea)

    if cv2.contourArea(en_buyuk) > 500:           # cok kucukse gurultudur, guvenme
        M = cv2.moments(en_buyuk)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"]) + roi_ust   # ROI'den tam goruntuye geri tasi

        vx, vy, _, _ = cv2.fitLine(en_buyuk, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        aci = np.degrees(np.arctan2(vx, vy))      # 0 = dik duruyor, + = saga yatik

        merkez = w // 2
        hata = cx - merkez
        print(f"HATA: {hata:+4d} px  |  ACI: {aci:+6.1f} derece")

        cv2.circle(img, (cx, cy), 8, (0, 255, 0), -1)
        cv2.line(img, (merkez, 0), (merkez, h), (255, 0, 0), 2)
        cv2.line(img, (0, roi_ust), (w, roi_ust), (0, 255, 255), 1)   # sari cizgi = ROI siniri
    else:
        print("Hat cok kucuk -> guvenilmez")
else:
    print("HAT BULUNAMADI -> failsafe")

cv2.imshow("Sonuc", img)
cv2.imshow("Maske (ROI)", maske)
cv2.waitKey(0)
cv2.destroyAllWindows()