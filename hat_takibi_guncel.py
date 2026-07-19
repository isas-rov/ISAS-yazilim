"""
Hat Takibi - Jetson + Laptop uyumlu
Kullanim:
  python hat_takibi_jetson.py            -> laptop webcam
  python hat_takibi_jetson.py --csi      -> Jetson CSI kamera
  python hat_takibi_jetson.py --video x.mp4  -> video dosyasi
Serite fareyle TIKLA -> rengini ogrenir.  ESC = cikis.  R = kalibrasyon sifirla.
"""

import cv2
import numpy as np
import time
import sys
import argparse

# ---------------- AYARLAR ----------------
ROI_ORAN  = 0.6
MIN_ALAN  = 800
MAX_KOMUT = 100
OLU_BOLGE = 10
LOG_DOSYA = "hat_takibi_log.csv"
# -----------------------------------------


def jetson_csi_pipeline(genislik=1280, yukseklik=720, fps=30, flip=0):
    """Jetson CSI kamera icin GStreamer pipeline. Normal webcam gibi acilmaz."""
    return (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width={genislik}, height={yukseklik}, "
        f"framerate={fps}/1 ! "
        f"nvvidconv flip-method={flip} ! "
        f"video/x-raw, format=BGRx ! videoconvert ! "
        f"video/x-raw, format=BGR ! appsink drop=true"
    )


class PID:
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.birikim = 0.0
        self.onceki_hata = 0.0
        self.onceki_zaman = time.time()

    def hesapla(self, hata):
        simdi = time.time()
        dt = max(simdi - self.onceki_zaman, 1e-3)
        p = self.kp * hata
        self.birikim = max(-500, min(500, self.birikim + hata * dt))
        i = self.ki * self.birikim
        d = self.kd * (hata - self.onceki_hata) / dt
        self.onceki_hata = hata
        self.onceki_zaman = simdi
        return max(-MAX_KOMUT, min(MAX_KOMUT, p + i + d))

    def sifirla(self):
        self.birikim = 0.0
        self.onceki_hata = 0.0


kalibre = {"var": False, "h": 0, "s": 0, "v": 0}
son_kare = None


def tikla(event, x, y, flags, param):
    global kalibre
    if event == cv2.EVENT_LBUTTONDOWN and son_kare is not None:
        h, w = son_kare.shape[:2]
        x0, x1 = max(0, x - 5), min(w, x + 6)
        y0, y1 = max(0, y - 5), min(h, y + 6)
        parca = cv2.cvtColor(son_kare[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        ort = np.median(parca.reshape(-1, 3), axis=0)
        kalibre = {"var": True, "h": int(ort[0]), "s": int(ort[1]), "v": int(ort[2])}
        print(f"KALIBRE -> H:{kalibre['h']} S:{kalibre['s']} V:{kalibre['v']}")


def hicbirsey(x):
    pass


def kamera_ac(args):
    if args.csi:
        print("Jetson CSI kamera aciliyor...")
        return cv2.VideoCapture(jetson_csi_pipeline(flip=args.flip), cv2.CAP_GSTREAMER)
    if args.video:
        return cv2.VideoCapture(args.video)
    return cv2.VideoCapture(args.kamera)


def main():
    global son_kare

    ap = argparse.ArgumentParser()
    ap.add_argument("--csi", action="store_true", help="Jetson CSI kamera kullan")
    ap.add_argument("--video", type=str, default=None, help="Video dosyasi yolu")
    ap.add_argument("--kamera", type=int, default=0, help="Webcam indeksi")
    ap.add_argument("--flip", type=int, default=0, help="CSI goruntu cevirme (0-3)")
    ap.add_argument("--log", action="store_true", help="Hata degerlerini CSV'ye kaydet")
    args = ap.parse_args()

    cap = kamera_ac(args)
    if not cap.isOpened():
        raise SystemExit("Kamera acilamadi. --csi / --kamera / --video ayarini kontrol et.")

    cv2.namedWindow("Kamera")
    cv2.setMouseCallback("Kamera", tikla)
    cv2.namedWindow("Ayarlar")
    cv2.createTrackbar("H tolerans", "Ayarlar", 12, 90,  hicbirsey)
    cv2.createTrackbar("S tolerans", "Ayarlar", 70, 255, hicbirsey)
    cv2.createTrackbar("V tolerans", "Ayarlar", 70, 255, hicbirsey)
    cv2.createTrackbar("CLAHE",      "Ayarlar",  1,   1, hicbirsey)
    cv2.createTrackbar("Kp x100",    "Ayarlar", 40, 200, hicbirsey)
    cv2.createTrackbar("Kd x1000",   "Ayarlar", 20, 500, hicbirsey)

    pid = PID(0.4, 0.0, 0.02)

    log_f = open(LOG_DOSYA, "w") if args.log else None
    if log_f:
        log_f.write("zaman,hata,aci,komut\n")

    onceki = time.time()
    print("Serite TIKLA. ESC=cikis, R=sifirla")

    while True:
        ret, img = cap.read()
        if not ret:
            print("Kare alinamadi.")
            break

        son_kare = img.copy()
        h, w = img.shape[:2]
        roi_ust = int(h * ROI_ORAN)
        roi = img[roi_ust:, :]

        # CLAHE - su alti kontrast kurtarma
        if cv2.getTrackbarPos("CLAHE", "Ayarlar"):
            lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
            roi = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

        roi = cv2.GaussianBlur(roi, (5, 5), 0)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        maske = np.zeros(hsv.shape[:2], np.uint8)

        pid.kp = cv2.getTrackbarPos("Kp x100",  "Ayarlar") / 100.0
        pid.kd = cv2.getTrackbarPos("Kd x1000", "Ayarlar") / 1000.0

        hata, aci, komut, hat_var = 0, 0.0, 0.0, False

        if kalibre["var"]:
            ht = cv2.getTrackbarPos("H tolerans", "Ayarlar")
            st = cv2.getTrackbarPos("S tolerans", "Ayarlar")
            vt = cv2.getTrackbarPos("V tolerans", "Ayarlar")
            H, S, V = kalibre["h"], kalibre["s"], kalibre["v"]
            s_alt, s_ust = max(0, S - st), min(255, S + st)
            v_alt, v_ust = max(0, V - vt), min(255, V + vt)

            if S < 40:   # siyah/beyaz/gri -> parlakliga bak
                maske = cv2.inRange(hsv, np.array([0, 0, v_alt]),
                                         np.array([180, 80, v_ust]))
            else:
                h_alt, h_ust = H - ht, H + ht
                if h_alt < 0 or h_ust > 180:
                    m1 = cv2.inRange(hsv, np.array([0, s_alt, v_alt]),
                                          np.array([h_ust % 180, s_ust, v_ust]))
                    m2 = cv2.inRange(hsv, np.array([h_alt % 180, s_alt, v_alt]),
                                          np.array([180, s_ust, v_ust]))
                    maske = m1 | m2
                else:
                    maske = cv2.inRange(hsv, np.array([h_alt, s_alt, v_alt]),
                                             np.array([h_ust, s_ust, v_ust]))

            kernel = np.ones((5, 5), np.uint8)
            maske = cv2.morphologyEx(maske, cv2.MORPH_OPEN,  kernel)
            maske = cv2.morphologyEx(maske, cv2.MORPH_CLOSE, kernel)

            konturlar, _ = cv2.findContours(maske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if konturlar:
                en_buyuk = max(konturlar, key=cv2.contourArea)
                if cv2.contourArea(en_buyuk) > MIN_ALAN:
                    hat_var = True
                    M = cv2.moments(en_buyuk)
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"]) + roi_ust
                    vx, vy, _, _ = cv2.fitLine(en_buyuk, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
                    aci = float(np.degrees(np.arctan2(vx, vy)))
                    hata = cx - w // 2
                    if abs(hata) < OLU_BOLGE:
                        hata = 0
                    komut = pid.hesapla(hata)
                    yon = "SAGA" if komut > 5 else ("SOLA" if komut < -5 else "DUZ")
                    cv2.circle(img, (cx, cy), 8, (0, 255, 0), -1)
                    cv2.drawContours(img[roi_ust:, :], [en_buyuk], -1, (0, 255, 255), 2)
                    cv2.putText(img, f"HATA:{hata:+4d} ACI:{aci:+5.1f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(img, f"{yon} guc:{komut:+6.1f}", (10, 58),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if not hat_var:
            pid.sifirla()
            mesaj = "TIKLA (rengi ogret)" if not kalibre["var"] else "HAT YOK -> FAILSAFE"
            cv2.putText(img, mesaj, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # FPS
        simdi = time.time()
        fps = 1.0 / max(simdi - onceki, 1e-6)
        onceki = simdi
        cv2.putText(img, f"FPS:{fps:.1f}", (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.line(img, (w // 2, 0), (w // 2, h), (255, 0, 0), 1)
        cv2.line(img, (0, roi_ust), (w, roi_ust), (0, 255, 255), 1)

        if log_f and hat_var:
            log_f.write(f"{simdi:.3f},{hata},{aci:.1f},{komut:.1f}\n")

        cv2.imshow("Kamera", img)
        cv2.imshow("Maske", maske)

        k = cv2.waitKey(1)
        if k == 27:
            break
        if k == ord("r"):
            kalibre["var"] = False
            print("Kalibrasyon sifirlandi.")

    if log_f:
        log_f.close()
        print(f"Log kaydedildi: {LOG_DOSYA}")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()