import os, shutil, glob

# Birlestirilecek kaynak setler
KAYNAKLAR = ["veri", "veri2"]
HEDEF = "veri_birlesik"
BOLUMLER = ["train", "valid", "test"]

# Hedef klasorleri olustur
for b in BOLUMLER:
    os.makedirs(f"{HEDEF}/{b}/images", exist_ok=True)
    os.makedirs(f"{HEDEF}/{b}/labels", exist_ok=True)

sayac = 0

for kaynak in KAYNAKLAR:
    for bolum in BOLUMLER:
        img_dir = f"{kaynak}/{bolum}/images"
        lbl_dir = f"{kaynak}/{bolum}/labels"
        if not os.path.isdir(img_dir):
            continue

        for img_yolu in glob.glob(f"{img_dir}/*"):
            ad = os.path.splitext(os.path.basename(img_yolu))[0]
            lbl_yolu = f"{lbl_dir}/{ad}.txt"

            # Cakismayi onlemek icin yeni benzersiz isim
            yeni_ad = f"{kaynak}_{sayac:05d}"
            uzanti = os.path.splitext(img_yolu)[1]

            # Gorseli kopyala
            shutil.copy(img_yolu, f"{HEDEF}/{bolum}/images/{yeni_ad}{uzanti}")

            # Etiketi kopyala AMA tum siniflari 0 (buoy) yap
            if os.path.exists(lbl_yolu):
                yeni_satirlar = []
                with open(lbl_yolu) as f:
                    for satir in f:
                        parcalar = satir.split()
                        if len(parcalar) >= 5:
                            parcalar[0] = "0"          # sinif no -> 0 (buoy)
                            yeni_satirlar.append(" ".join(parcalar))
                with open(f"{HEDEF}/{bolum}/labels/{yeni_ad}.txt", "w") as f:
                    f.write("\n".join(yeni_satirlar))

            sayac += 1

# Yeni data.yaml olustur
yaml_icerik = """train: train/images
val: valid/images
test: test/images

nc: 1
names: ['buoy']
"""
with open(f"{HEDEF}/data.yaml", "w") as f:
    f.write(yaml_icerik)

# Ozet
for bolum in BOLUMLER:
    n = len(glob.glob(f"{HEDEF}/{bolum}/images/*"))
    print(f"{bolum}: {n} goruntu")
print(f"\nToplam {sayac} goruntu birlestirildi -> {HEDEF}/")