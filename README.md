# Sualtı Projesi - Görüntü İşleme ve Otonom Sistemler

Bu depo, sualtı aracı projesi kapsamında geliştirilen görüntü işleme, otonom hat takibi, nesne/şamandıra tespiti ve mesafe hesaplama algoritmalarını içermektedir. Projede nesne tespiti için YOLO modelleri kullanılmaktadır.

## 📂 Dosya ve Klasör Yapısı

Karmaşıklığı önlemek ve geliştirme sürecini hızlandırmak amacıyla kodlar üç ana kategoriye ayrılmıştır:

* **`aktif/`**: Projede şu an aktif olarak kullanılan, güncel ve çalışan ana kodları içerir.
  * `hat_takibi_guncel.py`: Otonom hat takibi yapan güncel görüntü işleme kodumuz.
  * `tahmin.py`: Model üzerinden anlık nesne tespiti ve mesafe tahmini yapan dosya.

* **`egitim/`**: Yapay zeka modelini eğitmek ve veri setlerini düzenlemek için kullanılan scriptler.
  * `birlestir.py`: Farklı veri setlerini eğitim öncesi tek bir yapıda birleştirir.
  * `egit.py` / `egit2.py`: YOLO modelini kendi verilerimizle eğitmek için kullandığımız kodlar.

* **`arsiv/`**: Eski denemeler, ilk versiyonlar ve yedek kodlar burada tutulmaktadır. (Örn: `nesne_mesafe.py`, `hat_takibi_webcam.py`, vb.)

## 🛠️ Gereksinimler

Projeyi çalıştırmak için aşağıdaki Python kütüphanelerinin bilgisayarınızda (veya sanal ortamınızda) kurulu olması gerekmektedir:

* Python 3.8+
* OpenCV (`opencv-python`)
* Ultralytics (YOLO için)
* NumPy

Gerekli paketleri kurmak için terminalinizde şu komutu çalıştırabilirsiniz:
```bash
pip install opencv-python ultralytics numpy
