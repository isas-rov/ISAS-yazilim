from ultralytics import YOLO

def main():
    model = YOLO("yolo11n.pt")

    model.train(
        data="veri_birlesik/data.yaml",   # veri yerine veri_birlesik
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        name="buoy_karisik_v1",           # yeni isim
        patience=15,
    )

if __name__ == "__main__":
    main()