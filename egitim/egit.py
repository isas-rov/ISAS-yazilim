from ultralytics import YOLO

def main():
    model = YOLO("yolo11n.pt")

    model.train(
        data="veri/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        name="samandira_v1",
        patience=15,
    )

if __name__ == "__main__":
    main()