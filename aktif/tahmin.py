from ultralytics import YOLO

model = YOLO("runs/detect/samandira_v1/weights/best.pt")

results = model.predict(
    source="test.jpg",
    conf=0.5,
    save=True,
    show=True
)