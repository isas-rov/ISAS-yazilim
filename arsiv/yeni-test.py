from ultralytics import YOLO

eski = YOLO("runs/detect/samandira_v1/weights/best.pt")
yeni = YOLO("runs/detect/buoy_karisik_v1-2/weights/best.pt")

# test_gorseller klasorune 5-10 tane KARISIK gorsel koy
# (hem su alti hem su ustu samandira)

print("=== ESKI MODEL (su alti) ===")
eski.predict(source="test_gorseller", conf=0.4, save=True, project="karsilastirma", name="eski")

print("=== YENI MODEL (karisik) ===")
yeni.predict(source="test_gorseller", conf=0.4, save=True, project="karsilastirma", name="yeni")