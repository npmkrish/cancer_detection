from ultralytics import YOLO
model=YOLO('best5.pt')

results=model.predict(source='static/images/b8a2aa251bdf4a0b8a376137b6d7f8d9_image.jpeg', save=True, save_txt=True, conf=0.5, iou=0.5, device='0', show=True)
print(results)  

#//!!