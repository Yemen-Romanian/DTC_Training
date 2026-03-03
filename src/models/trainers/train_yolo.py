from ultralytics import YOLO

# Load a COCO-pretrained YOLO11n model
model = YOLO("yolo11n.pt")
results = model.train(data=r"Path To Yaml", 
                      epochs=20, single_cls=True, 
                      imgsz=640, classes=[0], batch=4)
print("============ Training finished!=================")