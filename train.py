from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    results = model.train(data="Roboflow_Dataset/data.yaml", epochs=150, 
                        imgsz=640, batch=16, optimizer="Adam", lr0=0.01,
                        mosaic=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
                        translate=0.1, scale=0.5, fliplr=0.5)
    results = model.val()

