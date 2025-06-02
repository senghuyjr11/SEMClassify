yolo detect train model=yolov8n.pt data=yolo_corrosion.yaml epochs=50 imgsz=640 project=model name=corrosion_yolov8

yolo detect val model=model/corrosion_yolov8/weights/best.pt data=yolo_corrosion.yaml project=model name=val_results

yolo detect predict model=model/corrosion_yolov8/weights/best.pt source=some_folder_or_image project=model name=inference