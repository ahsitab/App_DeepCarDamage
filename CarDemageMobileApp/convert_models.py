import tensorflow as tf
from ultralytics import YOLO
import os

def convert_yolo():
    print("Converting YOLOv12...")
    model_path = "e:/SITAB/Capstone_C_App/CarDemageMobileApp/Models_Weight files/best(1)yolov12.pt"
    # Load the YOLOv12 model
    model = YOLO(model_path)
    # Export to TFLite with int8 quantization if possible (here we do float32 first for simplicity)
    model.export(format="tflite", imgsz=640)
    print("YOLOv12 conversion complete.")

def convert_mobilenet():
    print("Converting MobileNetV2...")
    model_path = "e:/SITAB/Capstone_C_App/CarDemageMobileApp/Models_Weight files/mobilenetv2_best.h5"
    # Load the H5 model
    model = tf.keras.models.load_model(model_path)
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    # Save the model
    with open("e:/SITAB/Capstone_C_App/CarDemageMobileApp/Models_Weight files/mobilenetv2_best.tflite", 'wb') as f:
        f.write(tflite_model)
    print("MobileNetV2 conversion complete.")

if __name__ == "__main__":
    # Ensure ultralytics and tensorflow are installed
    convert_yolo() 
    # convert_mobilenet()
