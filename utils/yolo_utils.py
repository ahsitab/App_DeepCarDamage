import numpy as np
import cv2
from ultralytics import YOLO
import torch

def load_yolo_model(model_path):
    """
    Load a YOLO model from file.

    Args:
        model_path: Path to the YOLO model file (.pt)

    Returns:
        Loaded YOLO model
    """
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        raise ValueError(f"Error loading YOLO model from {model_path}: {str(e)}")

def detect_objects(model, image, conf_threshold=0.25):
    """
    Perform car damage detection using YOLO model.

    Args:
        model: Loaded YOLO model
        image: Preprocessed image array (OpenCV format)
        conf_threshold: Confidence threshold for detections (lowered to 0.25 for better detection)

    Returns:
        List of detections with bounding boxes, classes, and confidence scores
    """
    try:
        # Run inference with lower confidence threshold
        results = model(image, conf=conf_threshold, verbose=False, iou=0.45)

        detections = []

        # Process results
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                    # Get confidence and class
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())

                    # Get class name
                    class_name = model.names[cls]

                    # Map YOLO classes to damage severity levels
                    damage_mapping = {
                        'dent': 'Moderate Damage',
                        'scratch': 'Minor Damage',
                        'crack': 'Severe Damage',
                        'glass shatter': 'Severe Damage',
                        'lamp broken': 'Severe Damage',
                        'tire flat': 'Severe Damage'
                    }

                    # Get damage severity
                    damage_severity = damage_mapping.get(class_name.lower(), 'Unknown Damage')

                    # Calculate damage area
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    damage_area = bbox_width * bbox_height

                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'class': class_name,
                        'damage_type': damage_severity,
                        'confidence': float(conf * 100),  # Convert to percentage
                        'class_id': cls,
                        'area': float(damage_area),
                        'center': [(x1 + x2) / 2, (y1 + y2) / 2]
                    })

        # Sort by confidence (highest first)
        detections.sort(key=lambda x: x['confidence'], reverse=True)

        return detections

    except Exception as e:
        print(f"Error in damage detection: {str(e)}")
        return []

def draw_bounding_boxes(image, detections):
    """
    Draw bounding boxes on image with class labels.

    Args:
        image: OpenCV image array
        detections: List of detections

    Returns:
        Image with bounding boxes drawn
    """
    try:
        img_copy = image.copy()

        # Color mapping for different classes
        color_map = {
            'dent': (0, 255, 0),        # Green
            'scratch': (255, 255, 0),   # Yellow
            'crack': (0, 0, 255),       # Red
            'glass shatter': (255, 0, 255), # Magenta
            'lamp broken': (0, 255, 255),   # Cyan
            'tire flat': (255, 165, 0),     # Orange
        }

        # Default color for unknown classes
        default_color = (128, 128, 128)  # Gray

        for det in detections:
            bbox = det['bbox']
            class_name = det['class']
            confidence = det['confidence']

            # Get color based on class name
            color = color_map.get(class_name.lower(), default_color)

            # Draw rectangle with thicker border
            cv2.rectangle(img_copy, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)

            # Create label with class name and confidence
            label = f"{class_name} ({confidence:.1f}%)"

            # Draw label background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img_copy,
                         (bbox[0], bbox[1] - text_height - 10),
                         (bbox[0] + text_width, bbox[1]),
                         color, -1)

            # Draw label text
            cv2.putText(img_copy, label, (bbox[0], bbox[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return img_copy

    except Exception as e:
        print(f"Error drawing bounding boxes: {str(e)}")
        return image

def get_yolo_model_info(model_path):
    """
    Get information about a YOLO model.

    Args:
        model_path: Path to the YOLO model file

    Returns:
        Dictionary with model information
    """
    import os

    info = {
        'path': model_path,
        'size': os.path.getsize(model_path),
        'filename': os.path.basename(model_path)
    }

    # Try to infer architecture from filename
    filename = info['filename'].lower()
    if 'yolov8' in filename:
        info['architecture'] = 'YOLOv8'
        info['input_size'] = '640x640'
    elif 'yolov11' in filename:
        info['architecture'] = 'YOLOv11'
        info['input_size'] = '640x640'
    elif 'yolov12' in filename:
        info['architecture'] = 'YOLOv12'
        info['input_size'] = '640x640'
    else:
        info['architecture'] = 'YOLO'
        info['input_size'] = '640x640'

    # Try to load model to get class names
    try:
        model = YOLO(model_path)
        info['classes'] = list(model.names.values())
        info['num_classes'] = len(model.names)
    except:
        info['classes'] = []
        info['num_classes'] = 0

    return info

def preprocess_for_yolo(image, target_size=(640, 640)):
    """
    Preprocess image for YOLO inference.

    Args:
        image: PIL Image or OpenCV image
        target_size: Target size for YOLO (width, height)

    Returns:
        Preprocessed image
    """
    # Convert PIL to OpenCV if needed
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Resize image
    resized = cv2.resize(image, target_size)

    return resized
