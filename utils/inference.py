TENSORFLOW_AVAILABLE = True
import numpy as np

def load_model(model_path):
    """
    Load a TensorFlow/Keras model from file.
    """
    try:
        from tensorflow.keras.models import load_model as keras_load_model
        model = keras_load_model(model_path)
        return model
    except Exception as e:
        raise ValueError(f"Error loading model from {model_path}: {str(e)}")

def predict_classification(model, image):
    """
    Make classification prediction using CNN model.
    """
    try:
        import tensorflow as tf
        # Make prediction
        predictions = model.predict(image, verbose=0)

        # Get class names (assuming standard car damage severity classes)
        class_names = ['01-minor', '02-moderate', '03-severe']

        # Convert predictions to percentages
        predictions_percent = predictions[0] * 100

        # Create results list
        results = []
        for i, (class_name, confidence) in enumerate(zip(class_names, predictions_percent)):
            results.append({
                'class': class_name,
                'confidence': float(confidence)
            })

        # Sort by confidence (highest first)
        results.sort(key=lambda x: x['confidence'], reverse=True)

        return results

    except Exception as e:
        raise ValueError(f"Error making classification prediction: {str(e)}")

def predict_detection(model, image):
    """
    Make detection prediction using YOLO model.

    Args:
        model: Loaded YOLO model
        image: Preprocessed image array

    Returns:
        List of detections with bounding boxes, classes, and confidence scores
    """
    # This will be implemented in yolo_utils.py
    # Placeholder for now
    return []

def get_model_info(model_path):
    """
    Get information about a model.

    Args:
        model_path: Path to the model file

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
    if 'vgg16' in filename:
        info['architecture'] = 'VGG16'
        info['input_size'] = '224x224'
    elif 'vgg19' in filename:
        info['architecture'] = 'VGG19'
        info['input_size'] = '224x224'
    elif 'resnet50' in filename:
        info['architecture'] = 'ResNet50'
        info['input_size'] = '224x224'
    elif 'densenet121' in filename:
        info['architecture'] = 'DenseNet121'
        info['input_size'] = '224x224'
    elif 'mobilenetv2' in filename:
        info['architecture'] = 'MobileNetV2'
        info['input_size'] = '224x224'
    elif 'yolov8' in filename:
        info['architecture'] = 'YOLOv8'
        info['input_size'] = '640x640'
    elif 'yolov11' in filename:
        info['architecture'] = 'YOLOv11'
        info['input_size'] = '640x640'
    elif 'yolov12' in filename:
        info['architecture'] = 'YOLOv12'
        info['input_size'] = '640x640'
    else:
        info['architecture'] = 'Unknown'
        info['input_size'] = 'Unknown'

    return info
