import numpy as np
from PIL import Image
import cv2

def preprocess_image(image, model_type):
    """
    Preprocess image for CNN or YOLO models.

    Args:
        image: PIL Image object
        model_type: 'CNN Classification' or 'YOLO Detection'

    Returns:
        Preprocessed image as numpy array
    """
    if model_type == "CNN Classification":
        # Resize to 224x224 for CNN models
        image = image.resize((224, 224))

        # Convert to numpy array
        img_array = np.array(image)

        # Normalize to [0, 1]
        img_array = img_array.astype(np.float32) / 255.0

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    elif model_type == "YOLO Detection":
        # Convert PIL to OpenCV format
        img_array = np.array(image)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # YOLO models typically expect BGR format
        return img_array

    else:
        raise ValueError("Invalid model type. Must be 'CNN Classification' or 'YOLO Detection'")

def preprocess_for_xai(image):
    """
    Preprocess image for XAI explanations.

    Args:
        image: PIL Image object

    Returns:
        Preprocessed image for XAI
    """
    # Resize to 224x224
    image = image.resize((224, 224))

    # Convert to numpy array
    img_array = np.array(image)

    # Normalize to [0, 1]
    img_array = img_array.astype(np.float32) / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array
