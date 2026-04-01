import numpy as np
import cv2
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. XAI explanations will not work.")

import matplotlib.pyplot as plt
from PIL import Image

def generate_gradcam(model, image, layer_name=None, class_index=None):
    """
    Generate Grad-CAM heatmap for a given image and model.

    Args:
        model: TensorFlow/Keras model
        image: Preprocessed image array (batch_size, height, width, channels)
        layer_name: Name of the convolutional layer to use (optional)
        class_index: Index of the class to explain (optional, uses predicted class if None)

    Returns:
        Heatmap as numpy array
    """
    try:
        # Remove batch dimension if present
        if len(image.shape) == 4:
            image = image[0]

        # Add batch dimension
        img_array = np.expand_dims(image, axis=0)

        # If no layer specified, find the last convolutional layer
        if layer_name is None:
            for layer in reversed(model.layers):
                if 'conv' in layer.name.lower():
                    layer_name = layer.name
                    break

        if layer_name is None:
            raise ValueError("No convolutional layer found in the model")

        # Get the target layer
        target_layer = model.get_layer(layer_name)

        # Create a model that outputs the target layer's output and the final predictions
        grad_model = Model(
            inputs=model.inputs,
            outputs=[target_layer.output, model.output]
        )

        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if class_index is None:
                class_index = tf.argmax(predictions[0])
            loss = predictions[:, class_index]

        # Get gradients
        grads = tape.gradient(loss, conv_outputs)

        # Compute guided gradients
        cast_conv_outputs = tf.cast(conv_outputs > 0, "float32")
        cast_grads = tf.cast(grads > 0, "float32")
        guided_grads = cast_conv_outputs * cast_grads * grads

        # Compute average gradient
        weights = tf.reduce_mean(guided_grads, axis=(0, 1, 2))

        # Compute weighted sum
        cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1)

        # Apply ReLU
        cam = np.maximum(cam, 0)

        # Normalize
        cam = cam / np.max(cam)

        # Resize to original image size
        cam = cv2.resize(cam[0].numpy(), (image.shape[1], image.shape[0]))

        return cam

    except Exception as e:
        print(f"Error generating Grad-CAM: {str(e)}")
        return np.zeros((224, 224))

def generate_gradcam_pp(model, image, layer_name=None, class_index=None):
    """
    Generate Grad-CAM++ heatmap.

    Args:
        model: TensorFlow/Keras model
        image: Preprocessed image array
        layer_name: Name of the convolutional layer
        class_index: Index of the class to explain

    Returns:
        Heatmap as numpy array
    """
    try:
        # Remove batch dimension if present
        if len(image.shape) == 4:
            image = image[0]

        img_array = np.expand_dims(image, axis=0)

        if layer_name is None:
            for layer in reversed(model.layers):
                if 'conv' in layer.name.lower():
                    layer_name = layer.name
                    break

        target_layer = model.get_layer(layer_name)

        grad_model = Model(
            inputs=model.inputs,
            outputs=[target_layer.output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if class_index is None:
                class_index = tf.argmax(predictions[0])
            loss = predictions[:, class_index]

        grads = tape.gradient(loss, conv_outputs)

        # Grad-CAM++ specific computations
        first_derivative = grads
        second_derivative = grads * grads
        third_derivative = second_derivative * grads

        global_sum = tf.reduce_sum(conv_outputs, axis=(0, 1, 2))

        alpha_num = second_derivative
        alpha_denom = 2 * second_derivative + tf.reduce_sum(conv_outputs * third_derivative, axis=(0, 1, 2))
        alpha_denom = alpha_denom + tf.reduce_sum(conv_outputs * first_derivative, axis=(0, 1, 2)) * global_sum

        alphas = alpha_num / (alpha_denom + 1e-7)

        weights = tf.reduce_sum(alphas * tf.cast(first_derivative > 0, "float32"), axis=(0, 1, 2))

        cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1)

        cam = np.maximum(cam, 0)
        cam = cam / (np.max(cam) + 1e-7)

        cam = cv2.resize(cam[0].numpy(), (image.shape[1], image.shape[0]))

        return cam

    except Exception as e:
        print(f"Error generating Grad-CAM++: {str(e)}")
        return np.zeros((224, 224))

def generate_eigen_cam(model, image, layer_name=None):
    """
    Generate Eigen-CAM heatmap.

    Args:
        model: TensorFlow/Keras model
        image: Preprocessed image array
        layer_name: Name of the convolutional layer

    Returns:
        Heatmap as numpy array
    """
    try:
        if len(image.shape) == 4:
            image = image[0]

        img_array = np.expand_dims(image, axis=0)

        if layer_name is None:
            for layer in reversed(model.layers):
                if 'conv' in layer.name.lower():
                    layer_name = layer.name
                    break

        target_layer = model.get_layer(layer_name)

        # Create model to get intermediate activations
        activation_model = Model(inputs=model.inputs, outputs=target_layer.output)

        # Get activations
        activations = activation_model.predict(img_array, verbose=0)[0]

        # Apply PCA to get principal components
        reshaped_activations = activations.reshape(activations.shape[-1], -1).T

        # Compute covariance matrix
        cov = np.cov(reshaped_activations.T)

        # Get eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Get top eigenvector (corresponding to largest eigenvalue)
        top_eigenvector = eigenvectors[:, -1]

        # Project activations onto the top eigenvector
        cam = np.dot(activations, top_eigenvector)

        # Apply ReLU and normalize
        cam = np.maximum(cam, 0)
        cam = cam / (np.max(cam) + 1e-7)

        # Resize
        cam = cv2.resize(cam, (image.shape[1], image.shape[0]))

        return cam

    except Exception as e:
        print(f"Error generating Eigen-CAM: {str(e)}")
        return np.zeros((224, 224))

def overlay_heatmap(image, heatmap, alpha=0.5):
    """
    Overlay heatmap on original image.

    Args:
        image: Original image array
        heatmap: Heatmap array
        alpha: Transparency factor

    Returns:
        Image with heatmap overlay
    """
    try:
        # Convert heatmap to RGB
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

        # Convert image to uint8 if needed
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        # Overlay
        overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)

        return overlay

    except Exception as e:
        print(f"Error overlaying heatmap: {str(e)}")
        return image

def generate_xai_explanations(model, image, methods):
    """
    Generate explanations using multiple XAI methods.

    Args:
        model: TensorFlow/Keras model
        image: Preprocessed image array
        methods: List of XAI methods to use

    Returns:
        Dictionary of explanations
    """
    explanations = {}

    for method in methods:
        try:
            if method == "Grad-CAM":
                heatmap = generate_gradcam(model, image)
            elif method == "Grad-CAM++":
                heatmap = generate_gradcam_pp(model, image)
            elif method == "Eigen-CAM":
                heatmap = generate_eigen_cam(model, image)
            else:
                continue

            # Create overlay
            original_image = (image[0] * 255).astype(np.uint8) if len(image.shape) == 4 else (image * 255).astype(np.uint8)
            overlay = overlay_heatmap(original_image, heatmap)

            explanations[method] = {
                'heatmap': heatmap,
                'overlay': overlay
            }

        except Exception as e:
            print(f"Error generating {method} explanation: {str(e)}")
            continue

    return explanations
