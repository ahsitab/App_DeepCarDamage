import os
import time
import numpy as np
from PIL import Image
from utils.preprocessing import preprocess_image
from utils.yolo_utils import load_yolo_model, detect_objects

def compare_models_on_image(image, model_paths, model_names, conf_threshold=0.5):
    """
    Compare multiple YOLO models on the same image.

    Args:
        image: PIL Image object
        model_paths: List of paths to YOLO model files
        model_names: List of model names for display
        conf_threshold: Confidence threshold for detections

    Returns:
        Dictionary with comparison results
    """
    try:
        comparison_results = {
            'image_info': {
                'size': image.size,
                'mode': image.mode
            },
            'model_results': [],
            'metrics': {}
        }

        processing_times = []

        # Process each model
        for model_path, model_name in zip(model_paths, model_names):
            try:
                # Load model
                model = load_yolo_model(model_path)

                # Preprocess image
                processed_image = preprocess_image(image, "YOLO Detection")

                # Time the inference
                start_time = time.time()
                detections = detect_objects(model, processed_image, conf_threshold)
                processing_time = time.time() - start_time

                # Calculate metrics
                num_detections = len(detections)
                avg_confidence = np.mean([d['confidence'] for d in detections]) if detections else 0

                # Store results
                model_result = {
                    'model_name': model_name,
                    'model_path': model_path,
                    'num_detections': num_detections,
                    'processing_time': processing_time,
                    'avg_confidence': avg_confidence,
                    'detections': detections
                }

                comparison_results['model_results'].append(model_result)
                processing_times.append(processing_time)

            except Exception as e:
                # Handle model loading/inference errors
                error_result = {
                    'model_name': model_name,
                    'model_path': model_path,
                    'num_detections': 0,
                    'processing_time': 0,
                    'avg_confidence': 0,
                    'detections': [],
                    'error': str(e)
                }
                comparison_results['model_results'].append(error_result)
                processing_times.append(0)

        # Calculate overall metrics
        if processing_times:
            valid_times = [t for t in processing_times if t > 0]
            if valid_times:
                comparison_results['metrics'] = {
                    'avg_processing_time': np.mean(valid_times),
                    'fastest_model': model_names[np.argmin(processing_times)] if min(processing_times) > 0 else 'N/A',
                    'slowest_model': model_names[np.argmax(processing_times)] if max(processing_times) > 0 else 'N/A',
                    'total_models_compared': len(model_names)
                }
            else:
                comparison_results['metrics'] = {
                    'avg_processing_time': 0,
                    'fastest_model': 'N/A',
                    'slowest_model': 'N/A',
                    'total_models_compared': len(model_names)
                }

        return comparison_results

    except Exception as e:
        print(f"Error in model comparison: {str(e)}")
        return None

def create_comparison_visualization(comparison_results):
    """
    Create visualizations for model comparison results.

    Args:
        comparison_results: Results from compare_models_on_image

    Returns:
        Dictionary with visualization data
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        visualizations = {}

        if not comparison_results or 'model_results' not in comparison_results:
            return visualizations

        # Extract data for plotting
        model_names = [r['model_name'] for r in comparison_results['model_results']]
        processing_times = [r['processing_time'] for r in comparison_results['model_results']]
        num_detections = [r['num_detections'] for r in comparison_results['model_results']]
        avg_confidences = [r['avg_confidence'] for r in comparison_results['model_results']]

        # Processing time comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(model_names, processing_times, color='skyblue')
        ax.set_title('Model Processing Time Comparison')
        ax.set_xlabel('Model')
        ax.set_ylabel('Processing Time (seconds)')
        plt.xticks(rotation=45, ha='right')

        # Add value labels on bars
        for bar, time in zip(bars, processing_times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                   '.3f', ha='center', va='bottom')

        plt.tight_layout()
        visualizations['processing_time_chart'] = fig

        # Detections comparison
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        bars2 = ax2.bar(model_names, num_detections, color='lightgreen')
        ax2.set_title('Number of Detections Comparison')
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Number of Detections')
        plt.xticks(rotation=45, ha='right')

        # Add value labels on bars
        for bar, det in zip(bars2, num_detections):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   str(int(det)), ha='center', va='bottom')

        plt.tight_layout()
        visualizations['detections_chart'] = fig2

        # Confidence comparison (only for models with detections)
        models_with_detections = [(name, conf) for name, conf, det in zip(model_names, avg_confidences, num_detections) if det > 0]
        if models_with_detections:
            names, confs = zip(*models_with_detections)
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            bars3 = ax3.bar(names, confs, color='orange')
            ax3.set_title('Average Confidence Comparison')
            ax3.set_xlabel('Model')
            ax3.set_ylabel('Average Confidence (%)')
            plt.xticks(rotation=45, ha='right')

            # Add value labels on bars
            for bar, conf in zip(bars3, confs):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       '.1f', ha='center', va='bottom')

            plt.tight_layout()
            visualizations['confidence_chart'] = fig3

        return visualizations

    except Exception as e:
        print(f"Error creating comparison visualizations: {str(e)}")
        return {}

def generate_comparison_report(comparison_results):
    """
    Generate a detailed comparison report.

    Args:
        comparison_results: Results from compare_models_on_image

    Returns:
        Dictionary with report data
    """
    try:
        if not comparison_results or 'model_results' not in comparison_results:
            return None

        report = {
            'summary': {
                'total_models': len(comparison_results['model_results']),
                'image_size': comparison_results['image_info']['size'],
                'image_mode': comparison_results['image_info']['mode']
            },
            'performance_metrics': comparison_results['metrics'],
            'model_details': []
        }

        for result in comparison_results['model_results']:
            model_detail = {
                'model_name': result['model_name'],
                'performance': {
                    'processing_time': '.3f',
                    'num_detections': result['num_detections'],
                    'avg_confidence': '.2f'
                },
                'detections_summary': {
                    'total_detections': result['num_detections'],
                    'damage_types': {}
                }
            }

            # Summarize damage types
            if result['detections']:
                damage_types = {}
                for det in result['detections']:
                    damage_type = det.get('damage_type', 'Unknown')
                    damage_types[damage_type] = damage_types.get(damage_type, 0) + 1
                model_detail['detections_summary']['damage_types'] = damage_types

            report['model_details'].append(model_detail)

        return report

    except Exception as e:
        print(f"Error generating comparison report: {str(e)}")
        return None
