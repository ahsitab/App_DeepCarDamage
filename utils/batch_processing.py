import os
import pandas as pd
from PIL import Image
import streamlit as st
from utils.preprocessing import preprocess_image
from utils.yolo_utils import load_yolo_model, detect_objects
from utils.export_utils import create_damage_report, export_results_to_csv
import time

def process_batch_images(image_files, model_path, conf_threshold=0.5, progress_callback=None):
    """
    Process multiple images in batch mode.

    Args:
        image_files: List of image file paths
        model_path: Path to the YOLO model
        conf_threshold: Confidence threshold for detections
        progress_callback: Optional callback function for progress updates

    Returns:
        Dictionary with batch results
    """
    try:
        # Load model once for all images
        model = load_yolo_model(model_path)

        batch_results = {
            'total_images': len(image_files),
            'processed_images': 0,
            'successful_detections': 0,
            'total_detections': 0,
            'image_results': [],
            'processing_time': 0
        }

        start_time = time.time()

        for i, image_path in enumerate(image_files):
            try:
                # Load and preprocess image
                image = Image.open(image_path)
                processed_image = preprocess_image(image, "YOLO Detection")

                # Run detection
                detections = detect_objects(model, processed_image, conf_threshold)

                # Create damage report
                model_info = {'filename': os.path.basename(model_path)}
                report = create_damage_report(detections, image_path, model_info)

                # Store results
                image_result = {
                    'image_path': image_path,
                    'image_name': os.path.basename(image_path),
                    'detections': detections,
                    'num_detections': len(detections),
                    'report': report,
                    'status': 'success'
                }

                batch_results['image_results'].append(image_result)
                batch_results['processed_images'] += 1

                if len(detections) > 0:
                    batch_results['successful_detections'] += 1
                    batch_results['total_detections'] += len(detections)

                # Update progress
                if progress_callback:
                    progress = (i + 1) / len(image_files)
                    progress_callback(progress, f"Processing {os.path.basename(image_path)}...")

            except Exception as e:
                # Handle individual image errors
                error_result = {
                    'image_path': image_path,
                    'image_name': os.path.basename(image_path),
                    'detections': [],
                    'num_detections': 0,
                    'report': None,
                    'status': 'error',
                    'error_message': str(e)
                }
                batch_results['image_results'].append(error_result)
                batch_results['processed_images'] += 1

                if progress_callback:
                    progress = (i + 1) / len(image_files)
                    progress_callback(progress, f"Error processing {os.path.basename(image_path)}: {str(e)}")

        batch_results['processing_time'] = time.time() - start_time

        return batch_results

    except Exception as e:
        print(f"Error in batch processing: {str(e)}")
        return None

def create_batch_summary_report(batch_results):
    """
    Create a summary report for batch processing results.

    Args:
        batch_results: Results from batch processing

    Returns:
        Dictionary with summary statistics
    """
    try:
        if not batch_results or 'image_results' not in batch_results:
            return None

        # Calculate summary statistics
        successful_images = [r for r in batch_results['image_results'] if r['status'] == 'success']
        error_images = [r for r in batch_results['image_results'] if r['status'] == 'error']

        # Damage type distribution
        damage_distribution = {}
        severity_scores = []

        for result in successful_images:
            if result['report']:
                report = result['report']
                severity_scores.append(report.get('severity_score', 0))

                for damage_type, count in report.get('damage_breakdown', {}).items():
                    damage_distribution[damage_type] = damage_distribution.get(damage_type, 0) + count

        # Calculate averages
        avg_detections = sum(r['num_detections'] for r in successful_images) / len(successful_images) if successful_images else 0
        avg_severity = sum(severity_scores) / len(severity_scores) if severity_scores else 0

        summary = {
            'total_images_processed': batch_results['processed_images'],
            'successful_images': len(successful_images),
            'error_images': len(error_images),
            'images_with_detections': batch_results['successful_detections'],
            'total_detections': batch_results['total_detections'],
            'average_detections_per_image': round(avg_detections, 2),
            'average_severity_score': round(avg_severity, 2),
            'damage_type_distribution': damage_distribution,
            'processing_time_seconds': round(batch_results['processing_time'], 2),
            'processing_time_per_image': round(batch_results['processing_time'] / batch_results['processed_images'], 2) if batch_results['processed_images'] > 0 else 0,
            'error_rate': round(len(error_images) / batch_results['processed_images'] * 100, 2) if batch_results['processed_images'] > 0 else 0
        }

        return summary

    except Exception as e:
        print(f"Error creating batch summary: {str(e)}")
        return None

def export_batch_results(batch_results, output_dir="batch_results"):
    """
    Export batch processing results to files.

    Args:
        batch_results: Results from batch processing
        output_dir: Directory to save results

    Returns:
        Dictionary with export file paths
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        export_files = {}

        # Export individual image results
        for result in batch_results['image_results']:
            if result['status'] == 'success' and result['detections']:
                # Export CSV for each image
                csv_content = export_results_to_csv(
                    result['detections'],
                    result['image_path'],
                    {'filename': 'batch_processing'}
                )

                if csv_content:
                    csv_filename = f"{os.path.splitext(result['image_name'])[0]}_results.csv"
                    csv_path = os.path.join(output_dir, csv_filename)

                    with open(csv_path, 'w') as f:
                        f.write(csv_content)

                    export_files[result['image_name']] = {
                        'csv_path': csv_path,
                        'num_detections': result['num_detections']
                    }

        # Export summary report
        summary = create_batch_summary_report(batch_results)
        if summary:
            summary_df = pd.DataFrame([summary])
            summary_path = os.path.join(output_dir, "batch_summary.csv")
            summary_df.to_csv(summary_path, index=False)
            export_files['batch_summary'] = {'csv_path': summary_path}

        return export_files

    except Exception as e:
        print(f"Error exporting batch results: {str(e)}")
        return None

def validate_image_files(file_paths):
    """
    Validate that files are valid image files.

    Args:
        file_paths: List of file paths to validate

    Returns:
        Tuple of (valid_files, invalid_files)
    """
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    valid_files = []
    invalid_files = []

    for file_path in file_paths:
        if os.path.exists(file_path):
            _, ext = os.path.splitext(file_path.lower())
            if ext in valid_extensions:
                try:
                    # Try to open with PIL to verify it's a valid image
                    with Image.open(file_path) as img:
                        img.verify()
                    valid_files.append(file_path)
                except Exception:
                    invalid_files.append(file_path)
            else:
                invalid_files.append(file_path)
        else:
            invalid_files.append(file_path)

    return valid_files, invalid_files
