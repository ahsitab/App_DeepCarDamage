import pandas as pd
import json
from datetime import datetime
import os
from PIL import Image
import io
import base64

def export_results_to_csv(detections, image_path, model_info):
    """
    Export detection results to CSV format.

    Args:
        detections: List of detection results
        image_path: Path to the analyzed image
        model_info: Dictionary with model information

    Returns:
        CSV string content
    """
    try:
        # Prepare data for CSV
        csv_data = []
        for det in detections:
            csv_data.append({
                'Image': os.path.basename(image_path),
                'Damage_Type': det['damage_type'],
                'Specific_Class': det['class'],
                'Confidence': det['confidence'],
                'Area_px2': det['area'],
                'Bounding_Box': f"[{det['bbox'][0]}, {det['bbox'][1]}, {det['bbox'][2]}, {det['bbox'][3]}]",
                'Center_X': det['center'][0],
                'Center_Y': det['center'][1],
                'Model': model_info.get('filename', 'Unknown'),
                'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })

        df = pd.DataFrame(csv_data)
        return df.to_csv(index=False)
    except Exception as e:
        print(f"Error creating CSV: {str(e)}")
        return None

def export_results_to_json(detections, image_path, model_info, analysis_summary=None):
    """
    Export detection results to JSON format.

    Args:
        detections: List of detection results
        image_path: Path to the analyzed image
        model_info: Dictionary with model information
        analysis_summary: Optional summary statistics

    Returns:
        JSON string content
    """
    try:
        export_data = {
            'metadata': {
                'image': os.path.basename(image_path),
                'model': model_info.get('filename', 'Unknown'),
                'timestamp': datetime.now().isoformat(),
                'total_detections': len(detections)
            },
            'analysis_summary': analysis_summary or {},
            'detections': detections
        }

        return json.dumps(export_data, indent=2, default=str)
    except Exception as e:
        print(f"Error creating JSON: {str(e)}")
        return None

def create_damage_report(detections, image_path, model_info):
    """
    Create a comprehensive damage assessment report.

    Args:
        detections: List of detection results
        image_path: Path to the analyzed image
        model_info: Dictionary with model information

    Returns:
        Dictionary with report data
    """
    try:
        # Calculate summary statistics
        damage_counts = {}
        severity_score = 0
        total_area = 0

        for det in detections:
            damage_type = det['damage_type']
            damage_counts[damage_type] = damage_counts.get(damage_type, 0) + 1
            total_area += det['area']

            # Calculate severity score
            if damage_type == 'Severe Damage':
                severity_score += 1
            elif damage_type == 'Moderate Damage':
                severity_score += 0.5
            elif damage_type == 'Minor Damage':
                severity_score += 0.2

        # Determine overall assessment
        if severity_score >= 2:
            overall_assessment = "Critical Damage - Immediate Repair Required"
        elif severity_score >= 1:
            overall_assessment = "Moderate Damage - Repair Recommended"
        elif severity_score >= 0.5:
            overall_assessment = "Minor Damage - Cosmetic Repair"
        else:
            overall_assessment = "No Significant Damage"

        report = {
            'image_name': os.path.basename(image_path),
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_used': model_info.get('filename', 'Unknown'),
            'total_damages': len(detections),
            'damage_breakdown': damage_counts,
            'severity_score': round(severity_score, 2),
            'total_damage_area': round(total_area, 2),
            'overall_assessment': overall_assessment,
            'recommendations': generate_recommendations(damage_counts, severity_score)
        }

        return report
    except Exception as e:
        print(f"Error creating damage report: {str(e)}")
        return None

def generate_recommendations(damage_counts, severity_score):
    """
    Generate repair recommendations based on damage analysis.

    Args:
        damage_counts: Dictionary of damage type counts
        severity_score: Calculated severity score

    Returns:
        List of recommendations
    """
    recommendations = []

    if severity_score >= 2:
        recommendations.extend([
            "Immediate professional inspection required",
            "Vehicle may be unsafe to drive",
            "Contact insurance provider immediately",
            "Schedule comprehensive repair assessment"
        ])
    elif severity_score >= 1:
        recommendations.extend([
            "Schedule repair within 1-2 weeks",
            "Get multiple repair quotes",
            "Check for underlying structural damage",
            "Consider paintless dent repair for minor dents"
        ])
    elif severity_score >= 0.5:
        recommendations.extend([
            "Address cosmetic damage promptly",
            "Consider DIY repair for minor scratches",
            "Touch-up paint may be sufficient",
            "Monitor for rust development"
        ])
    else:
        recommendations.append("No immediate action required")

    # Specific recommendations based on damage types
    if damage_counts.get('Severe Damage', 0) > 0:
        recommendations.append("Structural integrity may be compromised - professional evaluation essential")

    if damage_counts.get('Moderate Damage', 0) > 2:
        recommendations.append("Multiple moderate damages suggest possible accident impact - full vehicle inspection recommended")

    return recommendations

def create_pdf_report(report_data, detections, image_path=None):
    """
    Create a PDF report (placeholder - would require reportlab or similar library).

    Args:
        report_data: Report dictionary
        detections: Detection results
        image_path: Optional path to include image in PDF

    Returns:
        Placeholder message (would return PDF bytes in full implementation)
    """
    # This is a placeholder - full PDF generation would require additional libraries
    # like reportlab, fpdf, or similar
    return "PDF generation requires additional libraries (reportlab/fpdf). Install with: pip install reportlab"

def encode_image_to_base64(image):
    """
    Encode PIL Image to base64 string for embedding in exports.

    Args:
        image: PIL Image object

    Returns:
        Base64 encoded string
    """
    try:
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        print(f"Error encoding image: {str(e)}")
        return None
