import streamlit as st
import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import cv2
from utils.preprocessing import preprocess_image
from utils.inference import load_model, predict_classification, predict_detection, TENSORFLOW_AVAILABLE
from utils.xai import generate_xai_explanations
from utils.yolo_utils import load_yolo_model, detect_objects, draw_bounding_boxes
from utils.export_utils import export_results_to_csv, export_results_to_json, create_damage_report
from utils.batch_processing import process_batch_images, create_batch_summary_report, validate_image_files
from utils.model_comparison import compare_models_on_image

# Set page configuration
st.set_page_config(
    page_title="DeepCarAnalysis",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for dark mode
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

def toggle_dark_mode():
    st.session_state.dark_mode = not st.session_state.dark_mode
    st.rerun()

# Custom CSS for dark mode and styling
def get_css():
    if st.session_state.dark_mode:
        return """
        <style>
            .main-header {
                font-size: 2.5rem;
                font-weight: bold;
                background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                text-align: center;
                margin-bottom: 2rem;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }
            .sub-header {
                font-size: 1.5rem;
                font-weight: bold;
                background: linear-gradient(45deg, #FF9800, #FF5722, #FFC107);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                margin-bottom: 1rem;
            }
            .card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 1rem;
                border-radius: 0.5rem;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                margin-bottom: 1rem;
                color: white;
                border: 1px solid rgba(255,255,255,0.1);
            }
            .prediction-box {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 1rem;
                border-radius: 0.5rem;
                border-left: 4px solid #FF6B6B;
                margin-bottom: 1rem;
                color: white;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            }
            body {
                background: linear-gradient(135deg, #0F0C29 0%, #302B63 50%, #24243e 100%);
                color: white;
            }
            .stTabs [data-baseweb="tab-list"] {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 0.5rem;
            }
            .stTabs [data-baseweb="tab"] {
                color: white;
                font-weight: bold;
            }
            .stButton>button {
                background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
                color: white;
                border: none;
                border-radius: 0.5rem;
                font-weight: bold;
                transition: all 0.3s ease;
            }
            .stButton>button:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 15px rgba(255,107,107,0.4);
            }
            .stMetric {
                background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
                border-radius: 0.5rem;
                padding: 1rem;
                border: 1px solid rgba(255,255,255,0.1);
            }
        </style>
        """
    else:
        return """
        <style>
            .main-header {
                font-size: 2.5rem;
                font-weight: bold;
                background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4, #FFEAA7, #DDA0DD);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                text-align: center;
                margin-bottom: 2rem;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
            }
            .sub-header {
                font-size: 1.5rem;
                font-weight: bold;
                background: linear-gradient(45deg, #FF9800, #FF5722, #FFC107, #FFEB3B);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                margin-bottom: 1rem;
            }
            .card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 1rem;
                border-radius: 0.5rem;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                margin-bottom: 1rem;
                color: white;
                border: 1px solid rgba(255,255,255,0.2);
            }
            .prediction-box {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 1rem;
                border-radius: 0.5rem;
                border-left: 4px solid #FF6B6B;
                margin-bottom: 1rem;
                color: white;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            }
            body {
                background: linear-gradient(135deg, #74b9ff 0%, #0984e3 50%, #00cec9 100%);
                color: #2d3436;
            }
            .stTabs [data-baseweb="tab-list"] {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 0.5rem;
            }
            .stTabs [data-baseweb="tab"] {
                color: white;
                font-weight: bold;
            }
            .stButton>button {
                background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
                color: white;
                border: none;
                border-radius: 0.5rem;
                font-weight: bold;
                transition: all 0.3s ease;
            }
            .stButton>button:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 15px rgba(255,107,107,0.4);
            }
            .stMetric {
                background: linear-gradient(135deg, rgba(255,255,255,0.8) 0%, rgba(255,255,255,0.6) 100%);
                border-radius: 0.5rem;
                padding: 1rem;
                border: 1px solid rgba(0,0,0,0.1);
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .stDataFrame {
                border-radius: 0.5rem;
                overflow: hidden;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
        </style>
        """

st.markdown(get_css(), unsafe_allow_html=True)

# Main title
st.markdown('<div class="main-header">🚗 DeepCarAnalysis</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Vehicle Detection & Classification System</p>', unsafe_allow_html=True)

# Dark mode toggle
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("🌙 Dark Mode" if not st.session_state.dark_mode else "☀️ Light Mode"):
        toggle_dark_mode()

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Single Image Analysis", "📦 Batch Processing", "⚖️ Model Comparison", "📊 Performance Dashboard"])

# Sidebar
st.sidebar.title("🚗 DeepCarAnalysis")
st.sidebar.markdown("---")

# Model selection
st.sidebar.markdown("### Model Selection")
models_dir = "Models_Weight_files"

try:
    if os.path.exists(models_dir):
        cnn_models = [f for f in os.listdir(models_dir) if f.endswith('.h5')]
        yolo_models = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
    else:
        st.sidebar.error(f"⚠️ Models directory '{models_dir}' not found!")
        st.sidebar.info("Please create the Models_Weight_files directory and add your model files.")
        cnn_models = []
        yolo_models = []
except Exception as e:
    st.sidebar.error(f"⚠️ Error loading models: {str(e)}")
    cnn_models = []
    yolo_models = []

available_options = ["YOLO Detection"]
if TENSORFLOW_AVAILABLE and cnn_models:
    available_options.insert(0, "CNN Classification")

model_type = st.sidebar.radio("Select Analysis Type:", available_options)

if not TENSORFLOW_AVAILABLE and model_type == "CNN Classification":
    st.sidebar.error("⚠️ TensorFlow is not available. CNN Classification is disabled.")
    st.sidebar.info("Please install TensorFlow to enable CNN classification features.")

if model_type == "CNN Classification":
    selected_model = st.sidebar.selectbox("Select CNN Model:", cnn_models if cnn_models else ["No models available"])
    if selected_model != "No models available":
        model_path = os.path.join(models_dir, selected_model)
        model_name = selected_model.replace('_best.h5', '').replace('.h5', '').upper()

        # Model info
        st.sidebar.markdown("### Model Information")
        st.sidebar.write(f"**Architecture:** {model_name}")
        st.sidebar.write(f"**Input Size:** 224x224")
        st.sidebar.write(f"**Classes:** 3 (Minor, Moderate, Severe)")

elif model_type == "YOLO Detection":
    selected_model = st.sidebar.selectbox("Select YOLO Model:", yolo_models if yolo_models else ["No models available"])
    if selected_model != "No models available":
        model_path = os.path.join(models_dir, selected_model)
        model_name = selected_model.replace('best', '').replace('.pt', '').replace('(', '').replace(')', '').upper()

        # Model info
        st.sidebar.markdown("### Model Information")
        st.sidebar.write(f"**Architecture:** {model_name}")
        st.sidebar.write(f"**Input Size:** 640x640")
        st.sidebar.write(f"**Task:** Object Detection")
        st.sidebar.write(f"**Path:** {model_path}")

# Image upload
st.sidebar.markdown("---")
st.sidebar.markdown("### Image Input")
uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
sample_images = ["sample1.jpg", "sample2.jpg", "sample3.jpg"]
selected_sample = st.sidebar.selectbox("Or select sample image:", ["None"] + sample_images)

# Single Image Analysis Tab
with tab1:
    st.markdown('<div class="sub-header">📷 Image Preview</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
    elif selected_sample != "None":
        image_path = os.path.join("assets", selected_sample)
        if os.path.exists(image_path):
            image = Image.open(image_path)
            st.image(image, caption="Sample Image", use_container_width=True)
        else:
            st.warning("Sample image not found.")
    else:
        st.info("Please upload an image or select a sample.")

    # Analysis section after image preview
    if 'image' in locals():
        st.markdown('<div class="sub-header">🔍 Analysis Results</div>', unsafe_allow_html=True)

        # Preprocess image
        processed_image = preprocess_image(image, model_type)

        if model_type == "CNN Classification" and TENSORFLOW_AVAILABLE:
            # Load model
            model = load_model(model_path)

            # Make prediction
            predictions = predict_classification(model, processed_image)

            # Display results
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown("### Classification Results")

            # Top prediction
            top_class = predictions[0]['class']
            top_conf = predictions[0]['confidence']
            st.success(f"**Predicted Class:** {top_class} ({top_conf:.2f}%)")

            # All predictions
            st.markdown("#### All Predictions:")
            for pred in predictions:
                st.markdown(f"- {pred['class']}: {pred['confidence']:.2f}%")
                st.progress(pred['confidence'] / 100)

            st.markdown('</div>', unsafe_allow_html=True)

            # XAI Explanations
            if st.checkbox("Show Explainable AI (XAI)"):
                st.markdown("### Explainable AI")
                xai_methods = st.multiselect("Select XAI Methods:",
                                           ["Grad-CAM", "Grad-CAM++", "Eigen-CAM"],
                                           default=["Grad-CAM"])

                if xai_methods:
                    explanations = generate_xai_explanations(model, processed_image, xai_methods)

                    # Display explanations in grid
                    cols = st.columns(len(explanations))
                    for i, (method, exp_img) in enumerate(explanations.items()):
                        with cols[i]:
                            st.image(exp_img, caption=f"{method} Explanation", use_container_width=True)

        elif model_type == "YOLO Detection":
            # Load model
            model = load_yolo_model(model_path)

            # Detection parameters
            conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)

            # Make prediction
            detections = detect_objects(model, processed_image, conf_threshold)

            # Display results
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown("### 🚗 Car Damage Detection Results")

            if detections:
                # Overall assessment
                damage_counts = {}
                total_damage_area = 0

                for det in detections:
                    damage_type = det['damage_type']
                    damage_counts[damage_type] = damage_counts.get(damage_type, 0) + 1
                    total_damage_area += det['area']

                # Display summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Damages", len(detections))
                with col2:
                    most_common = max(damage_counts, key=damage_counts.get)
                    st.metric("Primary Damage", most_common)
                with col3:
                    severity_score = sum(1 if d['damage_type'] == 'Severe Damage' else 0.5 if d['damage_type'] == 'Moderate Damage' else 0.2 for d in detections)
                    st.metric("Severity Score", f"{severity_score:.1f}")

                # Damage breakdown
                st.markdown("#### 📊 Damage Analysis")
                damage_df = []
                for det in detections:
                    damage_df.append({
                        'Damage Type': det['damage_type'],
                        'Specific Class': det['class'],
                        'Confidence': f"{det['confidence']:.1f}%",
                        'Area': f"{det['area']:.0f}px²"
                    })

                st.dataframe(damage_df, use_container_width=True)

                # Display image with bounding boxes
                annotated_image = draw_bounding_boxes(processed_image, detections)
                # Convert BGR to RGB for Streamlit
                annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                st.image(annotated_image_rgb, caption="🔍 Detected Damages", use_container_width=True)

                # Export options
                st.markdown("#### 📥 Export Results")
                col1, col2 = st.columns(2)
                with col1:
                    csv_data = export_results_to_csv(detections, uploaded_file.name if uploaded_file else selected_sample, {'filename': selected_model})
                    if csv_data:
                        st.download_button(
                            label="📊 Download CSV",
                            data=csv_data,
                            file_name=f"{uploaded_file.name.split('.')[0] if uploaded_file else selected_sample.split('.')[0]}_analysis.csv",
                            mime="text/csv"
                        )
                with col2:
                    json_data = export_results_to_json(detections, uploaded_file.name if uploaded_file else selected_sample, {'filename': selected_model})
                    if json_data:
                        st.download_button(
                            label="📋 Download JSON",
                            data=json_data,
                            file_name=f"{uploaded_file.name.split('.')[0] if uploaded_file else selected_sample.split('.')[0]}_analysis.json",
                            mime="application/json"
                        )

            else:
                st.warning("⚠️ No damages detected with current threshold. Try lowering the confidence threshold.")

            st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.info("Upload an image to start analysis.")

# Batch Processing Tab
with tab2:
    st.markdown('<div class="sub-header">📦 Batch Image Processing</div>', unsafe_allow_html=True)
    st.markdown("Upload multiple images for batch analysis")

    batch_files = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if batch_files:
        st.success(f"📁 {len(batch_files)} images uploaded for batch processing")

        # Batch processing parameters
        col1, col2 = st.columns(2)
        with col1:
            batch_conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.1, key="batch_conf")
        with col2:
            batch_model = st.selectbox("Select Model for Batch Processing", yolo_models if yolo_models else ["No models available"], key="batch_model")

        if st.button("🚀 Start Batch Processing") and batch_model != "No models available":
            with st.spinner("Processing images..."):
                # Save uploaded files temporarily
                temp_dir = "temp_batch"
                os.makedirs(temp_dir, exist_ok=True)

                image_paths = []
                for uploaded_file in batch_files:
                    temp_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    image_paths.append(temp_path)

                # Process batch
                batch_results = process_batch_images(
                    image_paths,
                    os.path.join(models_dir, batch_model),
                    batch_conf_threshold
                )

                if batch_results:
                    # Display summary
                    st.success("✅ Batch processing completed!")

                    summary = create_batch_summary_report(batch_results)
                    if summary:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Images", summary['total_images_processed'])
                        with col2:
                            st.metric("Images with Detections", summary['images_with_detections'])
                        with col3:
                            st.metric("Total Detections", summary['total_detections'])
                        with col4:
                            st.metric("Avg Severity Score", f"{summary['average_severity_score']:.2f}")

                    # Display detailed results
                    st.markdown("#### 📋 Detailed Results")
                    results_df = []
                    for result in batch_results['image_results']:
                        results_df.append({
                            'Image': result['image_name'],
                            'Detections': result['num_detections'],
                            'Status': result['status'],
                            'Severity Score': result['report']['severity_score'] if result['report'] else 0
                        })

                    st.dataframe(results_df, use_container_width=True)

                    # Display processed images with detections
                    st.markdown("#### 🖼️ Processed Images")
                    cols = st.columns(2)  # Display 2 images per row

                    for i, result in enumerate(batch_results['image_results']):
                        if result['status'] == 'success' and result['detections']:
                            # Load original image
                            try:
                                original_image = Image.open(result['image_path'])
                                processed_image = preprocess_image(original_image, "YOLO Detection")

                                # Draw bounding boxes
                                model = load_yolo_model(os.path.join(models_dir, batch_model))
                                annotated_image = draw_bounding_boxes(processed_image, result['detections'])
                                annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

                                with cols[i % 2]:
                                    st.image(annotated_image_rgb,
                                           caption=f"{result['image_name']} - {result['num_detections']} detections",
                                           use_container_width=True)
                                    st.markdown(f"**Severity Score:** {result['report']['severity_score']:.1f}" if result['report'] else "**No Report**")

                            except Exception as e:
                                with cols[i % 2]:
                                    st.error(f"Error displaying {result['image_name']}: {str(e)}")

                    # Cleanup
                    import shutil
                    shutil.rmtree(temp_dir)
                else:
                    st.error("❌ Batch processing failed")

# Model Comparison Tab
with tab3:
    st.markdown('<div class="sub-header">⚖️ Model Comparison</div>', unsafe_allow_html=True)
    st.markdown("Compare different models on the same image")

    # Model selection for comparison
    available_models = yolo_models if yolo_models else []
    selected_models = st.multiselect("Select Models to Compare", available_models, max_selections=3)

    if len(selected_models) >= 2:
        # Image selection
        comp_uploaded_file = st.file_uploader("Upload Image for Comparison", type=["jpg", "jpeg", "png"], key="comp_upload")
        comp_sample = st.selectbox("Or select sample image:", ["None"] + sample_images, key="comp_sample")

        if comp_uploaded_file or comp_sample != "None":
            # Load image
            if comp_uploaded_file:
                comp_image = Image.open(comp_uploaded_file)
                image_name = comp_uploaded_file.name
            else:
                image_path = os.path.join("assets", comp_sample)
                if os.path.exists(image_path):
                    comp_image = Image.open(image_path)
                    image_name = comp_sample
                else:
                    st.error("Sample image not found")
                    comp_image = None

            if comp_image is not None:
                st.image(comp_image, caption="Comparison Image", width=300)

                if st.button("🔍 Compare Models"):
                    with st.spinner("Comparing models..."):
                        model_paths = [os.path.join(models_dir, model) for model in selected_models]
                        comparison_results = compare_models_on_image(comp_image, model_paths, selected_models)

                        if comparison_results:
                            # Display comparison results
                            st.markdown("#### 📊 Comparison Results")

                            comp_df = []
                            for result in comparison_results['model_results']:
                                comp_df.append({
                                    'Model': result['model_name'],
                                    'Detections': result['num_detections'],
                                    'Processing Time': f"{result['processing_time']:.3f}s",
                                    'Avg Confidence': f"{result['avg_confidence']:.2f}%"
                                })

                            st.dataframe(comp_df, use_container_width=True)

                            # Performance metrics
                            st.markdown("#### ⚡ Performance Metrics")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Fastest Model", comparison_results['metrics']['fastest_model'])
                            with col2:
                                st.metric("Slowest Model", comparison_results['metrics']['slowest_model'])
                            with col3:
                                st.metric("Avg Processing Time", f"{comparison_results['metrics']['avg_processing_time']:.3f}s")

                            # Display comparison visualizations
                            st.markdown("#### 📊 Model Comparison Visualizations")

                            # Display processed images for each model
                            st.markdown("##### 🖼️ Detection Results by Model")
                            for result in comparison_results['model_results']:
                                if result['detections']:
                                    st.markdown(f"**{result['model_name']}** - {result['num_detections']} detections")

                                    # Load and process image for this model
                                    try:
                                        processed_image = preprocess_image(comp_image, "YOLO Detection")
                                        model = load_yolo_model(result['model_path'])
                                        annotated_image = draw_bounding_boxes(processed_image, result['detections'])
                                        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

                                        st.image(annotated_image_rgb,
                                               caption=f"{result['model_name']} - {result['num_detections']} detections (Time: {result['processing_time']:.3f}s)",
                                               use_container_width=True)
                                    except Exception as e:
                                        st.error(f"Error displaying results for {result['model_name']}: {str(e)}")
                                else:
                                    st.markdown(f"**{result['model_name']}** - No detections found")
                        else:
                            st.error("Comparison failed")

# Performance Dashboard Tab
with tab4:
    st.markdown('<div class="sub-header">📊 Performance Dashboard</div>', unsafe_allow_html=True)

    # System info
    st.markdown("#### 🖥️ System Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Models Available", len(cnn_models) + len(yolo_models))
    with col2:
        st.metric("CNN Models", len(cnn_models))
    with col3:
        st.metric("YOLO Models", len(yolo_models))

    # Model details
    if cnn_models or yolo_models:
        st.markdown("#### 📋 Available Models")

        model_info = []
        for model in cnn_models:
            model_info.append({
                'Type': 'CNN',
                'Name': model.replace('_best.h5', '').replace('.h5', '').upper(),
                'File': model,
                'Size': f"{os.path.getsize(os.path.join(models_dir, model)) / (1024*1024):.2f} MB"
            })

        for model in yolo_models:
            model_info.append({
                'Type': 'YOLO',
                'Name': model.replace('best', '').replace('.pt', '').replace('(', '').replace(')', '').upper(),
                'File': model,
                'Size': f"{os.path.getsize(os.path.join(models_dir, model)) / (1024*1024):.2f} MB"
            })

        st.dataframe(model_info, use_container_width=True)

# Footer
st.markdown("---")
st.markdown('<p style="text-align: center; color: #666;">© 2024 DeepCarAnalysis - AI-Powered Vehicle Analysis</p>', unsafe_allow_html=True)
