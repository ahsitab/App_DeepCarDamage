import streamlit as st
import os
import time
from PIL import Image
import numpy as np

# --- PERFORMANCE OPTIMIZATION: ENV SETTINGS ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="DeepCar Analytics",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- INITIALIZATION ---
if 'reset_id' not in st.session_state:
    st.session_state['reset_id'] = 0

# --- CONFIGURATION ---
MODELS_DIR = "Models_Weight_files"
DEFAULT_YOLO = "best(1)yolov12.pt"
DEFAULT_CNN = "mobilenetv2_best.h5"
LOGO_PATH = "assets/logo.png"

# --- CUSTOM CSS FOR MODERN UI ---
def apply_custom_css():
    st.markdown("""
    <style>
        .stApp {
            background-color: #0d1117;
            color: #e6edf3;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        .glass-card {
            background: rgba(22, 27, 34, 0.85);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.2rem;
        }
        
        .main-header {
            color: #3b82f6;
            font-size: 3rem;
            font-weight: 800;
            text-align: center;
            margin-bottom: 0px;
        }
        
        .sub-header-text {
            text-align: center;
            color: #8b949e;
            font-size: 1.1rem;
            margin-bottom: 2rem;
        }
        
        [data-testid="stSidebar"] {
            background-color: #010409;
        }
        
        .severity-badge {
            display: inline-block;
            padding: 0.5rem 1rem;
            border-radius: 50px;
            font-weight: 700;
            text-transform: uppercase;
            font-size: 0.8rem;
            color: white;
        }
        .minor { background: #238636; }
        .moderate { background: #9e6a03; }
        .severe { background: #da3633; }
    </style>
    """, unsafe_allow_html=True)

apply_custom_css()

# ---SIDEBAR & NAVIGATION ---
with st.sidebar:
    try:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, width=220)
        else:
            st.markdown('<div class="main-header" style="font-size: 1.8rem; text-align: left;">DeepCar</div>', unsafe_allow_html=True)
    except Exception:
        st.markdown('<div class="main-header" style="font-size: 1.8rem; text-align: left;">DeepCar</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.info("⚡ Parallel Mode Active")
    
    # SYSTEM INTEGRITY CHECK
    y_path = os.path.join(MODELS_DIR, DEFAULT_YOLO)
    c_path = os.path.join(MODELS_DIR, DEFAULT_CNN)
    y_ok = os.path.exists(y_path)
    c_ok = os.path.exists(c_path)
    
    with st.expander("🛡️ System Integrity", expanded=False):
        st.markdown(f"**YOLO File:** {'✅' if y_ok else '❌'}")
        st.markdown(f"**CNN File:** {'✅' if c_ok else '❌'}")
        if not (y_ok and c_ok):
            st.error("Weights missing or incorrect filename.")
            st.caption(f"Path: {MODELS_DIR}")
    
    st.markdown("### 📷 Analysis Input")
    uploaded_file = st.file_uploader("Upload Target", type=["jpg", "jpeg", "png"], key=f"uploader_{st.session_state.reset_id}")
    
    sample_dir = "assets"
    samples = []
    if os.path.exists(sample_dir):
        samples = [f for f in os.listdir(sample_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    selected_sample = st.selectbox("Sample Selector", ["None"] + samples, key=f"sample_{st.session_state.reset_id}")
        
    st.markdown("---")
    st.caption("DeepCar Analytics v2.0.7")

# --- HEADER ---
st.markdown('<div class="main-header">DeepCar Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header-text">Unified Structural Localization & Severity Pipeline</div>', unsafe_allow_html=True)

# --- APP TABS ---
tab1, tab2, tab3 = st.tabs(["🔍 Analysis Terminal", "📚 Multiple Image Analysis", "📊 System Performance"])

# --- CACHED MODEL LOADERS ---
@st.cache_resource
def get_yolo_model(path):
    try:
        from utils.yolo_utils import load_yolo_model
        return load_yolo_model(path)
    except Exception as e:
        st.sidebar.error(f"Failed to load YOLO: {str(e)}")
        return None

@st.cache_resource
def get_cnn_model(path):
    try:
        from utils.inference import load_model
        return load_model(path)
    except Exception as e:
        st.sidebar.error(f"Failed to load CNN: {str(e)}")
        return None

# --- CORE ANALYSIS LOGIC ---
def run_unified_inference(image):
    import numpy as np
    import cv2
    from utils.preprocessing import preprocess_image
    from utils.yolo_utils import detect_objects
    from utils.inference import predict_classification
    
    # Convert PIL to BGR for OpenCV compatibility
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    yolo_path = os.path.join(MODELS_DIR, DEFAULT_YOLO)
    cnn_path = os.path.join(MODELS_DIR, DEFAULT_CNN)
    
    # 1. Detection Stream
    yolo_model = get_yolo_model(yolo_path)
    if yolo_model is not None:
        # Lower threshold to 0.1 for initial Cloud verification
        detections = detect_objects(yolo_model, image_np, 0.10)
        st.sidebar.write(f"Raw Detections: {len(detections)}")
    else:
        detections = []
        st.sidebar.warning("YOLO Engine Unavailable")
    
    # 2. Classification Stream
    cnn_model = get_cnn_model(cnn_path)
    if cnn_model is not None:
        processed_cnn = preprocess_image(image, "CNN Classification")
        predictions = predict_classification(cnn_model, processed_cnn)
    else:
        predictions = [{'class': 'Unknown', 'confidence': 0.0}]
        st.sidebar.warning("CNN Engine Unavailable")
    
    return detections, predictions, image_np

# --- TAB 1: SINGLE ANALYSIS ---
with tab1:
    from PIL import Image
    target_image = None
    if uploaded_file is not None:
        target_image = Image.open(uploaded_file)
    elif selected_sample != "None":
        sample_dir = "assets"
        target_image = Image.open(os.path.join(sample_dir, selected_sample))

    col_view, col_diag = st.columns([1.2, 1], gap="large")
    
    with col_view:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🔘 Visual Feed")
        if target_image:
            st.image(target_image, caption="Current Target", width=450)
            
            # Action Buttons Row
            bc1, bc2 = st.columns(2)
            with bc1:
                if st.button("🚀 EXECUTE FULL SCAN", use_container_width=True):
                    with st.spinner("Analyzing structural integrity..."):
                        detections, predictions, image_np = run_unified_inference(target_image)
                        st.session_state['parallel_results'] = (detections, predictions, image_np)
            with bc2:
                if st.button("🆕 NEW ANALYSIS", use_container_width=True):
                    # FULL RESET: Re-instantiate input widgets by changing their ID
                    if 'parallel_results' in st.session_state:
                        del st.session_state['parallel_results']
                    st.session_state['reset_id'] += 1
                    st.rerun()
        else:
            st.info("Awaiting input image...")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_diag:
        if 'parallel_results' in st.session_state and target_image:
            detections, predictions, image_np = st.session_state['parallel_results']
            
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 📡 Analytic Results")
            
            # PARALLEL VIEW: CNN | YOLO SUMMARY
            col_sev, col_det = st.columns(2)
            
            with col_sev:
                st.markdown("##### 🔬 Severity")
                top_class = predictions[0]['class']
                sc_key = top_class.lower().split('-')[-1]
                st.markdown(f"<div class='severity-badge {sc_key}' style='width: 100%; text-align: center;'>{top_class}</div>", unsafe_allow_html=True)
                
            with col_det:
                st.markdown("##### 📦 Detections")
                st.metric("Total Damage", len(detections))
            
            st.markdown("---")
            
            # DAMAGE LOG (Interchanged to appear before map)
            st.markdown("##### 📝 Damage Log")
            if detections:
                import pandas as pd
                st.dataframe(pd.DataFrame([{
                    'Damage Type': d['damage_type'], 
                    'Confidence': f"{d['confidence']:.1f}%"
                } for d in detections]), use_container_width=True)
            else:
                st.info("No structural anomalies logged.")

            # LOCALIZATION MAP (Interchanged to appear after log)
            st.markdown("##### 🔍 Localization Map")
            if detections:
                from utils.yolo_utils import draw_bounding_boxes
                import cv2
                # image_np is in BGR from run_unified_inference
                annotated = draw_bounding_boxes(image_np, detections)
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), width=450)
                
                # EXPORT WIDGETS
                st.markdown("##### 📥 Export Data")
                from utils.export_utils import export_results_to_csv, export_results_to_json
                ec1, ec2 = st.columns(2)
                fname = uploaded_file.name if uploaded_file else selected_sample
                with ec1:
                    st.download_button("📊 CSV", data=export_results_to_csv(detections, fname, {'m': 'unified'}), file_name=f"report.csv", use_container_width=True)
                with ec2:
                    st.download_button("📋 JSON", data=export_results_to_json(detections, fname, {'m': 'unified'}), file_name=f"report.json", use_container_width=True)
            else:
                st.warning("No localization data available for this target.")
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="glass-card" style="height: 300px; display: flex; align-items: center; justify-content: center; flex-direction: column; text-align: center;"><h4>System Standby</h4><p>Execute scan to populate parallel diagnostics</p></div>', unsafe_allow_html=True)

# --- OTHER TABS ---
# --- TAB 2: MULTIPLE IMAGE ANALYSIS ---
with tab2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### 📚 Multiple Image Analysis")
    st.markdown("Upload multiple images to perform a unified structural scan across the entire batch.")
    
    batch_files = st.file_uploader("Select Batch Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key=f"batch_uploader_{st.session_state.reset_id}")
    
    if batch_files:
        st.markdown(f"**Batch Size:** {len(batch_files)} images selected.")
        
        # Batch Action Buttons
        bcol1, bcol2 = st.columns(2)
        with bcol1:
            if st.button("🚀 EXECUTE BATCH ANALYSIS", use_container_width=True):
                batch_results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, file in enumerate(batch_files):
                    status_text.text(f"Processing {file.name}... ({i+1}/{len(batch_files)})")
                    from PIL import Image
                    img = Image.open(file)
                    
                    # Run unified inference
                    detections, predictions, image_np = run_unified_inference(img)
                    
                    # Collect summary data
                    batch_results.append({
                        'Filename': file.name,
                        'Severity': predictions[0]['class'],
                        'Confidence': f"{predictions[0]['confidence']:.1f}%",
                        'Total Detections': len(detections),
                        'detections': detections,
                        'image_np': image_np
                    })
                    progress_bar.progress((i + 1) / len(batch_files))
                
                st.session_state['batch_analysis_results'] = batch_results
                status_text.success(f"Batch Analysis Complete! Processed {len(batch_files)} targets.")
        
        with bcol2:
            if st.button("🆕 NEW BATCH ANALYSIS", use_container_width=True):
                if 'batch_analysis_results' in st.session_state:
                    del st.session_state['batch_analysis_results']
                st.session_state['reset_id'] += 1
                st.rerun()
    
    # Display Summary Table if results exist
    if 'batch_analysis_results' in st.session_state:
        results = st.session_state['batch_analysis_results']
        import pandas as pd
        summary_df = pd.DataFrame([{
            'Filename': r['Filename'],
            'Severity Category': r['Severity'],
            'Score': r['Confidence'],
            'Anomalies': r['Total Detections']
        } for r in results])
        
        st.markdown("---")
        st.markdown("#### 📊 Batch Summary Report")
        st.dataframe(summary_df, use_container_width=True)
        
        # Detailed Inspector
        st.markdown("---")
        st.markdown("#### 🔍 Detail Inspector")
        inspect_file = st.selectbox("Select file to inspect:", [r['Filename'] for r in results])
        
        if inspect_file:
            selected_res = next(r for r in results if r['Filename'] == inspect_file)
            icol1, icol2 = st.columns(2)
            
            with icol1:
                from utils.yolo_utils import draw_bounding_boxes
                import cv2
                ann = draw_bounding_boxes(selected_res['image_np'], selected_res['detections'])
                st.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), caption=f"Localization: {inspect_file}", width=450)
            
            with icol2:
                st.markdown(f"##### 📋 Diagnostic Summary")
                st.write(f"**Target:** `{inspect_file}`")
                st.write(f"**Unified Severity:** {selected_res['Severity']}")
                st.write(f"**Total Structural Anomalies:** `{selected_res['Total Detections']}`")
                
                if selected_res['detections']:
                    import pandas as pd
                    inspector_df = pd.DataFrame([{
                        'Damage Type': d['damage_type'], 
                        'Confidence': f"{d['confidence']:.1f}%"
                    } for d in selected_res['detections']])
                    st.dataframe(inspector_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No anomalies detected for this target.")
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="glass-card"><h4>📊 System Performance</h4><p>Real-time neural telemetry and model latency monitoring (Coming Soon).</p></div>', unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("---")
st.markdown('<p style="text-align: center; color: #484f58; font-size: 0.8rem;">DEEPCAR ANALYTICS ECOSYSTEM | v2.0.4 | STABLE</p>', unsafe_allow_html=True)
