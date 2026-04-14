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
DEFAULT_YOLO = "best_yolov12.pt"
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
    
    y_exists = os.path.exists(y_path)
    c_exists = os.path.exists(c_path)
    
    # LFS Pointer Check (Files < 1024 bytes are likely pointers)
    y_size = os.path.getsize(y_path) if y_exists else 0
    c_size = os.path.getsize(c_path) if c_exists else 0
    
    y_ok = y_exists and y_size > 1024
    c_ok = c_exists and c_size > 1024
    
    with st.expander("🛡️ System Integrity", expanded=not (y_ok and c_ok)):
        st.markdown(f"**YOLO Model:** {'✅' if y_ok else '❌'}")
        if y_exists and not y_ok:
            st.warning("YOLO weight file is too small. Possibly a Git LFS pointer.")
            
        st.markdown(f"**CNN Model:** {'✅' if c_ok else '❌'}")
        if c_exists and not c_ok:
            st.warning("CNN weight file is too small. Possibly a Git LFS pointer.")
            
        if not (y_exists and c_exists):
            st.error("Model files missing in deployment.")
            st.caption(f"Target Directory: {MODELS_DIR}")
    
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
    import time
    import numpy as np
    import cv2
    from utils.preprocessing import preprocess_image
    from utils.yolo_utils import detect_objects
    from utils.inference import predict_classification
    
    metrics = {}
    start_total = time.perf_counter()
    
    # Convert PIL to BGR for OpenCV compatibility
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    yolo_path = os.path.join(MODELS_DIR, DEFAULT_YOLO)
    cnn_path = os.path.join(MODELS_DIR, DEFAULT_CNN)
    
    # 1. Detection Stream
    yolo_model = get_yolo_model(yolo_path)
    if yolo_model is not None:
        start_yolo = time.perf_counter()
        detections = detect_objects(yolo_model, image_np, 0.10)
        metrics['yolo_latency'] = round((time.perf_counter() - start_yolo) * 1000, 2)  # ms
    else:
        detections = []
        metrics['yolo_latency'] = 0
    
    # 2. Classification Stream
    cnn_model = get_cnn_model(cnn_path)
    if cnn_model is not None:
        start_cnn = time.perf_counter()
        processed_cnn = preprocess_image(image, "CNN Classification")
        predictions = predict_classification(cnn_model, processed_cnn)
        metrics['cnn_latency'] = round((time.perf_counter() - start_cnn) * 1000, 2)  # ms
    else:
        predictions = [{'class': 'Unknown', 'confidence': 0.0}]
        metrics['cnn_latency'] = 0
    
    metrics['total_latency'] = round((time.perf_counter() - start_total) * 1000, 2)
    return detections, predictions, image_np, metrics

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
                        detections, predictions, image_np, metrics = run_unified_inference(target_image)
                        st.session_state['parallel_results'] = (detections, predictions, image_np)
                        st.session_state['last_metrics'] = metrics
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

            # EXPORT WIDGETS
            st.markdown("##### 📥 Export Data")
            from utils.export_utils import export_results_to_csv, export_results_to_json
            ec1, ec2 = st.columns(2)
            fname = uploaded_file.name if uploaded_file else selected_sample
            
            # Prepare models info for export
            m_info = {
                'yolo': DEFAULT_YOLO,
                'cnn': DEFAULT_CNN,
                'timestamp': time.strftime("%Y%m%d-%H%M%S")
            }
            
            with ec1:
                csv_data = export_results_to_csv(detections, fname, m_info, predictions)
                st.download_button(
                    "📊 CSV Report", 
                    data=csv_data if csv_data else "", 
                    file_name=f"DeepCar_Report_{m_info['timestamp']}.csv", 
                    use_container_width=True,
                    disabled=csv_data is None
                )
            with ec2:
                json_data = export_results_to_json(detections, fname, m_info, None, predictions)
                st.download_button(
                    "📋 JSON Report", 
                    data=json_data if json_data else "", 
                    file_name=f"DeepCar_Report_{m_info['timestamp']}.json", 
                    use_container_width=True,
                    disabled=json_data is None
                )
            
            st.markdown("---")
            
            # LOCALIZATION MAP
            st.markdown("##### 🔍 Localization Map")
            if detections:
                from utils.yolo_utils import draw_bounding_boxes
                import cv2
                # image_np is in BGR from run_unified_inference
                annotated = draw_bounding_boxes(image_np, detections)
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), width=450)
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
                    detections, predictions, image_np, metrics = run_unified_inference(img)
                    
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
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### 📊 System Performance & Neural Telemetry")
    
    from utils.system_utils import get_system_metrics, get_model_file_info
    sys_metrics = get_system_metrics()
    
    # 1. LIVE INFERENCE METRICS
    st.markdown("##### ⚡ Real-Time Latency")
    if 'last_metrics' in st.session_state:
        m = st.session_state['last_metrics']
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.metric("YOLO Detection", f"{m['yolo_latency']}ms")
        with mc2:
            st.metric("CNN Classification", f"{m['cnn_latency']}ms")
        with mc3:
            st.metric("Total Pipeline", f"{m['total_latency']}ms", delta_color="inverse")
    else:
        st.info("No inference data in current session. Execute a scan to view latency.")
    
    st.markdown("---")
    
    # 2. HARDWARE TELEMETRY
    st.markdown("##### 💻 Hardware Utilization")
    if 'error' not in sys_metrics:
        hc1, hc2, hc3 = st.columns(3)
        with hc1:
            st.metric("CPU Usage", f"{sys_metrics['cpu_usage']}%")
        with hc2:
            st.metric("RAM Available", f"{sys_metrics['ram_total'] - sys_metrics['ram_used']:.1f} GB", f"Used: {sys_metrics['ram_percent']}%", delta_color="inverse")
        with hc3:
            st.write(f"**OS:** {sys_metrics['os']}")
            st.write(f"**Core Count:** {sys_metrics['cpu_count']}")
    else:
        st.error(f"Hardware telemetry unavailable: {sys_metrics['error']}")
        
    st.markdown("---")
    
    # 3. MODEL REGISTRY
    st.markdown("##### 📦 Model Registry")
    models_info = get_model_file_info(MODELS_DIR, [DEFAULT_YOLO, DEFAULT_CNN])
    
    import pandas as pd
    m_df = pd.DataFrame(models_info)
    st.table(m_df[['name', 'size_mb', 'status']])
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("---")
st.markdown('<p style="text-align: center; color: #484f58; font-size: 0.8rem;">DEEPCAR ANALYTICS ECOSYSTEM | v2.0.4 | STABLE</p>', unsafe_allow_html=True)
